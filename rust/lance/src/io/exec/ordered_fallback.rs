// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow_array::RecordBatch;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    CustomMetricValue, ExecutionPlanMetricsSet, MetricBuilder, MetricValue, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{TryStreamExt, stream};
use lance_core::{Error, Result};

use super::utils::InstrumentedRecordBatchStreamAdapter;

#[derive(Debug)]
pub struct OrderedFallbackExec {
    ordered: Arc<dyn ExecutionPlan>,
    sorted: Arc<dyn ExecutionPlan>,
    limit: usize,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl OrderedFallbackExec {
    pub fn try_new(
        ordered: Arc<dyn ExecutionPlan>,
        sorted: Arc<dyn ExecutionPlan>,
        limit: usize,
    ) -> Result<Self> {
        if ordered.schema().fields() != sorted.schema().fields() {
            return Err(Error::internal(format!(
                "an ordered read falls back to a plan with its fields, but the ordered read \
                 returns {:?} and the sorted plan returns {:?}",
                ordered.schema().fields(),
                sorted.schema().fields()
            )));
        }
        let partitions =
            [&ordered, &sorted].map(|plan| plan.output_partitioning().partition_count());
        if partitions != [1, 1] {
            return Err(Error::internal(format!(
                "an ordered read and its fallback each return one partition, but they return \
                 {} and {}",
                partitions[0], partitions[1]
            )));
        }
        let properties = Arc::new(PlanProperties::new(
            sorted.equivalence_properties().clone(),
            sorted.output_partitioning().clone(),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            ordered,
            sorted,
            limit,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for OrderedFallbackExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "OrderedFallback: limit={}", self.limit)
            }
            DisplayFormatType::TreeRender => write!(f, "OrderedFallback\nlimit={}", self.limit),
        }
    }
}

impl ExecutionPlan for OrderedFallbackExec {
    fn name(&self) -> &str {
        "OrderedFallbackExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.ordered, &self.sorted]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true, true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false, false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [ordered, sorted] = <[_; 2]>::try_from(children).map_err(|children| {
            DataFusionError::Internal(format!(
                "OrderedFallbackExec takes 2 children, but got {}",
                children.len()
            ))
        })?;
        Ok(Arc::new(Self::try_new(ordered, sorted, self.limit)?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let fell_back = Arc::new(FellBack::default());
        MetricBuilder::new(&self.metrics)
            .with_partition(partition)
            .build(MetricValue::Custom {
                name: "ordered_fallback".into(),
                value: fell_back.clone(),
            });
        let ordered = self.ordered.execute(partition, context.clone())?;
        let (sorted, limit, schema) = (self.sorted.clone(), self.limit, self.schema());
        let rows = stream::once(async move {
            let batches = ordered.try_collect::<Vec<_>>().await?;
            if batches.iter().map(RecordBatch::num_rows).sum::<usize>() < limit {
                fell_back.0.store(true, Ordering::Relaxed);
                return sorted.execute(partition, context);
            }
            let batches = batches
                .into_iter()
                .map(|batch| Ok(batch.with_schema(schema.clone())?))
                .collect::<DataFusionResult<Vec<_>>>()?;
            let batches = stream::iter(batches.into_iter().map(Ok));
            Ok(Box::pin(RecordBatchStreamAdapter::new(schema, batches))
                as SendableRecordBatchStream)
        })
        .try_flatten();
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            self.schema(),
            rows,
            partition,
            &self.metrics,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

#[derive(Debug, Default)]
struct FellBack(AtomicBool);

impl fmt::Display for FellBack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.load(Ordering::Relaxed))
    }
}

impl CustomMetricValue for FellBack {
    fn new_empty(&self) -> Arc<dyn CustomMetricValue> {
        Arc::new(Self::default())
    }

    fn aggregate(&self, other: Arc<dyn CustomMetricValue>) {
        let other_fell_back = other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0.load(Ordering::Relaxed));
        self.0.fetch_or(other_fell_back, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_eq(&self, other: &Arc<dyn CustomMetricValue>) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0.load(Ordering::Relaxed) == self.0.load(Ordering::Relaxed))
    }
}
