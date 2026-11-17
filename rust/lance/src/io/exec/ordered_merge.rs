// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Merge the fragments a B-tree index misses into a read of the index in key
//! order.
//!
//! The planner merges the ordered read of the covered fragments with a top-k
//! sort of the others, both by key and then by `_rowaddr`. The sort reads the
//! other fragments with the scan's filter, as the sort plan does:
//!
//! ```text
//! SharedVisitBudgetExec        with a mask that may sort instead
//!   SortPreservingMergeExec    fetch
//!     UnionExec
//!       OrderedIndexScanExec   the covered fragments
//!       SortExec               fetch
//!         VisitBudgetCutExec   with a mask that may sort instead
//!           FilteredReadExec   the other fragments: key, _rowaddr
//! ```

use std::fmt;
use std::sync::Arc;

use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::{StreamExt, future};

use super::ordered_index_scan::VisitBudget;
use super::utils::InstrumentedRecordBatchStreamAdapter;

/// Gives its input one [`VisitBudget`] per execution through the task context,
/// and ends its output once a read in the input spends it.
///
/// A masked ordered read that stops short ends its stream, and the merge above
/// it would then emit sorted rows in place of the index rows it never read.
#[derive(Debug)]
pub struct SharedVisitBudgetExec {
    input: Arc<dyn ExecutionPlan>,
    metrics: ExecutionPlanMetricsSet,
}

impl SharedVisitBudgetExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            input,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for SharedVisitBudgetExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SharedVisitBudget")
    }
}

impl ExecutionPlan for SharedVisitBudgetExec {
    fn name(&self) -> &str {
        "SharedVisitBudgetExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [input] = <[_; 1]>::try_from(children).map_err(|children| {
            DataFusionError::Internal(format!(
                "SharedVisitBudgetExec takes 1 child, but got {}",
                children.len()
            ))
        })?;
        Ok(Arc::new(Self::new(input)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let budget = Arc::new(VisitBudget::default());
        let config = context
            .session_config()
            .clone()
            .with_extension(budget.clone());
        let context = Arc::new(TaskContext::new(
            context.task_id(),
            context.session_id(),
            config,
            context.scalar_functions().clone(),
            context.higher_order_functions().clone(),
            context.aggregate_functions().clone(),
            context.window_functions().clone(),
            context.runtime_env(),
        ));
        let rows = self
            .input
            .execute(partition, context)?
            .take_while(move |batch| future::ready(batch.is_err() || !budget.is_spent()));
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

/// Ends its input, the read of the fragments an ordered read's index misses,
/// once the ordered read under the same [`SharedVisitBudgetExec`] spends the
/// budget, so that a read that stops short does not wait for the rest of it.
/// Errors always pass, so the sort runs only after a read that ended cleanly.
#[derive(Debug)]
pub struct VisitBudgetCutExec {
    input: Arc<dyn ExecutionPlan>,
    metrics: ExecutionPlanMetricsSet,
}

impl VisitBudgetCutExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            input,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for VisitBudgetCutExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VisitBudgetCut")
    }
}

impl ExecutionPlan for VisitBudgetCutExec {
    fn name(&self) -> &str {
        "VisitBudgetCutExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [input] = <[_; 1]>::try_from(children).map_err(|children| {
            DataFusionError::Internal(format!(
                "VisitBudgetCutExec takes 1 child, but got {}",
                children.len()
            ))
        })?;
        Ok(Arc::new(Self::new(input)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let budget = context.session_config().get_extension::<VisitBudget>();
        let is_spent = move || budget.as_ref().is_some_and(|budget| budget.is_spent());
        let rows = self
            .input
            .execute(partition, context)?
            .take_while(move |batch| future::ready(batch.is_err() || !is_spent()));
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

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use arrow_array::{Int32Array, RecordBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
    use datafusion::physical_expr::{EquivalenceProperties, LexOrdering, PhysicalSortExpr};
    use datafusion::physical_plan::Partitioning;
    use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
    use datafusion::physical_plan::expressions::col;
    use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion::physical_plan::union::UnionExec;
    use futures::{TryStreamExt, stream};
    use lance_core::{ROW_ADDR, ROW_ADDR_FIELD};
    use lance_datafusion::exec::OneShotExec;

    use super::*;
    use crate::io::exec::ordered_fallback::OrderedFallbackExec;

    const LIMIT: usize = 100;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Int32, true),
            ROW_ADDR_FIELD.clone(),
        ]))
    }

    fn rows(keys: Range<i32>) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from_iter_values(keys.clone())),
                Arc::new(UInt64Array::from_iter_values(keys.map(|key| key as u64))),
            ],
        )
        .unwrap()
    }

    /// An ordered read of the covered fragments that spends the budget before
    /// it reads a row, as one whose mask is too small to walk does.
    #[derive(Debug)]
    struct SpendsTheBudget {
        properties: Arc<PlanProperties>,
    }

    impl SpendsTheBudget {
        fn new() -> Self {
            Self {
                properties: Arc::new(PlanProperties::new(
                    EquivalenceProperties::new(schema()),
                    Partitioning::RoundRobinBatch(1),
                    EmissionType::Incremental,
                    Boundedness::Bounded,
                )),
            }
        }
    }

    impl DisplayAs for SpendsTheBudget {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "SpendsTheBudget")
        }
    }

    impl ExecutionPlan for SpendsTheBudget {
        fn name(&self) -> &str {
            "SpendsTheBudget"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            &self.properties
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
        ) -> DataFusionResult<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }

        fn execute(
            &self,
            _partition: usize,
            context: Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            context
                .session_config()
                .get_extension::<VisitBudget>()
                .unwrap()
                .mark_spent();
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema(),
                stream::empty(),
            )))
        }
    }

    /// The planner's merge of the covered read with `uncovered`, under one
    /// budget.
    fn merge(uncovered: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let schema = schema();
        let newest_first = SortOptions {
            descending: true,
            nulls_first: false,
        };
        let order = LexOrdering::from([
            PhysicalSortExpr::new(col("ts", &schema).unwrap(), newest_first),
            PhysicalSortExpr::new(col(ROW_ADDR, &schema).unwrap(), SortOptions::default()),
        ]);
        let union = UnionExec::try_new(vec![Arc::new(SpendsTheBudget::new()), uncovered]).unwrap();
        let merged = SortPreservingMergeExec::new(order, union).with_fetch(Some(LIMIT));
        Arc::new(SharedVisitBudgetExec::new(Arc::new(merged)))
    }

    #[tokio::test]
    async fn test_shared_budget_ends_the_merge_once_spent() {
        let uncovered = Arc::new(OneShotExec::from_batch(rows(0..LIMIT as i32)));
        let batches = merge(uncovered)
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    }

    #[tokio::test]
    async fn test_uncovered_error_after_the_budget_is_spent_fails_the_query() {
        let failing = RecordBatchStreamAdapter::new(
            schema(),
            stream::iter([Err(DataFusionError::Execution(
                "the uncovered read failed".into(),
            ))]),
        );
        let uncovered = Arc::new(VisitBudgetCutExec::new(Arc::new(OneShotExec::new(
            Box::pin(failing),
        ))));
        let sorted = Arc::new(OneShotExec::from_batch(rows(0..LIMIT as i32)));
        let query = OrderedFallbackExec::try_new(merge(uncovered), sorted, LIMIT).unwrap();
        let error = query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("the uncovered read failed"),
            "{error}"
        );
    }
}
