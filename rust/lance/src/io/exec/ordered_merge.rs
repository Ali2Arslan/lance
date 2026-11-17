// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Merge the fragments a B-tree index misses into a read of the index in key
//! order.
//!
//! The planner merges the ordered read of the covered fragments with a top-k
//! sort of the others, both by key and then by `_rowaddr`. The sort reads the
//! other fragments with the scan's filter, as the sort plan does. With a
//! residual filter, the merge carries whether each covered row passes it, and
//! keeps the rows that do and every sorted row:
//!
//! ```text
//! ProjectionExec                 key, _rowaddr           with a residual
//!   LanceFilterExec              passes                  with a residual
//!     SharedVisitBudgetExec      with a mask that may sort instead
//!       SortPreservingMergeExec  fetch
//!         UnionExec
//!           ProjectionExec       key, _rowaddr, passes   with a residual
//!             TakeExec           the residual's columns  with a residual
//!               OrderedIndexScanExec  the covered fragments
//!           ProjectionExec       passes is true          with a residual
//!             SortExec           fetch
//!               FilteredReadExec the other fragments: key, _rowaddr
//! ```

use std::fmt;
use std::sync::Arc;
use std::task::Poll;

use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::{StreamExt, stream};

use super::ordered_index_scan::VisitBudget;
use super::utils::InstrumentedRecordBatchStreamAdapter;

/// Gives its input one [`VisitBudget`] per execution through the task context,
/// and ends its output once a read in the input spends it, without waiting for
/// the input's next batch, and drops the input. A read of the other fragments
/// whose filter matches none of their rows yields no batch until it has read
/// them all. Errors from the input always pass.
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
        let mut input = Some(self.input.execute(partition, context)?);
        let rows = stream::poll_fn(move |cx| {
            let Some(rows) = input.as_mut() else {
                return Poll::Ready(None);
            };
            let polled = rows.poll_next_unpin(cx);
            let is_cut = match &polled {
                Poll::Ready(Some(Ok(_))) => budget.is_spent(),
                Poll::Pending => budget.is_spent_or_wake(cx.waker()),
                Poll::Ready(Some(Err(_)) | None) => false,
            };
            if is_cut {
                input = None;
                return Poll::Ready(None);
            }
            polled
        });
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
    use std::time::Duration;

    use arrow_array::{Int32Array, RecordBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
    use datafusion::physical_expr::{EquivalenceProperties, LexOrdering, PhysicalSortExpr};
    use datafusion::physical_plan::Partitioning;
    use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
    use datafusion::physical_plan::expressions::col;
    use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion::physical_plan::union::UnionExec;
    use futures::channel::oneshot;
    use futures::{TryStreamExt, future};
    use lance_core::{ROW_ADDR, ROW_ADDR_FIELD};
    use lance_datafusion::exec::OneShotExec;
    use rstest::rstest;
    use tokio::runtime::Builder;

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

    /// An ordered read of the covered fragments that spends the budget. With
    /// no `delay`, it spends it before it reads a row and ends, as one whose
    /// mask is too small to walk does. With one, it spends it after the delay
    /// and then waits, as a walk does while another of its segments still
    /// reads a page.
    #[derive(Debug)]
    struct SpendsTheBudget {
        delay: Option<Duration>,
        properties: Arc<PlanProperties>,
    }

    impl SpendsTheBudget {
        fn new(delay: Option<Duration>) -> Self {
            Self {
                delay,
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
            let budget = context
                .session_config()
                .get_extension::<VisitBudget>()
                .unwrap();
            let Some(delay) = self.delay else {
                budget.mark_spent();
                return Ok(Box::pin(RecordBatchStreamAdapter::new(
                    self.schema(),
                    stream::empty(),
                )));
            };
            let rows = stream::once(async move {
                tokio::time::sleep(delay).await;
                budget.mark_spent();
                future::pending().await
            });
            Ok(Box::pin(RecordBatchStreamAdapter::new(self.schema(), rows)))
        }
    }

    /// A read of the uncovered fragments that matches none of their rows, so
    /// it yields nothing until it is dropped, which resolves the receiver.
    fn pending_until_dropped() -> (Arc<dyn ExecutionPlan>, oneshot::Receiver<()>) {
        let (on_drop, dropped) = oneshot::channel::<()>();
        let rows = stream::pending().inspect(move |_: &DataFusionResult<RecordBatch>| {
            let _ = &on_drop;
        });
        let read = RecordBatchStreamAdapter::new(schema(), rows);
        (Arc::new(OneShotExec::new(Box::pin(read))), dropped)
    }

    /// The planner's merge of `covered` with `uncovered`, under one budget.
    fn merge(
        covered: SpendsTheBudget,
        uncovered: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = schema();
        let newest_first = SortOptions {
            descending: true,
            nulls_first: false,
        };
        let order = LexOrdering::from([
            PhysicalSortExpr::new(col("ts", &schema).unwrap(), newest_first),
            PhysicalSortExpr::new(col(ROW_ADDR, &schema).unwrap(), SortOptions::default()),
        ]);
        let union = UnionExec::try_new(vec![Arc::new(covered), uncovered]).unwrap();
        let merged = SortPreservingMergeExec::new(order, union).with_fetch(Some(LIMIT));
        Arc::new(SharedVisitBudgetExec::new(Arc::new(merged)))
    }

    #[tokio::test]
    async fn test_shared_budget_ends_the_merge_once_spent() {
        let uncovered = Arc::new(OneShotExec::from_batch(rows(0..LIMIT as i32)));
        let batches = merge(SpendsTheBudget::new(None), uncovered)
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
        let uncovered = Arc::new(OneShotExec::new(Box::pin(failing)));
        let sorted = Arc::new(OneShotExec::from_batch(rows(0..LIMIT as i32)));
        let merged = merge(SpendsTheBudget::new(None), uncovered);
        let query = OrderedFallbackExec::try_new(merged, sorted, LIMIT).unwrap();
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

    #[rstest]
    fn test_shared_budget_stops_an_uncovered_read_that_yields_nothing(
        #[values(false, true)] is_multi_thread: bool,
        #[values(None, Some(Duration::from_millis(50)))] spend_delay: Option<Duration>,
    ) {
        let mut runtime = if is_multi_thread {
            Builder::new_multi_thread()
        } else {
            Builder::new_current_thread()
        };
        runtime.enable_all().build().unwrap().block_on(async {
            let (uncovered, dropped) = pending_until_dropped();
            let sorted = rows(0..LIMIT as i32);
            let query = OrderedFallbackExec::try_new(
                merge(SpendsTheBudget::new(spend_delay), uncovered),
                Arc::new(OneShotExec::from_batch(sorted.clone())),
                LIMIT,
            )
            .unwrap();
            let rows = query
                .execute(0, Arc::new(TaskContext::default()))
                .unwrap()
                .try_collect::<Vec<_>>();
            let timeout = Duration::from_secs(5);
            let started = tokio::time::Instant::now();
            let rows = tokio::time::timeout(timeout, rows)
                .await
                .expect("the merge waits for the uncovered read after the budget is spent")
                .unwrap();
            // The timeout's own wake polls the merge again, which ends a merge that no
            // spend woke, so ending well before it shows that the spend woke the merge.
            assert!(started.elapsed() < timeout / 2, "{:?}", started.elapsed());
            assert_eq!(rows, [sorted]);
            tokio::time::timeout(timeout, dropped)
                .await
                .expect("the uncovered read is never dropped")
                .unwrap_err();
        });
    }
}
