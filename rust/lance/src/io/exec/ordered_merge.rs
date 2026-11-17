// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Merge the fragments a B-tree index misses into a read of the index in key
//! order.
//!
//! The planner merges the ordered read of the covered fragments with a top-k
//! sort of the others, both by key and then by `_rowaddr`:
//!
//! ```text
//! SharedVisitBudgetExec        with a mask that may sort instead
//!   SortPreservingMergeExec    fetch
//!     UnionExec
//!       OrderedIndexScanExec   the covered fragments
//!       SortExec               fetch
//!         IndexMaskExec        with a mask
//!           FilteredReadExec   the other fragments: key, _rowaddr
//! ```

use std::fmt;
use std::sync::Arc;

use arrow_array::{BooleanArray, cast::AsArray, types::UInt64Type};
use arrow_select::filter::filter_record_batch;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::{StreamExt, TryStreamExt, future, stream};
use lance_core::{Error, ROW_ADDR};
use roaring::RoaringBitmap;

use super::ordered_index_scan::{IndexMask, VisitBudget};

/// Gives its input one [`VisitBudget`] per execution through the task context,
/// and ends its output once a read in the input spends it.
///
/// A masked ordered read that stops short ends its stream, and the merge above
/// it would then emit sorted rows in place of the index rows it never read.
#[derive(Debug)]
pub struct SharedVisitBudgetExec {
    input: Arc<dyn ExecutionPlan>,
}

impl SharedVisitBudgetExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self { input }
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
        Ok(Box::pin(RecordBatchStreamAdapter::new(self.schema(), rows)))
    }
}

/// Keeps the rows of its input, a read of the fragments in `scope` with
/// `_rowaddr`, that an [`IndexMask`] lets through, as an ordered read keeps
/// the index rows it reads. Under a [`SharedVisitBudgetExec`], it stops
/// reading once the ordered read spends the budget.
#[derive(Debug)]
pub struct IndexMaskExec {
    input: Arc<dyn ExecutionPlan>,
    mask: IndexMask,
    scope: RoaringBitmap,
}

impl IndexMaskExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, mask: IndexMask, scope: RoaringBitmap) -> Self {
        Self { input, mask, scope }
    }
}

impl DisplayAs for IndexMaskExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "IndexMask: mask={}", self.mask.label())
            }
            DisplayFormatType::TreeRender => write!(f, "IndexMask\nmask={}", self.mask.label()),
        }
    }
}

impl ExecutionPlan for IndexMaskExec {
    fn name(&self) -> &str {
        "IndexMaskExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input, self.mask.search()]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true, false]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false, false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [input, search] = <[_; 2]>::try_from(children).map_err(|children| {
            DataFusionError::Internal(format!(
                "IndexMaskExec takes 2 children, but got {}",
                children.len()
            ))
        })?;
        Ok(Arc::new(Self::new(
            input,
            self.mask.with_search(search),
            self.scope.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let budget = context.session_config().get_extension::<VisitBudget>();
        let input = self.input.execute(partition, context.clone())?;
        let (mask, scope) = (self.mask.clone(), self.scope.clone());
        let rows = stream::once(async move {
            let selected = mask.select(&scope, partition, context).await?;
            let is_spent = move || budget.as_ref().is_some_and(|budget| budget.is_spent());
            Ok::<_, DataFusionError>(input.take_while(move |_| future::ready(!is_spent())).map(
                move |batch| {
                    let batch = batch?;
                    let addrs = batch
                        .column_by_name(ROW_ADDR)
                        .ok_or_else(|| {
                            Error::internal(format!(
                                "IndexMaskExec reads `{ROW_ADDR}`, but its input returns {:?}",
                                batch.schema().fields()
                            ))
                        })?
                        .as_primitive::<UInt64Type>();
                    let kept = BooleanArray::from(
                        addrs
                            .values()
                            .iter()
                            .map(|&addr| selected.selected(addr))
                            .collect::<Vec<_>>(),
                    );
                    Ok(filter_record_batch(&batch, &kept)?)
                },
            ))
        })
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(self.schema(), rows)))
    }
}
