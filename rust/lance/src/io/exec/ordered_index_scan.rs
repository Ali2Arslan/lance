// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read one B-tree index segment in key order.
//!
//! [`OrderedIndexScanExec`] streams [`ScalarIndex::search_ordered`] and keeps
//! only the rows the dataset still serves. Each index row falls in one of
//! three groups, and only live rows count toward `fetch`:
//!
//! ```text
//! index row      fragment            offset         result
//! (key, addr) -> outside the scope                -> rows_excluded
//!                in the scope, gone               -> rows_deleted
//!                in the scope     -> deleted      -> rows_deleted
//!                                 -> not deleted  -> emitted
//! ```
//!
//! The scope is the segment's fragment coverage, narrowed by an optional
//! fragment allow-list.

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, UInt32Array, cast::AsArray, types::UInt64Type};
use arrow_schema::{Field, Schema, SchemaRef, SortOptions};
use arrow_select::take::take_record_batch;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
    execution_plan::{Boundedness, EmissionType},
    metrics::{Count, ExecutionPlanMetricsSet, MetricsSet},
};
use datafusion::scalar::ScalarValue;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalSortExpr, expressions::Column};
use futures::{Stream, StreamExt, TryStreamExt, stream};
use lance_core::{Error, ROW_ADDR_FIELD, Result, utils::address::RowAddress};
use lance_datafusion::utils::{ExecutionPlanMetricsSetExt, ROWS_SCANNED_METRIC};
use lance_index::{metrics::MetricsCollector, scalar::AnyQuery};
use lance_select::{RowAddrTreeMap, RowSetOps};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;

use super::utils::InstrumentedRecordBatchStreamAdapter;
use crate::{Dataset, index::DatasetIndexInternalExt};

/// Streams one segment of a B-tree index in key order as two columns: the key,
/// named after the indexed column, then `_rowaddr`.
///
/// Deleted rows and rows outside the fragment scope never reach the output, so
/// they do not use up `fetch`. Rows with equal keys come in the order the
/// index yields them.
///
/// ```
/// # use std::sync::Arc;
/// # use lance::deps::arrow_schema::SortOptions;
/// # use lance::deps::datafusion::physical_plan::ExecutionPlan;
/// # use lance::index::DatasetIndexExt;
/// # use lance::io::exec::ordered_index_scan::OrderedIndexScanExec;
/// # use lance::{Dataset, Result};
/// # async fn test(dataset: Arc<Dataset>) -> Result<()> {
/// let segment = dataset.load_indices_by_name("ts_idx").await?.remove(0);
/// let newest_first = SortOptions { descending: true, nulls_first: false };
/// let scan = OrderedIndexScanExec::try_new(dataset, segment, None, newest_first, None)?;
/// let newest_ten = scan.with_fetch(Some(10));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct OrderedIndexScanExec {
    dataset: Arc<Dataset>,
    segment: IndexMetadata,
    column: String,
    query: Option<Arc<dyn AnyQuery>>,
    options: SortOptions,
    scope: RoaringBitmap,
    fetch: Option<usize>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl OrderedIndexScanExec {
    /// Plans an ordered read of `segment`, one segment of a single-column
    /// B-tree index on `dataset`.
    ///
    /// `query` limits the read to the rows it selects, and `options` is the key
    /// order. `fragments`, when set, limits the output to rows in those
    /// fragments.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when the segment indexes more or fewer
    /// than one column or records no fragment coverage. Returns
    /// [`Error::NotSupported`] when the dataset uses stable row ids and the
    /// segment stores them instead of row addresses, which segments written
    /// before the row address migration do.
    pub fn try_new(
        dataset: Arc<Dataset>,
        segment: IndexMetadata,
        query: Option<Arc<dyn AnyQuery>>,
        options: SortOptions,
        fragments: Option<&RoaringBitmap>,
    ) -> Result<Self> {
        let [field_id] = segment.fields[..] else {
            return Err(Error::invalid_input(format!(
                "ordered index scan reads a single-column index, but segment {} of index '{}' \
                 has field ids {:?}",
                segment.uuid, segment.name, segment.fields
            )));
        };
        let Some(coverage) = &segment.fragment_bitmap else {
            return Err(Error::invalid_input(format!(
                "ordered index scan needs fragment coverage, but segment {} of index '{}' \
                 records none",
                segment.uuid, segment.name
            )));
        };
        let ids_are_addresses =
            !dataset.manifest.uses_stable_row_ids() || segment.results_are_row_addrs();
        if !ids_are_addresses {
            return Err(Error::not_supported(format!(
                "ordered index scan needs row addresses, but segment {} of index '{}' stores \
                 stable row ids; rebuild the index to read it in order",
                segment.uuid, segment.name
            )));
        }
        let column = dataset.schema().field_path(field_id)?;
        let field = dataset
            .schema()
            .field_by_id(field_id)
            .expect("field_path found the field");
        let schema = Arc::new(Schema::new(vec![
            Field::new(&column, field.data_type(), field.nullable),
            ROW_ADDR_FIELD.clone(),
        ]));
        let ordering = PhysicalSortExpr::new(Arc::new(Column::new(&column, 0)), options);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new_with_orderings(schema, [[ordering]]),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        let scope = match fragments {
            Some(fragments) => coverage & fragments,
            None => coverage.clone(),
        };
        Ok(Self {
            dataset,
            segment,
            column,
            query,
            options,
            scope,
            fetch: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    async fn live_rows(
        self,
        partition: usize,
    ) -> Result<impl Stream<Item = Result<RecordBatch>> + Send> {
        let pages_loaded: Arc<dyn MetricsCollector> = Arc::new(PagesLoaded(
            self.metrics.new_count("pages_loaded", partition),
        ));
        let (index, deleted) = futures::try_join!(
            self.dataset
                .open_scalar_index(&self.column, &self.segment.uuid, pages_loaded.as_ref()),
            load_deleted_rows(&self.dataset, &self.scope),
        )?;
        let batches = index
            .search_ordered(self.query.as_deref(), self.options, pages_loaded)
            .await?;
        let schema = self.schema();
        Ok(LiveRows {
            batches,
            scope: self.scope,
            deleted,
            remaining: self.fetch,
            schema,
            index_name: self.segment.name,
            metrics: RowMetrics::new(&self.metrics, partition),
        }
        .into_stream())
    }
}

async fn load_deleted_rows(dataset: &Dataset, scope: &RoaringBitmap) -> Result<RowAddrTreeMap> {
    let mut deleted = RowAddrTreeMap::new();
    for fragment_id in &(scope - dataset.fragment_bitmap().as_ref()) {
        deleted.insert_fragment(fragment_id);
    }
    let fragments = dataset
        .get_fragments()
        .into_iter()
        .filter(|fragment| scope.contains(fragment.id() as u32));
    let mut deletion_vectors = stream::iter(fragments)
        .map(|fragment| async move {
            let deletions = fragment.get_deletion_vector().await?;
            Result::Ok((fragment.id() as u32, deletions))
        })
        .buffer_unordered(dataset.object_store.io_parallelism());
    while let Some((fragment_id, deletions)) = deletion_vectors.try_next().await? {
        if let Some(deletions) = deletions {
            deleted.insert_bitmap(fragment_id, RoaringBitmap::from(deletions.as_ref()));
        }
    }
    Ok(deleted)
}

struct PagesLoaded(Count);

impl MetricsCollector for PagesLoaded {
    fn record_parts_loaded(&self, num_parts: usize) {
        self.0.add(num_parts);
    }
    fn record_index_loads(&self, _num_indexes: usize) {}
    fn record_comparisons(&self, _num_comparisons: usize) {}
}

struct RowMetrics {
    scanned: Count,
    deleted: Count,
    excluded: Count,
}

impl RowMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            scanned: metrics.new_count(ROWS_SCANNED_METRIC, partition),
            deleted: metrics.new_count("rows_deleted", partition),
            excluded: metrics.new_count("rows_excluded", partition),
        }
    }
}

struct LiveRows {
    batches: SendableRecordBatchStream,
    scope: RoaringBitmap,
    deleted: RowAddrTreeMap,
    remaining: Option<usize>,
    schema: SchemaRef,
    index_name: String,
    metrics: RowMetrics,
}

impl LiveRows {
    fn into_stream(self) -> impl Stream<Item = Result<RecordBatch>> + Send {
        stream::try_unfold(self, |mut rows| async move {
            while rows.remaining != Some(0) {
                let Some(batch) = rows.batches.try_next().await? else {
                    break;
                };
                let live = rows.filter(&batch)?;
                if live.num_rows() > 0 {
                    return Ok(Some((live, rows)));
                }
            }
            Ok(None)
        })
    }

    fn filter(&mut self, batch: &RecordBatch) -> Result<RecordBatch> {
        let addrs = batch.column(1).as_primitive::<UInt64Type>();
        if addrs.null_count() > 0 {
            return Err(Error::index(format!(
                "index '{}' returned a null row address in the batch that starts at key {}",
                self.index_name,
                ScalarValue::try_from_array(batch.column(0), 0)?
            )));
        }
        let mut live = Vec::new();
        let (mut num_deleted, mut num_excluded) = (0, 0);
        for (row, &addr) in addrs.values().iter().enumerate() {
            if self.remaining == Some(live.len()) {
                break;
            }
            if !self.scope.contains(RowAddress::from(addr).fragment_id()) {
                num_excluded += 1;
            } else if self.deleted.contains(addr) {
                num_deleted += 1;
            } else {
                live.push(row as u32);
            }
        }
        self.metrics
            .scanned
            .add(live.len() + num_deleted + num_excluded);
        self.metrics.deleted.add(num_deleted);
        self.metrics.excluded.add(num_excluded);
        self.remaining = self.remaining.map(|remaining| remaining - live.len());
        let live = take_record_batch(batch, &UInt32Array::from(live))?;
        Ok(RecordBatch::try_new(
            self.schema.clone(),
            live.columns().to_vec(),
        )?)
    }
}

impl DisplayAs for OrderedIndexScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let separator = match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "OrderedIndexScan: ")?;
                ", "
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "OrderedIndexScan")?;
                "\n"
            }
        };
        write!(
            f,
            "index={}{separator}order=[{} {}]",
            self.segment.name, self.column, self.options
        )?;
        if let Some(fetch) = self.fetch {
            write!(f, "{separator}fetch={fetch}")?;
        }
        if let Some(query) = &self.query {
            write!(f, "{separator}query={}", query.format(&self.column))?;
        }
        Ok(())
    }
}

impl ExecutionPlan for OrderedIndexScanExec {
    fn name(&self) -> &str {
        "OrderedIndexScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Internal(
                "OrderedIndexScanExec does not have children".to_string(),
            ))
        }
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let rows = stream::once(self.clone().live_rows(partition))
            .try_flatten()
            .map_err(DataFusionError::from);
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

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn with_fetch(&self, fetch: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(Self {
            fetch,
            ..self.clone()
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::{HashMap, HashSet};
    use std::ops::{Bound, Range};

    use arrow_array::{Int32Array, RecordBatchIterator, UInt64Array, types::Int32Type};
    use arrow_schema::DataType;
    use datafusion::physical_plan::{
        ExecutionPlanProperties, displayable, stream::RecordBatchStreamAdapter,
    };
    use datafusion_physical_expr::LexOrdering;
    use lance_core::ROW_ADDR;
    use lance_datafusion::exec::{LanceExecutionOptions, analyze_plan};
    use lance_index::IndexType;
    use lance_index::scalar::btree::BTreeParameters;
    use lance_index::scalar::{BuiltinIndexType, SargableQuery, ScalarIndexParams};
    use rstest::rstest;

    use super::*;
    use crate::dataset::WriteParams;
    use crate::index::DatasetIndexExt;
    use crate::utils::test::copy_test_data_to_tmp;

    const ROWS: i32 = 240;
    const FRAGMENT_ROWS: usize = 60;
    const PAGE_ROWS: u64 = 16;

    const ASC_NULLS_FIRST: SortOptions = SortOptions {
        descending: false,
        nulls_first: true,
    };
    const ASC_NULLS_LAST: SortOptions = SortOptions {
        descending: false,
        nulls_first: false,
    };
    const DESC_NULLS_FIRST: SortOptions = SortOptions {
        descending: true,
        nulls_first: true,
    };
    const DESC_NULLS_LAST: SortOptions = SortOptions {
        descending: true,
        nulls_first: false,
    };

    type Row = (Option<i32>, u64);

    async fn fixture(stable_row_ids: bool, nulls: bool, delete: Option<&str>) -> Arc<Dataset> {
        let unique_key_out_of_address_order = |row: i32| row * 7 % ROWS;
        let null_row = |row: i32| nulls && row % 20 == 3;
        let keys = Int32Array::from_iter(
            (0..ROWS).map(|row| (!null_row(row)).then(|| unique_key_out_of_address_order(row))),
        );
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(keys)]).unwrap();
        let params = WriteParams {
            max_rows_per_file: FRAGMENT_ROWS,
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        };
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://",
            Some(params),
        )
        .await
        .unwrap();
        let btree = BTreeParameters {
            zone_size: Some(PAGE_ROWS),
            range_id: None,
        };
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree)
            .with_params(&serde_json::to_value(btree).unwrap());
        dataset
            .create_index(
                &["key"],
                IndexType::BTree,
                Some("key_idx".into()),
                &params,
                false,
            )
            .await
            .unwrap();
        if let Some(predicate) = delete {
            dataset.delete(predicate).await.unwrap();
        }
        Arc::new(dataset)
    }

    async fn key_segment(dataset: &Dataset) -> IndexMetadata {
        dataset
            .load_indices_by_name("key_idx")
            .await
            .unwrap()
            .remove(0)
    }

    async fn live_keys(dataset: &Dataset) -> HashMap<u64, Option<i32>> {
        let batch = dataset
            .scan()
            .project(&["key"])
            .unwrap()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let keys = batch["key"].as_primitive::<Int32Type>();
        let addrs = batch[ROW_ADDR].as_primitive::<UInt64Type>();
        addrs.values().iter().copied().zip(keys).collect()
    }

    fn key_order(a: &Option<i32>, b: &Option<i32>, options: SortOptions) -> Ordering {
        match (a, b) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) if options.nulls_first => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) if options.nulls_first => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(a), Some(b)) if options.descending => b.cmp(a),
            (Some(a), Some(b)) => a.cmp(b),
        }
    }

    async fn run(plan: Arc<dyn ExecutionPlan>) -> Vec<Row> {
        let batches: Vec<RecordBatch> = plan
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        batches
            .iter()
            .flat_map(|batch| {
                let keys = batch["key"].as_primitive::<Int32Type>();
                let addrs = batch[ROW_ADDR].as_primitive::<UInt64Type>();
                keys.iter()
                    .zip(addrs.values().iter().copied())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn metric(plan: &dyn ExecutionPlan, name: &str) -> usize {
        plan.metrics()
            .unwrap()
            .sum_by_name(name)
            .map_or(0, |value| value.as_usize())
    }

    #[rstest]
    #[case::deletes_in_first_page(Some("key < 4"), None, None, ASC_NULLS_LAST, Some(10), false)]
    #[case::deletes_across_pages(Some("key % 3 = 0"), None, None, DESC_NULLS_LAST, Some(50), false)]
    #[case::fetch_zero(None, None, None, ASC_NULLS_FIRST, Some(0), false)]
    #[case::fetch_past_end(Some("key % 5 = 1"), None, None, ASC_NULLS_LAST, Some(1000), false)]
    #[case::every_row_deleted(Some("key >= 0"), None, None, ASC_NULLS_LAST, Some(10), false)]
    #[case::allow_list(
        Some("key < 30"),
        Some(&[0, 2][..]),
        None,
        ASC_NULLS_LAST,
        Some(30),
        false
    )]
    #[case::asc_nulls_first(None, None, None, ASC_NULLS_FIRST, None, true)]
    #[case::asc_nulls_last(None, None, None, ASC_NULLS_LAST, None, true)]
    #[case::desc_nulls_first(None, None, None, DESC_NULLS_FIRST, None, true)]
    #[case::desc_nulls_last(None, None, None, DESC_NULLS_LAST, None, true)]
    #[case::range_query(
        Some("key % 3 = 0"),
        None,
        Some(40..120),
        DESC_NULLS_LAST,
        Some(20),
        false
    )]
    #[tokio::test]
    async fn scan_returns_live_rows_in_key_order(
        #[case] delete: Option<&str>,
        #[case] fragments: Option<&[u32]>,
        #[case] range: Option<Range<i32>>,
        #[case] options: SortOptions,
        #[case] fetch: Option<usize>,
        #[case] nulls: bool,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let dataset = fixture(stable_row_ids, nulls, delete).await;
        let allowed = fragments.map(|ids| RoaringBitmap::from_iter(ids.iter().copied()));
        let query = range.clone().map(|range| {
            Arc::new(SargableQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(range.start))),
                Bound::Excluded(ScalarValue::Int32(Some(range.end))),
            )) as Arc<dyn AnyQuery>
        });
        let plan = OrderedIndexScanExec::try_new(
            dataset.clone(),
            key_segment(&dataset).await,
            query,
            options,
            allowed.as_ref(),
        )
        .unwrap()
        .with_fetch(fetch)
        .unwrap();

        let in_scope = |addr: u64| {
            allowed
                .as_ref()
                .is_none_or(|ids| ids.contains(RowAddress::from(addr).fragment_id()))
        };
        let selected = |key: Option<i32>| {
            range
                .as_ref()
                .is_none_or(|range| key.is_some_and(|key| range.contains(&key)))
        };
        let live = live_keys(&dataset).await;
        let mut expected = live
            .iter()
            .filter(|(addr, key)| in_scope(**addr) && selected(**key))
            .map(|(_, key)| *key)
            .collect::<Vec<_>>();
        expected.sort_by(|a, b| key_order(a, b, options));
        expected.truncate(fetch.unwrap_or(usize::MAX));

        let rows = run(plan.clone()).await;
        let keys = rows.iter().map(|(key, _)| *key).collect::<Vec<_>>();
        assert_eq!(keys, expected, "keys differ from the sorted scan");
        for (key, addr) in &rows {
            assert_eq!(
                live.get(addr),
                Some(key),
                "row {addr} is not live with key {key:?}"
            );
            assert!(in_scope(*addr), "row {addr} is outside the allow-list");
        }
        let addrs = rows.iter().map(|(_, addr)| *addr).collect::<HashSet<_>>();
        assert_eq!(addrs.len(), rows.len(), "a row was emitted twice");
        assert_eq!(
            metric(plan.as_ref(), "rows_scanned"),
            rows.len()
                + metric(plan.as_ref(), "rows_deleted")
                + metric(plan.as_ref(), "rows_excluded")
        );
    }

    #[tokio::test]
    async fn fetch_backfills_past_deleted_rows() {
        let dataset = fixture(false, false, Some("key < 4")).await;
        let segment = key_segment(&dataset).await;
        let plan = || {
            OrderedIndexScanExec::try_new(
                dataset.clone(),
                segment.clone(),
                None,
                ASC_NULLS_LAST,
                None,
            )
            .unwrap()
            .with_fetch(Some(10))
            .unwrap()
        };

        let keys = run(plan())
            .await
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>();
        assert_eq!(keys, (4..14).map(Some).collect::<Vec<_>>());

        let analyzed = analyze_plan(plan(), LanceExecutionOptions::default())
            .await
            .unwrap();
        let line = analyzed
            .lines()
            .find(|line| line.contains("OrderedIndexScan"))
            .unwrap();
        assert!(
            line.contains("index=key_idx, order=[key ASC NULLS LAST], fetch=10"),
            "{analyzed}"
        );
        let metrics = line
            .split_once("metrics=[")
            .unwrap()
            .1
            .trim_end_matches(']')
            .split(", ")
            .collect::<HashSet<_>>();
        for expected in [
            "output_rows=10",
            "rows_deleted=4",
            "rows_scanned=14",
            "pages_loaded=1",
        ] {
            assert!(metrics.contains(expected), "{expected} in {analyzed}");
        }
    }

    #[rstest]
    #[case::asc_nulls_first(ASC_NULLS_FIRST, "key ASC NULLS FIRST")]
    #[case::asc_nulls_last(ASC_NULLS_LAST, "key ASC NULLS LAST")]
    #[case::desc_nulls_first(DESC_NULLS_FIRST, "key DESC NULLS FIRST")]
    #[case::desc_nulls_last(DESC_NULLS_LAST, "key DESC NULLS LAST")]
    #[tokio::test]
    async fn plan_declares_requested_order_and_fetch(
        #[case] options: SortOptions,
        #[case] order: &str,
    ) {
        let dataset = fixture(false, false, None).await;
        let segment = key_segment(&dataset).await;
        let plan: Arc<dyn ExecutionPlan> = Arc::new(
            OrderedIndexScanExec::try_new(dataset.clone(), segment.clone(), None, options, None)
                .unwrap(),
        );
        let ordering = LexOrdering::new([PhysicalSortExpr::new(
            Arc::new(Column::new("key", 0)),
            options,
        )]);
        assert_eq!(plan.output_ordering(), ordering.as_ref());
        assert_eq!(
            plan.schema().field(0),
            &Field::new("key", DataType::Int32, true)
        );
        assert_eq!(plan.schema().field(1), &*ROW_ADDR_FIELD);
        assert_eq!(plan.fetch(), None);

        let limited = plan.with_fetch(Some(7)).unwrap();
        assert_eq!(limited.fetch(), Some(7));
        assert_eq!(limited.output_ordering(), ordering.as_ref());
        assert_eq!(limited.schema(), plan.schema());
        assert_eq!(limited.with_fetch(None).unwrap().fetch(), None);
        assert_eq!(
            displayable(limited.as_ref()).one_line().to_string().trim(),
            format!("OrderedIndexScan: index=key_idx, order=[{order}], fetch=7")
        );

        let range = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(40))),
            Bound::Excluded(ScalarValue::Int32(Some(120))),
        );
        let filtered =
            OrderedIndexScanExec::try_new(dataset, segment, Some(Arc::new(range)), options, None)
                .unwrap();
        assert_eq!(
            displayable(&filtered).one_line().to_string().trim(),
            format!(
                "OrderedIndexScan: index=key_idx, order=[{order}], query=key >= 40 && key < 120"
            )
        );
    }

    #[rstest]
    #[case::no_fragment_coverage(|segment: &mut IndexMetadata| segment.fragment_bitmap = None)]
    #[case::two_columns(|segment: &mut IndexMetadata| segment.fields.push(segment.fields[0]))]
    #[tokio::test]
    async fn malformed_segment_is_rejected(#[case] malform: fn(&mut IndexMetadata)) {
        let dataset = fixture(false, false, None).await;
        let mut segment = key_segment(&dataset).await;
        malform(&mut segment);
        let error = OrderedIndexScanExec::try_new(dataset, segment, None, ASC_NULLS_FIRST, None)
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("index 'key_idx'"), "{error}");
    }

    #[tokio::test]
    async fn stable_row_id_segment_is_rejected() {
        let dir = copy_test_data_to_tmp("v14.0.0-beta.15/stable_row_id_btree.lance").unwrap();
        let dataset = Arc::new(Dataset::open(&dir.path_str()).await.unwrap());
        let segment = dataset
            .load_indices_by_name("value_idx")
            .await
            .unwrap()
            .remove(0);
        let error = OrderedIndexScanExec::try_new(dataset, segment, None, ASC_NULLS_FIRST, None)
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("index 'value_idx'"), "{error}");
    }

    #[rstest]
    #[case::nothing_deleted(RowAddrTreeMap::new())]
    #[case::rows_deleted(RowAddrTreeMap::from_iter([5_u64]))]
    #[tokio::test]
    async fn null_row_address_is_an_error_naming_the_index(#[case] deleted: RowAddrTreeMap) {
        let page_schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, true),
            Field::new(ROW_ADDR, DataType::UInt64, true),
        ]));
        let page = RecordBatch::try_new(
            page_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![11, 12])),
                Arc::new(UInt64Array::from(vec![None, Some(7)])),
            ],
        )
        .unwrap();
        let rows = LiveRows {
            batches: Box::pin(RecordBatchStreamAdapter::new(
                page_schema,
                stream::iter([Ok(page)]),
            )),
            scope: RoaringBitmap::from_iter([0]),
            deleted,
            remaining: Some(1),
            schema: Arc::new(Schema::new(vec![
                Field::new("key", DataType::Int32, true),
                ROW_ADDR_FIELD.clone(),
            ])),
            index_name: "key_idx".into(),
            metrics: RowMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        };
        let error = rows
            .into_stream()
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains(
                "index 'key_idx' returned a null row address in the batch that starts at key 11"
            ),
            "{error}"
        );
    }
}
