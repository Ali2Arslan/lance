// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use arrow_schema::SortOptions;
use datafusion::catalog::Session;
use datafusion::physical_plan::ExecutionPlan;
use lance_file::version::ConcreteFileVersion;
use lance_index::scalar::expression::ScalarIndexExpr;
use lance_index::scalar::registry::IndexOrderingCapability;
use lance_table::format::Fragment;
use roaring::RoaringBitmap;

use super::{ColumnOrdering, ExprFilterPlan, FilterPlan, Scanner, TakeOperation};
use crate::Error;
use crate::index::DatasetIndexExt;
use crate::index::scalar::{IndexDetails, fetch_index_details};
use crate::index::scalar_logical::load_named_scalar_segments;
use crate::io::exec::LanceFilterExec;
use crate::io::exec::ordered_fallback::OrderedFallbackExec;
use crate::io::exec::ordered_index_scan::OrderedIndexScanExec;

const ROWS_PER_RESIDUAL_CANDIDATE: usize = 1024;
const MIN_RESIDUAL_CANDIDATES_PER_ROW: usize = 4;

pub(super) struct OrderedRead {
    pub(super) scan: Arc<dyn ExecutionPlan>,
    pub(super) residual: Option<ResidualRead>,
}

pub(super) struct ResidualRead {
    fetch: usize,
    limit: usize,
    budget: Option<usize>,
}

impl ResidualRead {
    pub(super) fn stop_at_fetch(
        &self,
        filter: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>, Error> {
        filter
            .downcast_ref::<LanceFilterExec>()
            .and_then(|filter| filter.filter.with_fetch(Some(self.fetch)))
            .ok_or_else(|| {
                Error::internal(
                    "the residual filter of an ordered index read is not a LanceFilterExec",
                )
            })
    }
}

/// Why a scan that orders by a column sorts its rows instead of reading an
/// index on that column in key order.
///
/// With [`Scanner::require_ordered_index`] set, planning fails with an
/// [`Error::InvalidInput`] whose source is this value.
/// [`Scanner::explain_plan`] with `verbose` prints it after the plan.
#[derive(Debug)]
#[non_exhaustive]
pub enum OrderedIndexDecline {
    /// The scan orders by more than one column.
    MultipleOrderKeys { columns: Vec<String> },
    /// The scan option named by `option`, `use_ordered_index` or
    /// `use_scalar_index`, is `false`.
    Disabled {
        column: String,
        option: &'static str,
    },
    /// The order column is not a top-level field, such as a field of a struct.
    NotPlainColumn { column: String },
    /// The scan runs a vector, full-text, or MinHash search.
    UnsupportedSearch {
        column: String,
        search: &'static str,
    },
    /// No index on the column can be read in key order. When the filter is an
    /// exact search of one index on the column, only that index counts.
    NoIndex { column: String },
    /// The scan has no limit.
    NoLimit { column: String },
    /// The dataset or the scan has a property, described by `reason`, that
    /// the ordered read does not handle.
    UnsupportedScan {
        column: String,
        reason: Cow<'static, str>,
    },
    /// The filter's index query `search` is not an exact search of an index
    /// on the column, so the ordered read cannot use it. The sort reads only
    /// the rows that query selects, which can cost far less than reading the
    /// column in key order. With [`Scanner::require_ordered_index`], the
    /// ordered read applies the whole filter to the rows it reads instead.
    IndexSearch { column: String, search: String },
    /// The index covers `indexed_fragments` of the dataset's `fragments`
    /// fragments. `indexed_fragments` is `None` when a segment of the index
    /// records no fragment coverage, so the count is unknown.
    PartialCoverage {
        column: String,
        index: String,
        indexed_fragments: Option<usize>,
        fragments: usize,
    },
    /// `source` stopped the read. With an `index`, that index failed to open
    /// or rejected a read in key order. Without one, the dataset's indices
    /// failed to load.
    UnreadableIndex {
        column: String,
        index: Option<String>,
        source: Box<Error>,
    },
}

impl OrderedIndexDecline {
    fn unreadable(column: &str, index: Option<&str>, source: Error) -> Self {
        // The sort never needed the index, so an index that fails to open
        // must not fail the scan unless it requires the ordered read.
        if !matches!(source, Error::NotSupported { .. }) {
            log::warn!("sorting instead of reading the index on '{column}' in order: {source}");
        }
        Self::UnreadableIndex {
            column: column.to_owned(),
            index: index.map(str::to_owned),
            source: Box::new(source),
        }
    }
}

impl fmt::Display for OrderedIndexDecline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("OrderedIndexScan declined: ")?;
        match self {
            Self::MultipleOrderKeys { columns } => write!(
                f,
                "the scan orders by {} columns ({}), and an ordered index read serves one",
                columns.len(),
                columns.join(", ")
            ),
            Self::Disabled { column, option } => write!(
                f,
                "{option}=false turns off ordered index reads for column '{column}'"
            ),
            Self::NotPlainColumn { column } => {
                write!(f, "order column '{column}' is not a top-level field")
            }
            Self::UnsupportedSearch { column, search } => write!(
                f,
                "the scan runs a {search}, which an ordered index read of column '{column}' \
                 cannot serve"
            ),
            Self::NoIndex { column } => write!(
                f,
                "column '{column}' has no ordered index that this scan can read"
            ),
            Self::NoLimit { column } => write!(
                f,
                "the scan has no limit, so an ordered index read of column '{column}' could not \
                 stop early"
            ),
            Self::UnsupportedScan { column, reason } => write!(
                f,
                "column '{column}' cannot be read in key order from an index because {reason}"
            ),
            Self::IndexSearch { column, search } => write!(
                f,
                "the filter's index query `{search}` is not an exact search of an index on \
                 column '{column}', and sorting the rows it selects can cost less than an \
                 ordered read"
            ),
            Self::PartialCoverage {
                column,
                index,
                indexed_fragments: Some(indexed_fragments),
                fragments,
            } => write!(
                f,
                "index '{index}' on column '{column}' covers {indexed_fragments} of the \
                 dataset's {fragments} fragments"
            ),
            Self::PartialCoverage {
                column,
                index,
                indexed_fragments: None,
                fragments,
            } => write!(
                f,
                "index '{index}' on column '{column}' has a segment that records no fragment \
                 coverage, so it cannot prove that it covers the dataset's {fragments} fragments"
            ),
            Self::UnreadableIndex {
                column,
                index: Some(index),
                source,
            } => write!(
                f,
                "index '{index}' on column '{column}' cannot be read in key order: {source}"
            ),
            Self::UnreadableIndex {
                column,
                index: None,
                source,
            } => write!(
                f,
                "the dataset's indices failed to load to read column '{column}' in key order: \
                 {source}"
            ),
        }
    }
}

impl std::error::Error for OrderedIndexDecline {}

impl Scanner {
    /// Set whether to read a B-tree index in key order to serve `order_by`.
    ///
    /// By default, a scan with a limit that orders by one column with a
    /// B-tree index reads that index in key order and stops after
    /// `offset + limit` live rows, instead of sorting every row. This applies
    /// when the scan has no filter, or a filter that searches no index except
    /// with one exact search of the same index. The read applies the rest of
    /// the filter to each row it reads, and sorts instead when that keeps too
    /// few rows within a share of the index. Set this to `false` to always
    /// sort. Other uses of scalar indices are not affected.
    ///
    /// ```
    /// # use lance::dataset::scanner::ColumnOrdering;
    /// # use lance::{Dataset, Result};
    /// # async fn test(dataset: &Dataset) -> Result<()> {
    /// let mut scan = dataset.scan();
    /// scan.order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".to_string())]))?
    ///     .limit(Some(10), None)?
    ///     .use_ordered_index(false);
    /// let newest_ten = scan.try_into_batch().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn use_ordered_index(&mut self, use_ordered_index: bool) -> &mut Self {
        self.use_ordered_index = use_ordered_index;
        self
    }

    /// Set whether a scan that orders by a column must read an index in key
    /// order.
    ///
    /// When this is `true` and the scan cannot serve `order_by` from an index,
    /// as [`Self::use_ordered_index`] describes, planning fails with
    /// [`Error::InvalidInput`] instead of sorting. The error's source is the
    /// [`OrderedIndexDecline`] that says why. A required read applies the part
    /// of the filter the index does not answer, including a search of another
    /// index, to each row it reads and never sorts instead, so a selective
    /// filter can make it read the whole index. With `use_ordered_index(false)`,
    /// planning fails with [`Error::InvalidInput`] even without `order_by`.
    /// Otherwise a scan without `order_by` is not affected.
    ///
    /// ```
    /// # use lance::dataset::scanner::ColumnOrdering;
    /// # use lance::{Dataset, Result};
    /// # async fn test(dataset: &Dataset) -> Result<()> {
    /// let mut scan = dataset.scan();
    /// scan.order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".to_string())]))?
    ///     .limit(Some(10), None)?
    ///     .require_ordered_index(true);
    /// let newest_ten = scan.try_into_batch().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn require_ordered_index(&mut self, require_ordered_index: bool) -> &mut Self {
        self.require_ordered_index = require_ordered_index;
        self
    }

    pub(super) async fn try_ordered_read(
        &self,
        ordering: &[ColumnOrdering],
        filter_plan: &mut FilterPlan,
    ) -> Result<OrderedRead, OrderedIndexDecline> {
        let [ordering] = ordering else {
            return Err(OrderedIndexDecline::MultipleOrderKeys {
                columns: ordering
                    .iter()
                    .map(|ordering| ordering.column_name.clone())
                    .collect(),
            });
        };
        let column = &ordering.column_name;
        let disabled_by = if !self.use_ordered_index {
            Some("use_ordered_index")
        } else if !self.use_scalar_index {
            Some("use_scalar_index")
        } else {
            None
        };
        if let Some(option) = disabled_by {
            return Err(OrderedIndexDecline::Disabled {
                column: column.clone(),
                option,
            });
        }
        let search = if self.nearest.is_some() {
            Some("vector search")
        } else if self.full_text_query.is_some() {
            Some("full-text search")
        } else if self.minhash_query.is_some() {
            Some("MinHash search")
        } else {
            None
        };
        if let Some(search) = search {
            return Err(OrderedIndexDecline::UnsupportedSearch {
                column: column.clone(),
                search,
            });
        }
        let schema = self.dataset.schema();
        let Some(field) = schema.fields.iter().find(|field| &field.name == column) else {
            return Err(OrderedIndexDecline::NotPlainColumn {
                column: column.clone(),
            });
        };
        let fetch = self.ordered_fetch(column)?;
        let expr_filter = &filter_plan.expr_filter_plan;
        let unsupported: Option<Cow<'static, str>> =
            if self.filter_references_version_columns(expr_filter) {
                Some("the filter reads a row version column, which the read cannot take".into())
            } else {
                expr_filter
                    .full_expr
                    .as_ref()
                    .filter(|expr| TakeOperation::try_from_expr(expr).is_some())
                    .map(|expr| {
                        format!("the filter `{expr}` selects rows by id, which a take reads").into()
                    })
            };
        if let Some(reason) = unsupported {
            return Err(OrderedIndexDecline::UnsupportedScan {
                column: column.clone(),
                reason,
            });
        }
        let index_search = |index_query: &ScalarIndexExpr| OrderedIndexDecline::IndexSearch {
            column: column.clone(),
            search: index_query.to_string(),
        };
        let search = match &expr_filter.index_query {
            Some(index_query @ ScalarIndexExpr::Query(search)) if expr_filter.skip_recheck => {
                Some((index_query, search))
            }
            Some(index_query) if !self.require_ordered_index => {
                return Err(index_search(index_query));
            }
            _ => None,
        };
        let indices = self
            .dataset
            .load_indices()
            .await
            .map_err(|source| OrderedIndexDecline::unreadable(column, None, source))?;
        let (index_name, query, residual) = match search {
            Some((_, search))
                if indices
                    .iter()
                    .any(|index| index.name == search.index_name && index.fields == [field.id]) =>
            {
                (
                    Some(search.index_name.clone()),
                    Some(search.query.clone()),
                    expr_filter.refine_expr.clone(),
                )
            }
            Some((index_query, _)) if !self.require_ordered_index => {
                return Err(index_search(index_query));
            }
            _ => (None, None, expr_filter.full_expr.clone()),
        };
        let residual_read = residual.as_ref().map(|_| ResidualRead {
            fetch,
            limit: fetch - self.offset.unwrap_or(0) as usize,
            budget: (!self.require_ordered_index).then(|| self.residual_budget(fetch)),
        });
        let scan_fetch = match &residual_read {
            None => Some(fetch),
            Some(read) => read.budget,
        };
        let options = SortOptions {
            descending: !ordering.ascending,
            nulls_first: ordering.nulls_first,
        };
        let fragments = self.dataset.fragments();
        let mut decline = OrderedIndexDecline::NoIndex {
            column: column.clone(),
        };
        let mut checked = HashSet::new();
        for index in indices.iter() {
            if index.fields != [field.id]
                || index_name.as_ref().is_some_and(|name| *name != index.name)
                || !checked.insert(index.name.as_str())
            {
                continue;
            }
            let unreadable =
                |source| OrderedIndexDecline::unreadable(column, Some(&index.name), source);
            let details = IndexDetails(
                fetch_index_details(&self.dataset, column, index)
                    .await
                    .map_err(unreadable)?,
            );
            if details.is_vector()
                || details
                    .get_plugin()
                    .map_err(unreadable)?
                    .ordering_capability(&details.0)
                    != IndexOrderingCapability::Bidirectional
            {
                continue;
            }
            let segments = load_named_scalar_segments(&self.dataset, column, &index.name)
                .await
                .map_err(unreadable)?;
            let indexed_fragments = segments
                .iter()
                .try_fold(RoaringBitmap::new(), |indexed, segment| {
                    Some(indexed | segment.fragment_bitmap.as_ref()?)
                })
                .map(|indexed| {
                    fragments
                        .iter()
                        .filter(|fragment| indexed.contains(fragment.id as u32))
                        .count()
                });
            if segments.is_empty() || indexed_fragments != Some(fragments.len()) {
                // `segments` leaves out the segments that record no fragment coverage.
                let is_coverage_unknown = indices
                    .iter()
                    .any(|segment| segment.name == index.name && segment.fragment_bitmap.is_none());
                decline = OrderedIndexDecline::PartialCoverage {
                    column: column.clone(),
                    index: index.name.clone(),
                    indexed_fragments: indexed_fragments.filter(|_| !is_coverage_unknown),
                    fragments: fragments.len(),
                };
                continue;
            }
            let scan =
                OrderedIndexScanExec::try_new(self.dataset.clone(), segments, query, options, None)
                    .await
                    .map_err(unreadable)?
                    .with_residual_label(residual.clone())
                    .with_fetch(scan_fetch)
                    .ok_or_else(|| {
                        unreadable(Error::internal(
                            "OrderedIndexScanExec does not accept a fetch",
                        ))
                    })?;
            match residual {
                None => filter_plan.disable_refine(),
                Some(residual) => {
                    filter_plan.expr_filter_plan = ExprFilterPlan::new_refine_only(residual);
                }
            }
            return Ok(OrderedRead {
                scan,
                residual: residual_read,
            });
        }
        Err(decline)
    }

    pub(super) async fn with_sort_fallback(
        &self,
        ordered: Arc<dyn ExecutionPlan>,
        residual: &ResidualRead,
        session: Option<&dyn Session>,
    ) -> Result<Arc<dyn ExecutionPlan>, Error> {
        if residual.budget.is_none() {
            return Ok(ordered);
        }
        let mut sorted = self.clone();
        sorted.use_ordered_index(false);
        let sorted = match session {
            Some(session) => sorted.create_plan_with_session(session).await?,
            None => sorted.create_plan().await?,
        };
        Ok(Arc::new(OrderedFallbackExec::try_new(
            ordered,
            sorted,
            residual.limit,
        )?))
    }

    fn residual_budget(&self, fetch: usize) -> usize {
        let rows: usize = self
            .dataset
            .fragments()
            .iter()
            .filter_map(Fragment::num_rows)
            .sum();
        (rows / ROWS_PER_RESIDUAL_CANDIDATE)
            .max(fetch.saturating_mul(MIN_RESIDUAL_CANDIDATES_PER_ROW))
    }

    /// Returns `offset + limit` when the dataset and the scan's options let an
    /// ordered read stop there, or the reason they do not.
    fn ordered_fetch(&self, column: &str) -> Result<usize, OrderedIndexDecline> {
        let offset = self.offset.unwrap_or(0);
        let Some(limit) = self.limit else {
            return Err(OrderedIndexDecline::NoLimit {
                column: column.to_owned(),
            });
        };
        let Some(fetch) = offset
            .checked_add(limit)
            .and_then(|fetch| usize::try_from(fetch).ok())
        else {
            return Err(OrderedIndexDecline::UnsupportedScan {
                column: column.to_owned(),
                reason: format!(
                    "offset {offset} plus limit {limit} overflows the number of rows the read \
                     can fetch"
                )
                .into(),
            });
        };
        let dataset_fragments = self.dataset.fragments();
        let projection = &self.projection_plan.physical_projection;
        let unsupported: Option<Cow<'static, str>> = if self
            .dataset
            .manifest()
            .data_storage_format
            .lance_file_format()
            == ConcreteFileVersion::V1
        {
            Some("the dataset stores its rows in the legacy file format".into())
        } else if let Some(fragments) = self
            .fragments
            .as_ref()
            .filter(|fragments| *fragments != dataset_fragments.as_ref())
        {
            // The node applies the dataset's deletion files, not the list's, and has to
            // skip the index rows of every other fragment, which can be most of the index.
            Some(
                format!(
                    "the scan reads a list of {} fragments that differs from the dataset's {} \
                     fragments",
                    fragments.len(),
                    dataset_fragments.len()
                )
                .into(),
            )
        } else if self.dataset.manifest().uses_stable_row_ids() {
            // With stable row ids, an update can add the fragment it writes to the index's
            // fragment bitmap without indexing its rows (issue 10351), so the bitmap
            // cannot prove that the index covers every fragment.
            Some(
                "the dataset uses stable row ids, so an index's fragment bitmap cannot prove \
                 that it covers every fragment"
                    .into(),
            )
        } else if self.include_deleted_rows {
            Some("the scan includes deleted rows".into())
        } else if projection.with_row_created_at_version
            || projection.with_row_last_updated_at_version
        {
            Some("the scan reads a row version column, which the read cannot take".into())
        } else if self.external_row_mask.is_some() {
            Some(
                "the scan has a row id prefilter, which the read cannot apply before the limit"
                    .into(),
            )
        } else if self.physical_row_addr_prefilter.is_some() {
            Some(
                "the scan has a physical row address prefilter, which the read cannot apply \
                 before the limit"
                    .into(),
            )
        } else {
            None
        };
        if let Some(reason) = unsupported {
            return Err(OrderedIndexDecline::UnsupportedScan {
                column: column.to_owned(),
                reason,
            });
        }
        Ok(fetch)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ops::Range;
    use std::sync::Arc;

    use arrow_array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
        RecordBatchIterator, StringArray, StructArray, UInt64Array,
        cast::AsArray,
        types::{Int32Type, Int64Type, UInt64Type},
    };
    use arrow_schema::{DataType, Field, Schema, SortOptions};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_ADDR, ROW_ADDR_FIELD, utils::address::RowAddress};
    use lance_file::version::LanceFileVersion;
    use lance_index::IndexType;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::scalar::minhash_lsh::MinHashQuery;
    use lance_index::scalar::{
        BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams,
    };
    use lance_table::format::IndexMetadata;
    use rstest::rstest;

    use super::super::{ColumnOrdering, RowAddrMask, RowAddrTreeMap, Scanner};
    use crate::dataset::transaction::{Operation, Transaction};
    use crate::dataset::{UpdateBuilder, WriteDestination, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::{Dataset, Error};

    const ROWS: i32 = 300;
    const FRAGMENT_ROWS: usize = 100;
    const INDEX: &str = "ts_idx";

    fn rows(ids: Range<i32>) -> RecordBatch {
        let keys = Int32Array::from_iter(
            ids.clone()
                .map(|id| (id % 40 != 3).then_some(id * 37 % ROWS)),
        );
        let values =
            Int64Array::from_iter(keys.iter().map(|key| key.map(|key| i64::from(key) * 10)));
        let nested = StructArray::from(vec![(
            Arc::new(Field::new("k", DataType::Int32, true)),
            Arc::new(keys.clone()) as ArrayRef,
        )]);
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            2,
            Arc::new(Float32Array::from_iter_values(
                ids.clone().flat_map(|id| [id as f32; 2]),
            )),
            None,
        )
        .unwrap();
        let text = StringArray::from_iter_values(
            ids.clone()
                .map(|id| if id % 2 == 0 { "even" } else { "odd" }),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("ts", DataType::Int32, true)
                .with_metadata(HashMap::from([("unit".into(), "s".into())])),
            Field::new("v", DataType::Int64, true),
            Field::new("s", nested.data_type().clone(), false),
            Field::new("vec", vectors.data_type().clone(), false),
            Field::new("text", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(ids)),
                Arc::new(keys),
                Arc::new(values),
                Arc::new(nested),
                Arc::new(vectors),
                Arc::new(text),
            ],
        )
        .unwrap()
    }

    async fn write(uri: &str, params: WriteParams) -> Dataset {
        let batch = rows(0..ROWS);
        let params = WriteParams {
            max_rows_per_file: FRAGMENT_ROWS,
            ..params
        };
        Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            uri,
            Some(params),
        )
        .await
        .unwrap()
    }

    fn btree() -> ScalarIndexParams {
        ScalarIndexParams::for_builtin(BuiltinIndexType::BTree)
    }

    async fn delete(dataset: &mut Dataset) {
        dataset
            .delete("ts < 3 OR ts > 296 OR ts % 11 = 5 OR id = 43")
            .await
            .unwrap();
    }

    async fn dataset(params: WriteParams) -> Dataset {
        let mut dataset = write("memory://", params).await;
        dataset
            .create_index(
                &["ts"],
                IndexType::BTree,
                Some(INDEX.into()),
                &btree(),
                false,
            )
            .await
            .unwrap();
        delete(&mut dataset).await;
        dataset
    }

    async fn scan(
        dataset: &Dataset,
        configure: &impl Fn(&mut Scanner),
        use_ordered_index: bool,
    ) -> (String, RecordBatch) {
        let mut scanner = dataset.scan();
        configure(&mut scanner);
        scanner.use_ordered_index(use_ordered_index);
        let plan = scanner.explain_plan(false).await.unwrap();
        (plan, scanner.try_into_batch().await.unwrap())
    }

    enum Expect<'a> {
        OrderedScan(&'a str),
        Sort { reason_prefix: &'a str },
        SortUnlessRequired { reason_prefix: &'a str },
    }

    async fn assert_ordered_read(
        dataset: &Dataset,
        configure: impl Fn(&mut Scanner),
        expect: Expect<'_>,
    ) -> RecordBatch {
        let (plan, rows) = scan(dataset, &configure, true).await;
        let (sorted_plan, sorted_rows) = scan(dataset, &configure, false).await;
        assert!(!sorted_plan.contains("OrderedIndexScan"), "{sorted_plan}");
        let mut scanner = dataset.scan();
        configure(&mut scanner);
        let verbose_plan = scanner.explain_plan(true).await.unwrap();
        let required_plan = scanner
            .require_ordered_index(true)
            .explain_plan(false)
            .await;
        match expect {
            Expect::OrderedScan(node) => {
                assert!(plan.contains(node) && !plan.contains("SortExec"), "{plan}");
                assert!(!verbose_plan.contains("declined"), "{verbose_plan}");
                assert_eq!(required_plan.unwrap(), plan);
            }
            Expect::Sort { reason_prefix } | Expect::SortUnlessRequired { reason_prefix } => {
                assert_eq!(plan, sorted_plan);
                let declined = format!("OrderedIndexScan declined: {reason_prefix}");
                let last_line = verbose_plan.lines().last().unwrap();
                assert!(
                    last_line.starts_with(&declined)
                        && verbose_plan.matches("declined").count() == 1
                        && !plan.contains("declined"),
                    "{verbose_plan}"
                );
                if let Expect::Sort { .. } = expect {
                    let error = required_plan.unwrap_err();
                    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
                    assert!(error.to_string().contains(&declined), "{error}");
                } else {
                    let required_plan = required_plan.unwrap();
                    assert!(
                        required_plan.contains("OrderedIndexScan")
                            && required_plan.contains("residual=")
                            && !required_plan.contains("OrderedFallback"),
                        "{required_plan}"
                    );
                    let mut required = dataset.scan();
                    configure(&mut required);
                    let required_rows = required
                        .require_ordered_index(true)
                        .try_into_batch()
                        .await
                        .unwrap();
                    assert_eq!(required_rows, sorted_rows);
                }
            }
        }
        assert_eq!(rows, sorted_rows);
        rows
    }

    fn order_by_ts(scanner: &mut Scanner) -> &mut Scanner {
        scanner
            .order_by(Some(vec![ColumnOrdering::asc_nulls_last("ts".into())]))
            .unwrap()
            .limit(Some(10), None)
            .unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_read_replaces_the_sort(
        #[values(true, false)] ascending: bool,
        #[values(true, false)] nulls_first: bool,
        #[values(
            None,
            Some("ts BETWEEN 20 AND 80"),
            Some("ts = 42"),
            Some("ts IN (3, 50, 77, 200)"),
            Some("ts IS NULL")
        )]
        filter: Option<&str>,
    ) {
        let dataset = dataset(WriteParams::default()).await;
        let ordering = ColumnOrdering {
            ascending,
            nulls_first,
            column_name: "ts".into(),
        };
        let order = SortOptions {
            descending: !ascending,
            nulls_first,
        };
        for (columns, limit, offset) in [(&["v"][..], 7, 0), (&["ts", "v"][..], 5, 4)] {
            let configure = |scanner: &mut Scanner| {
                scanner
                    .project(columns)
                    .unwrap()
                    .order_by(Some(vec![ordering.clone()]))
                    .unwrap()
                    .limit(Some(limit), Some(offset))
                    .unwrap();
                if let Some(filter) = filter {
                    scanner.filter(filter).unwrap();
                }
            };
            let node = format!(
                "OrderedIndexScan: index={INDEX}, order=[ts {order}], fetch={}",
                offset + limit
            );
            let rows = assert_ordered_read(&dataset, configure, Expect::OrderedScan(&node)).await;
            assert_eq!(rows.schema().fields().len(), columns.len());
            if offset == 0 {
                assert!(rows.num_rows() > 0, "{filter:?} selects live rows");
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_read_takes_row_ids_with_every_fragment_listed(
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let dataset = dataset(WriteParams {
            enable_stable_row_ids,
            ..Default::default()
        })
        .await;
        let fragments = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.metadata().clone())
            .collect::<Vec<_>>();
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .with_row_id()
                .with_fragments(fragments.clone())
                .order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".into())]))
                .unwrap()
                .limit(Some(20), Some(3))
                .unwrap();
        };
        let node = format!("OrderedIndexScan: index={INDEX}, order=[ts DESC NULLS LAST], fetch=23");
        let expect = if enable_stable_row_ids {
            Expect::Sort {
                reason_prefix: "column 'ts' cannot be read in key order from an index because the dataset uses stable row ids, so an index's fragment \
                 bitmap cannot prove that it covers every fragment",
            }
        } else {
            Expect::OrderedScan(&node)
        };
        let rows = assert_ordered_read(&dataset, configure, expect).await;
        assert_eq!(rows.num_rows(), 20);
    }

    #[rstest]
    #[case::scalar_index_disabled(|scanner: &mut Scanner| {
        scanner.use_scalar_index(false);
    }, "use_scalar_index=false turns off ordered index reads for column 'ts'")]
    #[case::no_limit(|scanner: &mut Scanner| {
        scanner.limit(None, Some(3)).unwrap();
    }, "the scan has no limit, so an ordered index read of column 'ts' could not stop early")]
    #[case::offset_plus_limit_overflows(|scanner: &mut Scanner| {
        scanner.limit(Some(i64::MAX), Some(i64::MAX)).unwrap();
    }, "column 'ts' cannot be read in key order from an index because offset 9223372036854775807 plus limit 9223372036854775807 overflows the number of rows the read can fetch")]
    #[case::two_order_columns(|scanner: &mut Scanner| {
        scanner
            .order_by(Some(vec![
                ColumnOrdering::asc_nulls_last("ts".into()),
                ColumnOrdering::asc_nulls_last("id".into()),
            ]))
            .unwrap();
    }, "the scan orders by 2 columns (ts, id), and an ordered index read serves one")]
    #[case::unindexed_order_column(|scanner: &mut Scanner| {
        scanner
            .order_by(Some(vec![ColumnOrdering::asc_nulls_last("v".into())]))
            .unwrap();
    }, "column 'v' has no ordered index that this scan can read")]
    #[case::row_id_prefilter(|scanner: &mut Scanner| {
        scanner.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(0..50)));
    }, "column 'ts' cannot be read in key order from an index because the scan has a row id prefilter, which the read cannot apply before the limit")]
    #[case::physical_row_prefilter(|scanner: &mut Scanner| {
        scanner.with_physical_row_addr_prefilter(RowAddrTreeMap::from_iter(0..50));
    }, "column 'ts' cannot be read in key order from an index because the scan has a physical row address prefilter, which the read cannot apply before the limit")]
    #[case::row_version_projection(|scanner: &mut Scanner| {
        scanner.project(&["id", "_row_last_updated_at_version"]).unwrap();
    }, "column 'ts' cannot be read in key order from an index because the scan reads a row version column, which the read cannot take")]
    #[case::row_version_filter(|scanner: &mut Scanner| {
        scanner.filter("_row_created_at_version = 1").unwrap();
    }, "column 'ts' cannot be read in key order from an index because the filter reads a row version column, which the read cannot take")]
    #[case::deleted_rows_included(|scanner: &mut Scanner| {
        scanner.with_row_id().include_deleted_rows();
    }, "column 'ts' cannot be read in key order from an index because the scan includes deleted rows")]
    #[case::fragment_subset(|scanner: &mut Scanner| {
        let fragments = scanner.dataset.fragments()[1..].to_vec();
        scanner.with_fragments(fragments);
    }, "column 'ts' cannot be read in key order from an index because the scan reads a list of 2 fragments that differs from the dataset's 3 fragments")]
    #[case::row_id_take(|scanner: &mut Scanner| {
        scanner.filter("_rowid IN (1, 2, 3)").unwrap();
    }, "column 'ts' cannot be read in key order from an index because the filter `_rowid = UInt64(1) OR _rowid = UInt64(2) OR _rowid = UInt64(3)` selects rows by id, which a take reads")]
    #[case::vector_search(|scanner: &mut Scanner| {
        scanner.nearest("vec", &Float32Array::from(vec![10.0, 10.0]), 30).unwrap();
    }, "the scan runs a vector search, which an ordered index read of column 'ts' cannot serve")]
    #[tokio::test]
    async fn test_ordered_decline_scan_options(
        #[case] decline: fn(&mut Scanner),
        #[case] reason: &str,
    ) {
        let dataset = dataset(WriteParams::default()).await;
        let configure = |scanner: &mut Scanner| {
            decline(order_by_ts(scanner));
        };
        let expect = Expect::Sort {
            reason_prefix: reason,
        };
        assert_ordered_read(&dataset, configure, expect).await;
    }

    #[rstest]
    #[case::required_first(true)]
    #[case::disabled_first(false)]
    #[tokio::test]
    async fn test_ordered_decline_with_use_ordered_index_false(#[case] required_first: bool) {
        let dataset = dataset(WriteParams::default()).await;
        let mut scanner = dataset.scan();
        order_by_ts(&mut scanner);
        if required_first {
            scanner.require_ordered_index(true).use_ordered_index(false);
        } else {
            scanner.use_ordered_index(false).require_ordered_index(true);
        }
        let error = scanner.explain_plan(false).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains(
                "require_ordered_index=true and use_ordered_index=false are mutually exclusive"
            ),
            "{error}"
        );
        let plan = scanner
            .require_ordered_index(false)
            .explain_plan(true)
            .await
            .unwrap();
        assert!(
            plan.ends_with(
                "OrderedIndexScan declined: use_ordered_index=false turns off ordered index \
                 reads for column 'ts'\n"
            ),
            "{plan}"
        );
    }

    #[tokio::test]
    async fn test_ordered_decline_precedes_a_failing_source() {
        let dataset = dataset(WriteParams::default()).await;
        let mut scanner = dataset.scan();
        order_by_ts(&mut scanner)
            .minhash_search(MinHashQuery::new("even", "text"))
            .unwrap();
        let source_error = scanner.explain_plan(false).await.unwrap_err();
        assert!(
            source_error
                .to_string()
                .contains("No MinHash LSH index found for column text"),
            "{source_error}"
        );
        let error = scanner
            .require_ordered_index(true)
            .explain_plan(false)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains(
                "OrderedIndexScan declined: the scan runs a MinHash search, which an ordered \
                 index read of column 'ts' cannot serve"
            ),
            "{error}"
        );
    }

    async fn overlay_key(dataset: Dataset, old_key: i32, new_key: i32) -> Dataset {
        let batch = dataset
            .scan()
            .project(&["ts"])
            .unwrap()
            .with_row_address()
            .filter(&format!("ts = {old_key}"))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch["ts"].as_primitive::<Int32Type>().value(0), old_key);
        let addr = batch[ROW_ADDR].as_primitive::<UInt64Type>().value(0);
        let field_id = dataset.schema().field("ts").unwrap().id;
        let mut overlay = dataset
            .get_fragment(RowAddress::from(addr).fragment_id() as usize)
            .unwrap()
            .write_overlay(&dataset.schema().project_by_ids(&[field_id], true))
            .await
            .unwrap();
        let values = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                ROW_ADDR_FIELD.clone(),
                Field::from(dataset.schema().field("ts").unwrap()),
            ])),
            vec![
                Arc::new(UInt64Array::from(vec![addr])),
                Arc::new(Int32Array::from(vec![new_key])),
            ],
        )
        .unwrap();
        overlay.write_batch(&values).await.unwrap();
        let group = overlay.finish().await.unwrap().unwrap();
        let read_version = dataset.version().version;
        Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataOverlay {
                groups: vec![group],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap()
    }

    async fn segmented_dataset(uri: &str, segment_fragments: &[&[u32]]) -> Dataset {
        let mut dataset = write(uri, WriteParams::default()).await;
        let mut segments = Vec::new();
        for fragment_ids in segment_fragments {
            let segment = dataset
                .create_index_builder(&["ts"], IndexType::BTree, &btree())
                .name(INDEX.into())
                .fragments(fragment_ids.to_vec())
                .execute_uncommitted()
                .await
                .unwrap();
            segments.push(segment);
        }
        dataset
            .commit_existing_index_segments(INDEX, "ts", segments)
            .await
            .unwrap();
        delete(&mut dataset).await;
        dataset
    }

    async fn recommit_segment(
        dataset: &mut Dataset,
        segment: IndexMetadata,
        fragments: Option<&[u32]>,
    ) {
        let recommitted = IndexMetadata {
            fragment_bitmap: fragments.map(|fragments| fragments.iter().copied().collect()),
            ..segment.clone()
        };
        let transaction = Transaction::new(
            dataset.version().version,
            Operation::CreateIndex {
                new_indices: vec![recommitted],
                removed_indices: vec![segment],
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();
    }

    #[derive(Debug)]
    enum Segments {
        AppendedAndOptimized,
        OnePerFragment,
        OneEmptiedByDeletes,
        FragmentClaimedTwice,
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_read_merges_index_segments(
        #[values(
            Segments::AppendedAndOptimized,
            Segments::OnePerFragment,
            Segments::OneEmptiedByDeletes,
            Segments::FragmentClaimedTwice
        )]
        segments: Segments,
        #[values(true, false)] ascending: bool,
        #[values(true, false)] nulls_first: bool,
        #[values(None, Some("ts BETWEEN 20 AND 80"))] filter: Option<&str>,
    ) {
        let (dataset, count) = match segments {
            Segments::AppendedAndOptimized => {
                let mut dataset = dataset(WriteParams::default()).await;
                let batch = rows(ROWS..ROWS + 30);
                dataset
                    .append(
                        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                        None,
                    )
                    .await
                    .unwrap();
                dataset
                    .optimize_indices(&OptimizeOptions::append())
                    .await
                    .unwrap();
                (dataset, 2)
            }
            Segments::OnePerFragment => {
                (segmented_dataset("memory://", &[&[0], &[1], &[2]]).await, 3)
            }
            Segments::OneEmptiedByDeletes => {
                let mut dataset = segmented_dataset("memory://", &[&[0], &[1], &[2]]).await;
                dataset.delete("id >= 200").await.unwrap();
                (dataset, 2)
            }
            Segments::FragmentClaimedTwice => {
                // The commit finds the overlap and recomputes each segment's
                // coverage from its rows, so the scan sees disjoint segments.
                let mut dataset = segmented_dataset("memory://", &[&[0, 1], &[2]]).await;
                let last = dataset
                    .load_indices_by_name(INDEX)
                    .await
                    .unwrap()
                    .into_iter()
                    .find(|segment| segment.fragment_bitmap.as_ref().unwrap().contains(2))
                    .unwrap();
                recommit_segment(&mut dataset, last, Some(&[1, 2])).await;
                (dataset, 2)
            }
        };
        let ordering = ColumnOrdering {
            ascending,
            nulls_first,
            column_name: "ts".into(),
        };
        let order = SortOptions {
            descending: !ascending,
            nulls_first,
        };
        for (columns, limit, offset) in [(&["v"][..], 7, 0), (&["ts", "v"][..], 5, 4)] {
            let configure = |scanner: &mut Scanner| {
                scanner
                    .project(columns)
                    .unwrap()
                    .order_by(Some(vec![ordering.clone()]))
                    .unwrap()
                    .limit(Some(limit), Some(offset))
                    .unwrap();
                if let Some(filter) = filter {
                    scanner.filter(filter).unwrap();
                }
            };
            let node = format!(
                "OrderedIndexScan: index={INDEX}, segments={count}, order=[ts {order}], fetch={}",
                offset + limit
            );
            let rows = assert_ordered_read(&dataset, configure, Expect::OrderedScan(&node)).await;
            if offset == 0 {
                assert!(rows.num_rows() > 0, "{filter:?} selects live rows");
            }
        }
    }

    #[derive(Debug)]
    enum IndexShape {
        AppendedFragment,
        FragmentSubsetOfPartialIndex,
        SegmentsMissingAFragment,
        SegmentWithoutCoverage,
        SegmentWithNewerKeyOverlay,
        BitmapIndex,
        NestedKey,
        LegacyStorage,
        NewerKeyOverlay,
        MissingIndexFiles,
        FullTextSearch,
        StableRowIdsClaimingAnUpdatedFragment,
    }

    impl IndexShape {
        fn reason(&self) -> &'static str {
            match self {
                Self::AppendedFragment => {
                    "index 'ts_idx' on column 'ts' covers 3 of the dataset's 4 fragments"
                }
                Self::FragmentSubsetOfPartialIndex => {
                    "column 'ts' cannot be read in key order from an index because the scan reads \
                     a list of 3 fragments that differs from the dataset's 4 fragments"
                }
                Self::SegmentsMissingAFragment => {
                    "index 'ts_idx' on column 'ts' covers 2 of the dataset's 3 fragments"
                }
                Self::SegmentWithoutCoverage => {
                    "index 'ts_idx' on column 'ts' has a segment that records no fragment \
                     coverage, so it cannot prove that it covers the dataset's 3 fragments"
                }
                Self::SegmentWithNewerKeyOverlay | Self::NewerKeyOverlay => {
                    "index 'ts_idx' on column 'ts' cannot be read in key order: Not supported: \
                     ordered index scan cannot read segment"
                }
                Self::BitmapIndex => "column 'id' has no ordered index that this scan can read",
                Self::NestedKey => "order column 's.k' is not a top-level field",
                Self::LegacyStorage => {
                    "column 'ts' cannot be read in key order from an index because the dataset \
                     stores its rows in the legacy file format"
                }
                Self::MissingIndexFiles => {
                    "index 'ts_idx' on column 'ts' cannot be read in key order: "
                }
                Self::FullTextSearch => {
                    "the scan runs a full-text search, which an ordered index read of column 'ts' \
                     cannot serve"
                }
                Self::StableRowIdsClaimingAnUpdatedFragment => {
                    "column 'ts' cannot be read in key order from an index because the dataset \
                     uses stable row ids, so an index's fragment bitmap cannot prove that it \
                     covers every fragment"
                }
            }
        }
    }

    async fn claim_updated_fragment() -> Dataset {
        let dataset = dataset(WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        })
        .await;
        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("ts < 10")
            .unwrap()
            .set("v", "v + 1")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        assert_eq!(updated.rows_updated, 6);
        let mut dataset = Arc::unwrap_or_clone(updated.new_dataset);
        let index = dataset.load_indices_by_name(INDEX).await.unwrap().remove(0);
        let fragments = dataset
            .fragments()
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect::<Vec<_>>();
        recommit_segment(&mut dataset, index, Some(&fragments)).await;
        dataset
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_decline_index_shapes(
        #[values(
            IndexShape::AppendedFragment,
            IndexShape::FragmentSubsetOfPartialIndex,
            IndexShape::SegmentsMissingAFragment,
            IndexShape::SegmentWithoutCoverage,
            IndexShape::SegmentWithNewerKeyOverlay,
            IndexShape::BitmapIndex,
            IndexShape::NestedKey,
            IndexShape::LegacyStorage,
            IndexShape::NewerKeyOverlay,
            IndexShape::MissingIndexFiles,
            IndexShape::FullTextSearch,
            IndexShape::StableRowIdsClaimingAnUpdatedFragment
        )]
        shape: IndexShape,
    ) {
        let mut order_column = "ts";
        let mut full_text_query = None;
        let tempdir = TempStrDir::default();
        let dataset = match shape {
            IndexShape::AppendedFragment | IndexShape::FragmentSubsetOfPartialIndex => {
                let mut dataset = dataset(WriteParams::default()).await;
                let batch = rows(ROWS..ROWS + 30);
                dataset
                    .append(
                        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                        None,
                    )
                    .await
                    .unwrap();
                dataset
            }
            IndexShape::SegmentsMissingAFragment => {
                segmented_dataset("memory://", &[&[0], &[2]]).await
            }
            IndexShape::SegmentWithoutCoverage => {
                let dataset = segmented_dataset(tempdir.as_str(), &[&[0, 1], &[2]]).await;
                let segment = dataset
                    .load_indices_by_name(INDEX)
                    .await
                    .unwrap()
                    .into_iter()
                    .find(|segment| segment.fragment_bitmap.as_ref().unwrap().contains(2))
                    .unwrap();
                // A commit recomputes a missing bitmap from any segment it can open.
                dataset
                    .object_store
                    .remove_dir_all(dataset.indices_dir().join(segment.uuid.to_string()))
                    .await
                    .unwrap();
                let mut dataset = Dataset::open(tempdir.as_str()).await.unwrap();
                recommit_segment(&mut dataset, segment, None).await;
                dataset
            }
            IndexShape::SegmentWithNewerKeyOverlay => {
                let key_in_fragment_2 = 250;
                overlay_key(
                    segmented_dataset("memory://", &[&[0, 1], &[2]]).await,
                    key_in_fragment_2,
                    -1,
                )
                .await
            }
            IndexShape::BitmapIndex => {
                let mut dataset = dataset(WriteParams::default()).await;
                dataset
                    .create_index(
                        &["id"],
                        IndexType::Bitmap,
                        None,
                        &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
                        false,
                    )
                    .await
                    .unwrap();
                order_column = "id";
                dataset
            }
            IndexShape::NestedKey => {
                let mut dataset = dataset(WriteParams::default()).await;
                dataset
                    .create_index(&["s.k"], IndexType::BTree, None, &btree(), false)
                    .await
                    .unwrap();
                order_column = "s.k";
                dataset
            }
            IndexShape::LegacyStorage => {
                dataset(WriteParams {
                    data_storage_version: Some(LanceFileVersion::Legacy),
                    ..Default::default()
                })
                .await
            }
            IndexShape::NewerKeyOverlay => {
                overlay_key(dataset(WriteParams::default()).await, 150, -1).await
            }
            IndexShape::MissingIndexFiles => {
                let mut dataset = write(tempdir.as_str(), WriteParams::default()).await;
                dataset
                    .create_index(
                        &["ts"],
                        IndexType::BTree,
                        Some(INDEX.into()),
                        &btree(),
                        false,
                    )
                    .await
                    .unwrap();
                let segment = dataset.load_indices_by_name(INDEX).await.unwrap().remove(0);
                dataset
                    .object_store
                    .remove_dir_all(dataset.indices_dir().join(segment.uuid.to_string()))
                    .await
                    .unwrap();
                Dataset::open(tempdir.as_str()).await.unwrap()
            }
            IndexShape::FullTextSearch => {
                let mut dataset = dataset(WriteParams::default()).await;
                dataset
                    .create_index(
                        &["text"],
                        IndexType::Inverted,
                        None,
                        &InvertedIndexParams::default(),
                        false,
                    )
                    .await
                    .unwrap();
                full_text_query = Some(FullTextSearchQuery::new("even".into()));
                dataset
            }
            IndexShape::StableRowIdsClaimingAnUpdatedFragment => claim_updated_fragment().await,
        };
        let fragments = match shape {
            IndexShape::FragmentSubsetOfPartialIndex => Some(dataset.fragments()[..3].to_vec()),
            _ => None,
        };
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .order_by(Some(vec![ColumnOrdering::asc_nulls_last(
                    order_column.into(),
                )]))
                .unwrap()
                .limit(Some(10), None)
                .unwrap();
            if let Some(query) = &full_text_query {
                scanner.full_text_search(query.clone()).unwrap();
            }
            if let Some(fragments) = &fragments {
                scanner.with_fragments(fragments.clone());
            }
        };
        let expect = Expect::Sort {
            reason_prefix: shape.reason(),
        };
        let rows = assert_ordered_read(&dataset, configure, expect).await;
        assert_eq!(rows.num_rows(), 10, "{shape:?}");
    }

    #[tokio::test]
    async fn test_ordered_decline_with_a_partial_index_before_a_full_one() {
        let mut dataset = write("memory://", WriteParams::default()).await;
        let partial = dataset
            .create_index_builder(&["ts"], IndexType::BTree, &btree())
            .name("partial_ts_idx".into())
            .fragments(vec![0])
            .execute_uncommitted()
            .await
            .unwrap();
        dataset
            .commit_existing_index_segments("partial_ts_idx", "ts", vec![partial])
            .await
            .unwrap();
        dataset
            .create_index(
                &["ts"],
                IndexType::BTree,
                Some(INDEX.into()),
                &btree(),
                false,
            )
            .await
            .unwrap();
        delete(&mut dataset).await;
        let node = format!("OrderedIndexScan: index={INDEX}, order=[ts ASC NULLS LAST], fetch=10");
        assert_ordered_read(
            &dataset,
            |scanner: &mut Scanner| {
                order_by_ts(scanner);
            },
            Expect::OrderedScan(&node),
        )
        .await;
        let expect = Expect::Sort {
            reason_prefix: "the scan has no limit, so an ordered index read of column 'ts' \
                            could not stop early",
        };
        assert_ordered_read(
            &dataset,
            |scanner: &mut Scanner| {
                order_by_ts(scanner).limit(None, None).unwrap();
            },
            expect,
        )
        .await;
    }

    #[rstest]
    #[case::two_searches_of_the_index(
        false,
        "ts < 10 OR ts > 290",
        "OR([ts < 10]@ts_idx(BTree),[ts > 290]@ts_idx(BTree))"
    )]
    #[case::search_of_another_index(true, "id < 100", "[id < 100]@id_idx(BTree)")]
    #[tokio::test]
    async fn test_ordered_decline_index_searches(
        #[case] index_id: bool,
        #[case] filter: &str,
        #[case] search: &str,
    ) {
        let mut dataset = dataset(WriteParams::default()).await;
        if index_id {
            dataset
                .create_index(&["id"], IndexType::BTree, None, &btree(), false)
                .await
                .unwrap();
        }
        let reason = format!(
            "the filter's index query `{search}` is not an exact search of an index on column \
             'ts', and sorting the rows it selects can cost less than an ordered read"
        );
        let configure = |scanner: &mut Scanner| {
            order_by_ts(scanner).filter(filter).unwrap();
        };
        let expect = Expect::SortUnlessRequired {
            reason_prefix: &reason,
        };
        assert_ordered_read(&dataset, configure, expect).await;
    }

    #[derive(Debug)]
    struct Row {
        id: i32,
        ts: Option<i32>,
        v: Option<i64>,
        text: String,
    }

    async fn live_rows(dataset: &Dataset) -> Vec<Row> {
        let batch = dataset
            .scan()
            .project(&["id", "ts", "v", "text"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let (ids, ts, v, text) = (
            batch["id"].as_primitive::<Int32Type>(),
            batch["ts"].as_primitive::<Int32Type>(),
            batch["v"].as_primitive::<Int64Type>(),
            batch["text"].as_string::<i32>(),
        );
        (0..batch.num_rows())
            .map(|row| Row {
                id: ids.value(row),
                ts: ts.is_valid(row).then(|| ts.value(row)),
                v: v.is_valid(row).then(|| v.value(row)),
                text: text.value(row).to_owned(),
            })
            .collect()
    }

    fn ids(batch: &RecordBatch) -> Vec<i32> {
        batch["id"].as_primitive::<Int32Type>().values().to_vec()
    }

    fn sorted_ids(
        rows: Vec<Row>,
        keep: fn(&Row) -> bool,
        ordering: &ColumnOrdering,
        offset: usize,
        limit: usize,
    ) -> Vec<i32> {
        let mut kept = rows.into_iter().filter(keep).collect::<Vec<_>>();
        kept.sort_by(|a, b| {
            let by_key = match (a.ts, b.ts) {
                (Some(a), Some(b)) if ordering.ascending => a.cmp(&b),
                (Some(a), Some(b)) => b.cmp(&a),
                (a, b) if ordering.nulls_first => a.is_some().cmp(&b.is_some()),
                (a, b) => b.is_some().cmp(&a.is_some()),
            };
            by_key.then(a.id.cmp(&b.id))
        });
        kept.into_iter()
            .skip(offset)
            .take(limit)
            .map(|row| row.id)
            .collect()
    }

    #[derive(Clone, Copy, Debug)]
    enum Rows {
        Indexed,
        MoreDeleted,
        Segmented,
    }

    impl Rows {
        async fn dataset(self) -> Dataset {
            match self {
                Self::Indexed => dataset(WriteParams::default()).await,
                Self::MoreDeleted => {
                    let mut dataset = dataset(WriteParams::default()).await;
                    dataset.delete("id % 5 = 0").await.unwrap();
                    dataset
                }
                Self::Segmented => segmented_dataset("memory://", &[&[0], &[1], &[2]]).await,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Read {
        Ordered,
        FallsBack,
        Required,
    }

    #[rstest]
    #[case::half(
        Rows::Indexed,
        "text = 'even'",
        |row: &Row| row.text == "even",
        ColumnOrdering::asc_nulls_last("ts".into()),
        (10, 0),
        Read::Ordered,
        r#"OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=40, residual=text = Utf8("even")"#
    )]
    #[case::one_percent(
        Rows::Indexed,
        "id % 100 = 7",
        |row: &Row| row.id % 100 == 7,
        ColumnOrdering::asc_nulls_last("ts".into()),
        (10, 0),
        Read::FallsBack,
        "OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=40, residual=CAST(id AS Int64) % Int64(100) = Int64(7)"
    )]
    #[case::none(
        Rows::Indexed,
        "id < 0",
        |row: &Row| row.id < 0,
        ColumnOrdering::desc_nulls_last("ts".into()),
        (10, 0),
        Read::FallsBack,
        "OrderedIndexScan: index=ts_idx, order=[ts DESC NULLS LAST], fetch=40, residual=id < Int32(0)"
    )]
    #[case::nullable_column(
        Rows::Indexed,
        "v > 1500",
        |row: &Row| row.v.is_some_and(|v| v > 1500),
        ColumnOrdering::desc_nulls_first("ts".into()),
        (10, 0),
        Read::Ordered,
        "OrderedIndexScan: index=ts_idx, order=[ts DESC NULLS FIRST], fetch=40, residual=v > Int64(1500)"
    )]
    #[case::order_column_and_another(
        Rows::Indexed,
        "ts % 4 = 0 OR id % 2 = 0",
        |row: &Row| row.ts.is_some_and(|ts| ts % 4 == 0) || row.id % 2 == 0,
        ColumnOrdering::asc_nulls_first("ts".into()),
        (10, 0),
        Read::Ordered,
        "OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS FIRST], fetch=40, residual=CAST(ts AS Int64) % Int64(4) = Int64(0) OR CAST(id AS Int64) % Int64(2) = Int64(0)"
    )]
    #[case::search_and_residual(
        Rows::Indexed,
        "ts > 100 AND text = 'odd'",
        |row: &Row| row.ts.is_some_and(|ts| ts > 100) && row.text == "odd",
        ColumnOrdering::asc_nulls_last("ts".into()),
        (10, 0),
        Read::Ordered,
        r#"OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=40, query=ts > 100, residual=text = Utf8("odd")"#
    )]
    #[case::deletions(
        Rows::MoreDeleted,
        "text = 'even'",
        |row: &Row| row.text == "even",
        ColumnOrdering::desc_nulls_last("ts".into()),
        (10, 0),
        Read::Ordered,
        r#"OrderedIndexScan: index=ts_idx, order=[ts DESC NULLS LAST], fetch=40, residual=text = Utf8("even")"#
    )]
    #[case::segments(
        Rows::Segmented,
        "text = 'odd'",
        |row: &Row| row.text == "odd",
        ColumnOrdering::asc_nulls_last("ts".into()),
        (10, 0),
        Read::Ordered,
        r#"OrderedIndexScan: index=ts_idx, segments=3, order=[ts ASC NULLS LAST], fetch=40, residual=text = Utf8("odd")"#
    )]
    #[case::offset(
        Rows::Indexed,
        "text = 'even'",
        |row: &Row| row.text == "even",
        ColumnOrdering::asc_nulls_last("ts".into()),
        (5, 4),
        Read::Ordered,
        r#"OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=36, residual=text = Utf8("even")"#
    )]
    #[case::row_id(
        Rows::Indexed,
        "_rowid >= 4294967296",
        |row: &Row| row.id >= 100,
        ColumnOrdering::asc_nulls_last("ts".into()),
        (10, 0),
        Read::Ordered,
        "OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=40, residual=_rowid >= UInt64(4294967296)"
    )]
    #[case::row_address(
        Rows::Indexed,
        "_rowaddr < 4294967296",
        |row: &Row| row.id < 100,
        ColumnOrdering::desc_nulls_last("ts".into()),
        (10, 0),
        Read::Ordered,
        "OrderedIndexScan: index=ts_idx, order=[ts DESC NULLS LAST], fetch=40, residual=_rowaddr < UInt64(4294967296)"
    )]
    #[case::budget_trip(
        Rows::Indexed,
        "ts % 50 = 7 OR ts > 250",
        |row: &Row| row.ts.is_some_and(|ts| ts % 50 == 7 || ts > 250),
        ColumnOrdering::asc_nulls_last("ts".into()),
        (10, 0),
        Read::FallsBack,
        "OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=40, residual=CAST(ts AS Int64) % Int64(50) = Int64(7) OR ts > Int32(250)"
    )]
    #[case::required(
        Rows::Indexed,
        "ts % 50 = 7 OR ts > 250",
        |row: &Row| row.ts.is_some_and(|ts| ts % 50 == 7 || ts > 250),
        ColumnOrdering::asc_nulls_last("ts".into()),
        (10, 0),
        Read::Required,
        "OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], residual=CAST(ts AS Int64) % Int64(50) = Int64(7) OR ts > Int32(250)"
    )]
    #[tokio::test]
    async fn test_ordered_residual(
        #[case] rows: Rows,
        #[case] filter: &str,
        #[case] keep: fn(&Row) -> bool,
        #[case] ordering: ColumnOrdering,
        #[case] limit_offset: (i64, i64),
        #[case] read: Read,
        #[case] scan_line: &str,
    ) {
        let dataset = rows.dataset().await;
        let (limit, offset) = limit_offset;
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts", "v"])
                .unwrap()
                .filter(filter)
                .unwrap()
                .order_by(Some(vec![ordering.clone()]))
                .unwrap()
                .limit(Some(limit), Some(offset))
                .unwrap()
                .require_ordered_index(read == Read::Required);
        };
        let mut scanner = dataset.scan();
        configure(&mut scanner);
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(plan.lines().any(|line| line.trim() == scan_line), "{plan}");
        let filter_fetch = format!(", fetch={}", limit + offset);
        assert!(
            plan.lines().any(|line| {
                line.trim_start().starts_with("FilterExec: ") && line.ends_with(&filter_fetch)
            }),
            "{plan}"
        );
        assert_eq!(
            plan.contains("OrderedFallback: limit="),
            read != Read::Required,
            "{plan}"
        );
        let analyzed = scanner.analyze_plan().await.unwrap();
        let fallback = match read {
            Read::Ordered => Some("ordered_fallback=false"),
            Read::FallsBack => Some("ordered_fallback=true"),
            Read::Required => None,
        };
        assert_eq!(
            fallback.is_some_and(|fallback| analyzed.contains(fallback)),
            fallback.is_some(),
            "{analyzed}"
        );
        assert!(
            fallback.is_some() || !analyzed.contains("ordered_fallback"),
            "{analyzed}"
        );

        let expected = sorted_ids(
            live_rows(&dataset).await,
            keep,
            &ordering,
            offset as usize,
            limit as usize,
        );
        assert!(
            !expected.is_empty() || filter == "id < 0",
            "{filter} keeps live rows"
        );
        assert_eq!(
            ids(&scanner.try_into_batch().await.unwrap()),
            expected,
            "{plan}"
        );

        let mut sorted = dataset.scan();
        configure(&mut sorted);
        sorted.require_ordered_index(false).use_ordered_index(false);
        let sorted_plan = sorted.explain_plan(false).await.unwrap();
        assert!(!sorted_plan.contains("OrderedIndexScan"), "{sorted_plan}");
        assert_eq!(ids(&sorted.try_into_batch().await.unwrap()), expected);
    }

    #[rstest]
    #[case::ordered("ts % 2 = 0", "ordered_fallback=false")]
    #[case::falls_back("ts % 50 = 7 OR ts > 250", "ordered_fallback=true")]
    #[tokio::test]
    async fn test_ordered_residual_keeps_schema_metadata(
        #[case] filter: &str,
        #[case] fallback: &str,
    ) {
        let batch = rows(0..ROWS);
        let schema = Arc::new(
            batch
                .schema()
                .as_ref()
                .clone()
                .with_metadata(HashMap::from([("format".into(), "test".into())])),
        );
        let batch = RecordBatch::try_new(schema.clone(), batch.columns().to_vec()).unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: FRAGMENT_ROWS,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["ts"],
                IndexType::BTree,
                Some(INDEX.into()),
                &btree(),
                false,
            )
            .await
            .unwrap();
        delete(&mut dataset).await;
        let configure = |scanner: &mut Scanner| {
            order_by_ts(scanner)
                .project(&["ts"])
                .unwrap()
                .filter(filter)
                .unwrap();
        };
        let mut scanner = dataset.scan();
        configure(&mut scanner);
        let analyzed = scanner.analyze_plan().await.unwrap();
        assert!(analyzed.contains(fallback), "{analyzed}");
        let mut sorted = dataset.scan();
        configure(&mut sorted);
        let expected = sorted
            .use_ordered_index(false)
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(expected.schema().metadata()["format"], "test");
        assert_eq!(scanner.try_into_batch().await.unwrap(), expected);
    }

    async fn scored_dataset(stable_row_ids: bool, appended: bool) -> Dataset {
        let rows = 0..128;
        let nullable = |row: i32| (row % 11 != 0).then_some(row);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("tenant", DataType::Int32, false),
                Field::new("score", DataType::Int32, false),
                Field::new("payload", DataType::Int32, true),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    rows.clone().map(|row| if row < 24 { 7 } else { 9 }),
                )),
                Arc::new(Int32Array::from_iter_values(rows.clone())),
                Arc::new(Int32Array::from_iter(rows.map(nullable))),
            ],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 32,
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(&["score"], IndexType::BTree, None, &btree(), false)
            .await
            .unwrap();
        dataset.delete("score = 3 OR score = 7").await.unwrap();
        if appended {
            let batch = batch.slice(0, 16);
            dataset
                .append(
                    RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                    None,
                )
                .await
                .unwrap();
        }
        dataset
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum DatasetShape {
        Indexed,
        StableRowIds,
        Appended,
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_residual_across_dataset_shapes(
        #[values(None, Some("payload % 4 = 3"), Some("tenant = 7 AND payload % 4 = 1"))]
        filter: Option<&str>,
        #[values(
            DatasetShape::Indexed,
            DatasetShape::StableRowIds,
            DatasetShape::Appended
        )]
        shape: DatasetShape,
        #[values(false, true)] descending: bool,
        #[values(None, Some(3))] limit: Option<i64>,
    ) {
        let dataset = scored_dataset(
            shape == DatasetShape::StableRowIds,
            shape == DatasetShape::Appended,
        )
        .await;
        let ordering = if descending {
            ColumnOrdering::desc_nulls_last("score".into())
        } else {
            ColumnOrdering::asc_nulls_first("score".into())
        };
        let mut scan = dataset.scan();
        scan.project(&["tenant", "score", "payload"]).unwrap();
        if let Some(filter) = filter {
            scan.filter(filter).unwrap();
        }
        scan.order_by(Some(vec![ordering]))
            .unwrap()
            .limit(limit, Some(2))
            .unwrap();
        let mut expected = scan.clone();
        expected.use_scalar_index(false).use_ordered_index(false);
        let ordered = limit.is_some() && shape == DatasetShape::Indexed;
        let plan = scan.explain_plan(false).await.unwrap();
        assert_eq!(plan.contains("OrderedIndexScan"), ordered, "{plan}");
        assert_eq!(
            plan.contains("residual="),
            ordered && filter.is_some(),
            "{plan}"
        );
        let actual = scan.try_into_batch().await.unwrap();
        if let Some(limit) = limit {
            assert_eq!(actual.num_rows(), limit as usize, "{plan}");
        }
        assert_eq!(actual, expected.try_into_batch().await.unwrap(), "{plan}");
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_residual_after_an_index_search(
        #[values(false, true)] stable_row_ids: bool,
        #[values(None, Some("score >= 0 AND payload = 'keep'"))] filter: Option<&str>,
    ) {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("score", DataType::Int32, false),
                Field::new("payload", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values(0..40)),
                Arc::new(StringArray::from_iter_values(
                    (0..40).map(|score| if score % 4 == 3 { "keep" } else { "drop" }),
                )),
            ],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 13,
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(&["score"], IndexType::BTree, None, &btree(), false)
            .await
            .unwrap();
        let mut scan = dataset.scan();
        scan.project(&["payload"])
            .unwrap()
            .order_by(Some(vec![ColumnOrdering::asc_nulls_first("score".into())]))
            .unwrap()
            .limit(Some(5), Some(1))
            .unwrap();
        if let Some(filter) = filter {
            scan.filter(filter).unwrap();
        }
        let mut expected = scan.clone();
        expected.use_scalar_index(false).use_ordered_index(false);
        let plan = scan.explain_plan(false).await.unwrap();
        assert_eq!(plan.contains("OrderedIndexScan"), !stable_row_ids, "{plan}");
        if filter.is_some() && !stable_row_ids {
            assert!(
                plan.contains(r#"query=score >= 0, residual=payload = Utf8("keep")"#),
                "{plan}"
            );
        }
        let actual = scan.try_into_batch().await.unwrap();
        assert_eq!(actual.num_rows(), 5, "{plan}");
        assert_eq!(actual, expected.try_into_batch().await.unwrap(), "{plan}");
        if filter.is_some() {
            let payload = actual["payload"].as_string::<i32>();
            assert!(
                payload.iter().all(|payload| payload == Some("keep")),
                "{plan}"
            );
        }
    }
}
