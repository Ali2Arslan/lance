// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use arrow_schema::SortOptions;
use datafusion::physical_plan::ExecutionPlan;
use lance_file::version::ConcreteFileVersion;
use lance_index::scalar::expression::ScalarIndexExpr;
use lance_index::scalar::registry::IndexOrderingCapability;
use roaring::RoaringBitmap;

use super::{ColumnOrdering, FilterPlan, Scanner};
use crate::Error;
use crate::index::DatasetIndexExt;
use crate::index::scalar::{IndexDetails, fetch_index_details};
use crate::index::scalar_logical::load_named_scalar_segments;
use crate::io::exec::ordered_index_scan::OrderedIndexScanExec;

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
    /// `limit` is `None` when the scan has no limit, and `Some` when `offset`
    /// plus `limit` overflows the number of rows the read can fetch.
    NoLimit {
        column: String,
        index: String,
        offset: i64,
        limit: Option<i64>,
    },
    /// The dataset or the scan has a property, described by `reason`, that
    /// the ordered read does not handle.
    UnsupportedScan {
        column: String,
        index: String,
        reason: Cow<'static, str>,
    },
    /// `filter` would have to run after the read, so the read could not stop
    /// at the limit.
    ResidualFilter {
        column: String,
        index: String,
        filter: Cow<'static, str>,
    },
    /// The index covers `indexed_fragments` of the dataset's `fragments`
    /// fragments. A segment that records no fragment coverage makes the count
    /// zero.
    PartialCoverage {
        column: String,
        index: String,
        indexed_fragments: usize,
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
            Self::NoLimit {
                column,
                index,
                limit: None,
                ..
            } => write!(
                f,
                "the scan has no limit, so an ordered read of index '{index}' on column \
                 '{column}' could not stop early"
            ),
            Self::NoLimit {
                column,
                index,
                offset,
                limit: Some(limit),
            } => write!(
                f,
                "offset {offset} plus limit {limit} overflows the row count of an ordered read \
                 of index '{index}' on column '{column}'"
            ),
            Self::UnsupportedScan {
                column,
                index,
                reason,
            } => write!(
                f,
                "index '{index}' on column '{column}' cannot be read in key order because \
                 {reason}"
            ),
            Self::ResidualFilter {
                column,
                index,
                filter,
            } => write!(
                f,
                "{filter} is not one exact search of index '{index}', so an ordered read of \
                 column '{column}' cannot apply it before the limit"
            ),
            Self::PartialCoverage {
                column,
                index,
                indexed_fragments,
                fragments,
            } => write!(
                f,
                "index '{index}' on column '{column}' covers {indexed_fragments} of the \
                 dataset's {fragments} fragments"
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
    /// when the scan has no filter or a filter that the same index answers
    /// exactly. Set this to `false` to always sort. Other uses of scalar
    /// indices are not affected.
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
    /// [`OrderedIndexDecline`] that says why. With `use_ordered_index(false)`,
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
    ) -> Result<Arc<dyn ExecutionPlan>, OrderedIndexDecline> {
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
        let indices = self
            .dataset
            .load_indices()
            .await
            .map_err(|source| OrderedIndexDecline::unreadable(column, None, source))?;
        let expr_filter = &filter_plan.expr_filter_plan;
        let (index_name, query, residual_filter) =
            match (&expr_filter.index_query, &expr_filter.full_expr) {
                (None, None) => (None, None, None),
                (Some(ScalarIndexExpr::Query(search)), _)
                    if expr_filter.is_exact_index_search()
                        && indices.iter().any(|index| {
                            index.name == search.index_name && index.fields == [field.id]
                        }) =>
                {
                    (
                        Some(search.index_name.clone()),
                        Some(search.query.clone()),
                        None,
                    )
                }
                (_, Some(expr)) => (None, None, Some(format!("filter `{expr}`"))),
                (Some(index_query), None) => {
                    (None, None, Some(format!("index query `{index_query}`")))
                }
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
                decline = OrderedIndexDecline::PartialCoverage {
                    column: column.clone(),
                    index: index.name.clone(),
                    indexed_fragments: indexed_fragments.unwrap_or(0),
                    fragments: fragments.len(),
                };
                continue;
            }
            // Checked at the index the scan would read, so the reason names it.
            let fetch = self.ordered_fetch(column, &index.name, residual_filter.as_deref())?;
            let plan =
                OrderedIndexScanExec::try_new(self.dataset.clone(), segments, query, options, None)
                    .await
                    .map_err(unreadable)?
                    .with_fetch(Some(fetch))
                    .ok_or_else(|| {
                        unreadable(Error::internal(
                            "OrderedIndexScanExec does not accept a fetch",
                        ))
                    })?;
            filter_plan.disable_refine();
            return Ok(plan);
        }
        Err(decline)
    }

    /// Returns `offset + limit` when the scan's options and filter let an
    /// ordered read of `index` stop there, or the reason they do not.
    fn ordered_fetch(
        &self,
        column: &str,
        index: &str,
        residual_filter: Option<&str>,
    ) -> Result<usize, OrderedIndexDecline> {
        let (column, index) = (column.to_owned(), index.to_owned());
        let offset = self.offset.unwrap_or(0);
        let Some(limit) = self.limit else {
            return Err(OrderedIndexDecline::NoLimit {
                column,
                index,
                offset,
                limit: None,
            });
        };
        let Some(fetch) = offset
            .checked_add(limit)
            .and_then(|fetch| usize::try_from(fetch).ok())
        else {
            return Err(OrderedIndexDecline::NoLimit {
                column,
                index,
                offset,
                limit: Some(limit),
            });
        };
        let dataset_fragments = self.dataset.fragments();
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
                "the dataset uses stable row ids, so the index's fragment bitmap cannot prove \
                 that it covers every fragment"
                    .into(),
            )
        } else if self.include_deleted_rows {
            Some("the scan includes deleted rows".into())
        } else {
            None
        };
        if let Some(reason) = unsupported {
            return Err(OrderedIndexDecline::UnsupportedScan {
                column,
                index,
                reason,
            });
        }
        let filter: Option<Cow<'static, str>> = if self.external_row_mask.is_some() {
            Some("the row id prefilter".into())
        } else if self.physical_row_addr_prefilter.is_some() {
            Some("the physical row address prefilter".into())
        } else {
            residual_filter.map(|filter| filter.to_owned().into())
        };
        if let Some(filter) = filter {
            return Err(OrderedIndexDecline::ResidualFilter {
                column,
                index,
                filter,
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
        types::{Int32Type, UInt64Type},
    };
    use arrow_schema::{DataType, Field, Schema, SortOptions};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_ADDR, ROW_ADDR_FIELD, utils::address::RowAddress};
    use lance_file::version::LanceFileVersion;
    use lance_index::IndexType;
    use lance_index::optimize::OptimizeOptions;
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
            Expect::Sort { reason_prefix } => {
                assert_eq!(plan, sorted_plan);
                let declined = format!("OrderedIndexScan declined: {reason_prefix}");
                let last_line = verbose_plan.lines().last().unwrap();
                assert!(
                    last_line.starts_with(&declined)
                        && verbose_plan.matches("declined").count() == 1
                        && !plan.contains("declined"),
                    "{verbose_plan}"
                );
                let error = required_plan.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
                assert!(error.to_string().contains(&declined), "{error}");
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
                reason_prefix: "index 'ts_idx' on column 'ts' cannot be read in key order because the dataset uses stable row ids, so the index's fragment \
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
    }, "the scan has no limit, so an ordered read of index 'ts_idx' on column 'ts' could not stop early")]
    #[case::offset_plus_limit_overflows(|scanner: &mut Scanner| {
        scanner.limit(Some(i64::MAX), Some(i64::MAX)).unwrap();
    }, "offset 9223372036854775807 plus limit 9223372036854775807 overflows the row count of an ordered read of index 'ts_idx' on column 'ts'")]
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
    #[case::filter_on_another_column(|scanner: &mut Scanner| {
        scanner.filter("v > 100").unwrap();
    }, "filter `v > Int64(100)` is not one exact search of index 'ts_idx', so an ordered read of column 'ts' cannot apply it before the limit")]
    #[case::filter_with_a_residual(|scanner: &mut Scanner| {
        scanner.filter("ts > 5 AND v < 2000").unwrap();
    }, "filter `ts > Int32(5) AND v < Int64(2000)` is not one exact search of index 'ts_idx', so an ordered read of column 'ts' cannot apply it before the limit")]
    #[case::filter_of_two_index_queries(|scanner: &mut Scanner| {
        scanner.filter("ts < 10 OR ts > 290").unwrap();
    }, "filter `ts < Int32(10) OR ts > Int32(290)` is not one exact search of index 'ts_idx', so an ordered read of column 'ts' cannot apply it before the limit")]
    #[case::row_id_prefilter(|scanner: &mut Scanner| {
        scanner.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(0..50)));
    }, "the row id prefilter is not one exact search of index 'ts_idx', so an ordered read of column 'ts' cannot apply it before the limit")]
    #[case::physical_row_prefilter(|scanner: &mut Scanner| {
        scanner.with_physical_row_addr_prefilter(RowAddrTreeMap::from_iter(0..50));
    }, "the physical row address prefilter is not one exact search of index 'ts_idx', so an ordered read of column 'ts' cannot apply it before the limit")]
    #[case::deleted_rows_included(|scanner: &mut Scanner| {
        scanner.with_row_id().include_deleted_rows();
    }, "index 'ts_idx' on column 'ts' cannot be read in key order because the scan includes deleted rows")]
    #[case::fragment_subset(|scanner: &mut Scanner| {
        let fragments = scanner.dataset.fragments()[1..].to_vec();
        scanner.with_fragments(fragments);
    }, "index 'ts_idx' on column 'ts' cannot be read in key order because the scan reads a list of 2 fragments that differs from the dataset's 3 fragments")]
    #[case::row_id_take(|scanner: &mut Scanner| {
        scanner.filter("_rowid IN (1, 2, 3)").unwrap();
    }, "filter `_rowid = UInt64(1) OR _rowid = UInt64(2) OR _rowid = UInt64(3)` is not one exact search of index 'ts_idx', so an ordered read of column 'ts' cannot apply it before the limit")]
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

    async fn segmented_dataset(segment_fragments: &[&[u32]]) -> Dataset {
        let mut dataset = write("memory://", WriteParams::default()).await;
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

    async fn recommit_segment(dataset: &mut Dataset, segment: IndexMetadata, fragments: &[u32]) {
        let recommitted = IndexMetadata {
            fragment_bitmap: Some(fragments.iter().copied().collect()),
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
            Segments::OnePerFragment => (segmented_dataset(&[&[0], &[1], &[2]]).await, 3),
            Segments::OneEmptiedByDeletes => {
                let mut dataset = segmented_dataset(&[&[0], &[1], &[2]]).await;
                dataset.delete("id >= 200").await.unwrap();
                (dataset, 2)
            }
            Segments::FragmentClaimedTwice => {
                // The commit finds the overlap and recomputes each segment's
                // coverage from its rows, so the scan sees disjoint segments.
                let mut dataset = segmented_dataset(&[&[0, 1], &[2]]).await;
                let last = dataset
                    .load_indices_by_name(INDEX)
                    .await
                    .unwrap()
                    .into_iter()
                    .find(|segment| segment.fragment_bitmap.as_ref().unwrap().contains(2))
                    .unwrap();
                recommit_segment(&mut dataset, last, &[1, 2]).await;
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
        SegmentsMissingAFragment,
        SegmentWithNewerKeyOverlay,
        BitmapIndex,
        NestedKey,
        LegacyStorage,
        NewerKeyOverlay,
        MissingIndexFiles,
        FullTextSearch,
        StableRowIdsClaimingAnUpdatedFragment,
        FilterOnAnotherIndex,
    }

    impl IndexShape {
        fn reason(&self) -> &'static str {
            match self {
                Self::AppendedFragment => {
                    "index 'ts_idx' on column 'ts' covers 3 of the dataset's 4 fragments"
                }
                Self::SegmentsMissingAFragment => {
                    "index 'ts_idx' on column 'ts' covers 2 of the dataset's 3 fragments"
                }
                Self::SegmentWithNewerKeyOverlay | Self::NewerKeyOverlay => {
                    "index 'ts_idx' on column 'ts' cannot be read in key order: Not supported: \
                     ordered index scan cannot read segment"
                }
                Self::BitmapIndex => "column 'id' has no ordered index that this scan can read",
                Self::NestedKey => "order column 's.k' is not a top-level field",
                Self::LegacyStorage => {
                    "index 'ts_idx' on column 'ts' cannot be read in key order because the dataset \
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
                    "index 'ts_idx' on column 'ts' cannot be read in key order because the dataset \
                     uses stable row ids, so the index's fragment bitmap cannot prove that it \
                     covers every fragment"
                }
                Self::FilterOnAnotherIndex => {
                    "filter `id < Int32(100)` is not one exact search of index 'ts_idx', so an \
                     ordered read of column 'ts' cannot apply it before the limit"
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
        recommit_segment(&mut dataset, index, &fragments).await;
        dataset
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_decline_index_shapes(
        #[values(
            IndexShape::AppendedFragment,
            IndexShape::SegmentsMissingAFragment,
            IndexShape::SegmentWithNewerKeyOverlay,
            IndexShape::BitmapIndex,
            IndexShape::NestedKey,
            IndexShape::LegacyStorage,
            IndexShape::NewerKeyOverlay,
            IndexShape::MissingIndexFiles,
            IndexShape::FullTextSearch,
            IndexShape::StableRowIdsClaimingAnUpdatedFragment,
            IndexShape::FilterOnAnotherIndex
        )]
        shape: IndexShape,
    ) {
        let mut order_column = "ts";
        let mut full_text_query = None;
        let mut filter = None;
        let tempdir = TempStrDir::default();
        let dataset = match shape {
            IndexShape::AppendedFragment => {
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
            IndexShape::SegmentsMissingAFragment => segmented_dataset(&[&[0], &[2]]).await,
            IndexShape::SegmentWithNewerKeyOverlay => {
                let key_in_fragment_2 = 250;
                overlay_key(
                    segmented_dataset(&[&[0, 1], &[2]]).await,
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
            IndexShape::FilterOnAnotherIndex => {
                let mut dataset = dataset(WriteParams::default()).await;
                dataset
                    .create_index(&["id"], IndexType::BTree, None, &btree(), false)
                    .await
                    .unwrap();
                filter = Some("id < 100");
                dataset
            }
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
            if let Some(filter) = filter {
                scanner.filter(filter).unwrap();
            }
        };
        let expect = Expect::Sort {
            reason_prefix: shape.reason(),
        };
        let rows = assert_ordered_read(&dataset, configure, expect).await;
        assert_eq!(rows.num_rows(), 10, "{shape:?}");
    }

    #[tokio::test]
    async fn test_ordered_decline_names_the_index_the_scan_reads() {
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
            reason_prefix: "the scan has no limit, so an ordered read of index 'ts_idx' on \
                            column 'ts' could not stop early",
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
}
