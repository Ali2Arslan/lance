// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use arrow_schema::{SchemaRef, SortOptions};
use datafusion::catalog::Session;
use datafusion::common::Column;
use datafusion::logical_expr::{Expr, Literal, when};
use datafusion::physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    ExecutionPlan,
    expressions::{col, lit},
};
use lance_core::ROW_ADDR;
use lance_core::datatypes::OnMissing;
use lance_datafusion::planner::Planner;
use lance_file::version::ConcreteFileVersion;
use lance_index::scalar::expression::ScalarIndexExpr;
use lance_index::scalar::registry::IndexOrderingCapability;
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;

use super::{ColumnOrdering, ExprFilterPlan, FilterPlan, Scanner, TakeOperation};
use crate::Error;
use crate::index::DatasetIndexExt;
use crate::index::scalar::{IndexDetails, fetch_index_details};
use crate::index::scalar_logical::load_named_scalar_segments;
use crate::io::exec::LanceFilterExec;
use crate::io::exec::ordered_fallback::OrderedFallbackExec;
use crate::io::exec::ordered_index_scan::{
    IndexMask, MaskedRead, OrderedIndexScanExec, UnindexedFragments,
};
use crate::io::exec::ordered_merge::SharedVisitBudgetExec;
use crate::io::exec::scalar_index::ScalarIndexExec;

// On 10,000,000 rows with `LIMIT 100`, 1024 was the fastest of 256, 512, 1024, and 2048 that
// kept the read's p50 at most 0.3 times the sort's with a residual that keeps 50% of rows and
// at most 1.5 times with one that keeps 0.01%. `rows / 256` took 2.0 times the sort at 0.01%.
const ROWS_PER_RESIDUAL_CANDIDATE: u64 = 1024;
const MIN_RESIDUAL_CANDIDATES_PER_ROW: usize = 4;

// Under `OrderedIndexMode::Auto`, an index that misses fragments holding more than this share
// of the live rows declines, because the read sorts those rows. On 10,000,000 indexed rows with
// `LIMIT 100`, the merged read's p50 was at most 0.21 times the sort's with 9% of the rows
// unindexed and at most 0.58 times with 49%, in each of three rounds.
const MAX_UNINDEXED_SHARE: f64 = 0.5;

pub(super) struct OrderedRead {
    pub(super) scan: Arc<dyn ExecutionPlan>,
    pub(super) residual: Option<ResidualRead>,
    /// The limit that the read's rows must fill, or else the plan sorts
    /// instead. `None` when the read never stops short of the sort's rows.
    pub(super) fallback_limit: Option<usize>,
}

pub(super) struct ResidualRead {
    fetch: usize,
}

impl ResidualRead {
    pub(super) fn stop_at_fetch(
        &self,
        filter: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>, Error> {
        let filter = filter.downcast_ref::<LanceFilterExec>().ok_or_else(|| {
            Error::internal("the residual filter of an ordered index read is not a LanceFilterExec")
        })?;
        Ok(Arc::new(filter.with_fetch_limit(self.fetch)?))
    }
}

fn residual_budget(live_rows: impl IntoIterator<Item = usize>, fetch: usize) -> usize {
    let rows = live_rows.into_iter().fold(0_u64, |rows, fragment_rows| {
        rows.saturating_add(fragment_rows as u64)
    });
    usize::try_from(rows / ROWS_PER_RESIDUAL_CANDIDATE)
        .unwrap_or(usize::MAX)
        .max(fetch.saturating_mul(MIN_RESIDUAL_CANDIDATES_PER_ROW))
}

/// The live rows of fragments with these row counts, or `usize::MAX` when a
/// count is missing or the total does not fit, so that a fetch of it never
/// cuts off a row.
fn live_rows_bound(row_counts: impl IntoIterator<Item = Option<usize>>) -> usize {
    row_counts
        .into_iter()
        .try_fold(0_usize, |rows, fragment_rows| {
            Some(rows.saturating_add(fragment_rows?))
        })
        .unwrap_or(usize::MAX)
}

fn unindexed_share(covered: &[&Fragment], unindexed: &[&Fragment]) -> Option<f64> {
    let rows = |fragments: &[&Fragment]| -> Option<u64> {
        fragments
            .iter()
            .map(|fragment| Some(fragment.num_rows()? as u64))
            .sum()
    };
    match (rows(covered), rows(unindexed)) {
        (Some(0), _) => Some(1.0),
        (Some(covered), Some(unindexed)) => Some(unindexed as f64 / (covered + unindexed) as f64),
        _ => None,
    }
}

/// Why a scan that orders by a column sorts its rows instead of reading an
/// index on that column in key order.
///
/// With [`OrderedIndexMode::Required`], planning fails with an
/// [`Error::InvalidInput`] whose source is this value.
/// [`Scanner::explain_plan`] with `verbose` prints it after the plan under
/// [`OrderedIndexMode::Auto`] and [`OrderedIndexMode::Off`]. Under `Required`,
/// `explain_plan` fails with the same error that planning does.
#[derive(Debug)]
#[non_exhaustive]
pub enum OrderedIndexDecline {
    /// The scan orders by more than one column.
    MultipleOrderKeys { columns: Vec<String> },
    /// The scan setting in `option`, `ordered_index=off` or
    /// `use_scalar_index=false`, turns off ordered index reads.
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
    /// No index on the column can be read in key order.
    NoIndex { column: String },
    /// Every index on the column that could serve the read is named in
    /// [`Scanner::with_ignored_scalar_indices`]. `indices` lists them.
    IgnoredIndices {
        column: String,
        indices: Vec<String>,
    },
    /// The scan has no limit.
    NoLimit { column: String },
    /// The dataset or the scan has a property, described by `reason`, that
    /// the ordered read does not handle.
    UnsupportedScan {
        column: String,
        reason: Cow<'static, str>,
    },
    /// The index covers `indexed_fragments` of the dataset's `fragments`
    /// fragments. `indexed_fragments` is `None` when a segment of the index
    /// records no fragment coverage, so the count is unknown.
    /// `unindexed_fragments` counts the fragments the index misses that the
    /// scan reads: all of them, or under [`Scanner::fast_search`] those that
    /// the filter's index searches cover. `unindexed_share` is the share
    /// those fragments hold of the live rows in them and the covered
    /// fragments, or `None` when one of them records no row count. Under
    /// [`OrderedIndexMode::Auto`], the read sorts their rows and merges them
    /// in unless they hold more than half of those live rows. Under
    /// [`OrderedIndexMode::Required`], it does so at any share, unless the
    /// index covers none of the live rows.
    PartialCoverage {
        column: String,
        index: String,
        indexed_fragments: Option<usize>,
        fragments: usize,
        unindexed_fragments: usize,
        unindexed_share: Option<f64>,
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
                "{option} turns off ordered index reads for column '{column}'"
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
            Self::IgnoredIndices { column, indices } => write!(
                f,
                "every ordered index on column '{column}' is ignored by the scan: '{}'",
                indices.join("', '")
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
            Self::PartialCoverage {
                column,
                index,
                indexed_fragments: Some(indexed_fragments),
                fragments,
                unindexed_fragments,
                unindexed_share,
            } => {
                write!(
                    f,
                    "index '{index}' on column '{column}' covers {indexed_fragments} of the \
                     dataset's {fragments} fragments"
                )?;
                let (unindexed, live_rows) = if indexed_fragments + unindexed_fragments < *fragments
                {
                    (
                        format!("{unindexed_fragments} it misses that the fast search reads"),
                        format!("the live rows in those and the {indexed_fragments} it covers"),
                    )
                } else {
                    (
                        format!("{unindexed_fragments} it misses"),
                        "its live rows".to_owned(),
                    )
                };
                match unindexed_share {
                    Some(share) if *share >= 1.0 => {
                        write!(f, ", which hold none of its live rows")
                    }
                    Some(share) => write!(
                        f,
                        ", and the {unindexed} hold {:.1}% of {live_rows}, more than the {:.0}% \
                         that the read sorts and merges in",
                        share * 100.0,
                        MAX_UNINDEXED_SHARE * 100.0
                    ),
                    None => write!(
                        f,
                        ", and a fragment records no row count, so the share of live rows in the \
                         {unindexed} is unknown, and the read sorts and merges them in only when \
                         they hold at most {:.0}%",
                        MAX_UNINDEXED_SHARE * 100.0
                    ),
                }
            }
            Self::PartialCoverage {
                column,
                index,
                indexed_fragments: None,
                fragments,
                ..
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

/// Whether a scan may read a B-tree index in key order to serve `order_by`.
///
/// Parse a mode from its name with [`str::parse`], ignoring ASCII case.
///
/// ```
/// # use lance::dataset::scanner::OrderedIndexMode;
/// # use lance::Result;
/// # fn test() -> Result<()> {
/// assert_eq!("off".parse::<OrderedIndexMode>()?, OrderedIndexMode::Off);
/// assert_eq!(OrderedIndexMode::default(), OrderedIndexMode::Auto);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OrderedIndexMode {
    /// Always sort.
    Off,
    /// Read the index in key order when it can serve the ordering, and sort
    /// otherwise.
    #[default]
    Auto,
    /// Read the index in key order, and fail planning with
    /// [`Error::InvalidInput`] when the scan cannot. Unlike [`Self::Auto`],
    /// the read never sorts instead, so a selective filter, or a search of
    /// another index in the filter that selects few rows, can make it read
    /// the whole index.
    Required,
}

impl OrderedIndexMode {
    const ALL: [Self; 3] = [Self::Off, Self::Auto, Self::Required];

    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Required => "required",
        }
    }
}

impl fmt::Display for OrderedIndexMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for OrderedIndexMode {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                let accepted = Self::ALL.map(Self::name).join(", ");
                Error::invalid_input(format!(
                    "unknown ordered_index mode '{name}', expected one of: {accepted}"
                ))
            })
    }
}

impl Scanner {
    /// Set whether to read a B-tree index in key order to serve `order_by`.
    ///
    /// With [`OrderedIndexMode::Auto`], the default, a scan with a limit that
    /// orders by one column with a B-tree index reads that index in key order
    /// and stops after `offset + limit` live rows, instead of sorting every
    /// row. The scan sorts the rows of fragments the index does not cover, as
    /// rows appended since the index was last optimized, and merges them into
    /// the read, unless they hold more than half of the live rows. An exact
    /// search of the same index in the filter limits the keys the read
    /// visits. Any other index search in the filter runs first, and
    /// the read skips the rows it rules out. The scan sorts the rows that
    /// search selects instead when they are too few for the read to pay off,
    /// or when the read visits a budget of index rows without filling the
    /// limit, as when the selected rows hold the keys the read reaches last.
    /// The read applies the rest of the filter to each row it reads, and
    /// sorts instead when that keeps too few rows within a share of the
    /// index. An index named in [`Self::with_ignored_scalar_indices`] never
    /// serves the read or answers its filter. [`OrderedIndexMode::Off`]
    /// always sorts.
    ///
    /// [`OrderedIndexMode::Required`] never sorts instead. When the scan
    /// cannot serve `order_by` from an index, planning fails with
    /// [`Error::InvalidInput`]. The error's source is the
    /// [`OrderedIndexDecline`] that says why. Under `Auto` and `Off`,
    /// [`Self::explain_plan`] with `verbose` prints the same reason after the
    /// plan. Under `Required`, `explain_plan` fails with the same error that
    /// planning does. A required read skips the rows that the filter's index
    /// searches rule out and applies the rest of the filter to each row it
    /// reads, so a selective filter can make it read the whole index. It merges
    /// in the rows of fragments the index does not cover however many they
    /// are, unless the index covers none of the live rows. A scan without
    /// `order_by` is not affected. Other uses of scalar indices are not
    /// affected.
    ///
    /// ```
    /// # use lance::dataset::scanner::{ColumnOrdering, OrderedIndexMode};
    /// # use lance::{Dataset, Result};
    /// # async fn test(dataset: &Dataset) -> Result<()> {
    /// let mut scan = dataset.scan();
    /// scan.order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".to_string())]))?
    ///     .limit(Some(10), None)?
    ///     .ordered_index(OrderedIndexMode::Required);
    /// let newest_ten = scan.try_into_batch().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn ordered_index(&mut self, mode: OrderedIndexMode) -> &mut Self {
        self.ordered_index = mode;
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
        let disabled_by = if self.ordered_index == OrderedIndexMode::Off {
            Some("ordered_index=off")
        } else if !self.use_scalar_index {
            Some("use_scalar_index=false")
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
        let indices = self
            .dataset
            .load_indices()
            .await
            .map_err(|source| OrderedIndexDecline::unreadable(column, None, source))?;
        let exact_search = match &expr_filter.index_query {
            Some(ScalarIndexExpr::Query(search)) if expr_filter.skip_recheck => Some(search),
            _ => None,
        };
        // A fast search reads only the fragments that the filter's index searches cover, as
        // the sort plan does.
        let searched_fragments = match &expr_filter.index_query {
            Some(index_query) if self.fast_search => Some(
                ScalarIndexExec::fragments_covered_by_index_query(index_query, &self.dataset)
                    .await
                    .map_err(|source| OrderedIndexDecline::unreadable(column, None, source))?,
            ),
            _ => None,
        };
        let options = SortOptions {
            descending: !ordering.ascending,
            nulls_first: ordering.nulls_first,
        };
        let fragments = self.dataset.fragments();
        let live_rows = || fragments.iter().filter_map(Fragment::num_rows);
        let mut decline = OrderedIndexDecline::NoIndex {
            column: column.clone(),
        };
        let mut checked = HashSet::new();
        let mut ignored = Vec::new();
        let mut chosen = None;
        let mut order_indices = indices
            .iter()
            .filter(|index| index.keyed_field() == Some(field.id))
            .collect::<Vec<_>>();
        // The searched index goes first, so that its search limits the keys it reads
        // instead of masking a walk of another index.
        order_indices
            .sort_by_key(|index| exact_search.is_none_or(|search| search.index_name != index.name));
        for index in order_indices {
            if !checked.insert(index.name.as_str()) {
                continue;
            }
            if self.ignored_scalar_index_names.contains(&index.name) {
                ignored.push(index);
                continue;
            }
            let unreadable =
                |source| OrderedIndexDecline::unreadable(column, Some(&index.name), source);
            if !self
                .is_ordered_index(column, index)
                .await
                .map_err(unreadable)?
            {
                continue;
            }
            let segments = load_named_scalar_segments(&self.dataset, column, &index.name)
                .await
                .map_err(unreadable)?;
            let coverage = segments
                .iter()
                .filter_map(|segment| segment.fragment_bitmap.as_ref())
                .fold(RoaringBitmap::new(), |coverage, segment| coverage | segment);
            let (covered, mut unindexed): (Vec<_>, Vec<_>) = fragments
                .iter()
                .partition(|fragment| coverage.contains(fragment.id as u32));
            // `segments` leaves out the segments that record no fragment coverage.
            let is_coverage_unknown = indices
                .iter()
                .any(|segment| segment.name == index.name && segment.fragment_bitmap.is_none());
            let is_full = unindexed.is_empty();
            if segments.is_empty() || !is_full {
                let share = unindexed_share(&covered, &unindexed);
                if let Some(searched) = &searched_fragments {
                    unindexed.retain(|fragment| searched.contains(fragment.id as u32));
                }
                let merged_share = unindexed_share(&covered, &unindexed);
                let is_merged = !is_coverage_unknown
                    && share != Some(1.0)
                    && (unindexed.is_empty()
                        || self.ordered_index == OrderedIndexMode::Required
                        || merged_share.is_some_and(|share| share <= MAX_UNINDEXED_SHARE));
                if !is_merged {
                    decline = OrderedIndexDecline::PartialCoverage {
                        column: column.clone(),
                        index: index.name.clone(),
                        indexed_fragments: (!is_coverage_unknown).then_some(covered.len()),
                        fragments: fragments.len(),
                        unindexed_fragments: unindexed.len(),
                        unindexed_share: merged_share,
                    };
                    continue;
                }
            }
            if is_full || chosen.is_none() {
                chosen = Some((index, segments, unindexed));
            }
            if is_full {
                break;
            }
        }
        let Some((index, segments, unindexed)) = chosen else {
            if matches!(decline, OrderedIndexDecline::NoIndex { .. }) {
                let mut ignored_indices = Vec::new();
                for index in ignored {
                    // The scan never reads an ignored index, so an error loading one
                    // only leaves it out of the reason.
                    if self.is_ordered_index(column, index).await.unwrap_or(false) {
                        ignored_indices.push(index.name.clone());
                    }
                }
                if !ignored_indices.is_empty() {
                    return Err(OrderedIndexDecline::IgnoredIndices {
                        column: column.clone(),
                        indices: ignored_indices,
                    });
                }
            }
            return Err(decline);
        };
        let unreadable =
            |source| OrderedIndexDecline::unreadable(column, Some(&index.name), source);
        let order_search = exact_search.filter(|search| search.index_name == index.name);
        let (mask, residual) = match (order_search, &expr_filter.index_query) {
            // The search is exact for the rows the index holds, not for the fragments it misses.
            (Some(_), _) if !unindexed.is_empty() => (None, expr_filter.full_expr.clone()),
            (Some(_), _) => (None, expr_filter.refine_expr.clone()),
            (None, Some(index_query)) => {
                let read = if self.ordered_index == OrderedIndexMode::Required {
                    MaskedRead::AlwaysWalk
                } else {
                    MaskedRead::MaySort
                };
                let (mask, is_exact) = self
                    .index_mask(index_query, read)
                    .await
                    .map_err(|source| OrderedIndexDecline::unreadable(column, None, source))?;
                let residual = if is_exact {
                    &expr_filter.refine_expr
                } else {
                    &expr_filter.full_expr
                };
                (Some(mask), residual.clone())
            }
            (None, None) => (None, expr_filter.full_expr.clone()),
        };
        let scan_fetch = match &residual {
            None => Some(fetch),
            Some(_) if self.ordered_index == OrderedIndexMode::Required => None,
            Some(_) => Some(residual_budget(live_rows(), fetch)),
        };
        let fallback_limit = (self.ordered_index != OrderedIndexMode::Required
            && (residual.is_some() || mask.is_some()))
        .then(|| fetch - self.offset.unwrap_or(0) as usize);
        let query = order_search.map(|search| search.query.clone());
        let scan =
            OrderedIndexScanExec::try_new(self.dataset.clone(), segments, query, options, None)
                .await
                .map_err(unreadable)?
                .with_residual_label(residual.clone())
                .with_mask(mask.clone())
                .with_fetch(scan_fetch)
                .ok_or_else(|| {
                    unreadable(Error::internal(
                        "OrderedIndexScanExec does not accept a fetch",
                    ))
                })?;
        let (scan, residual) = if unindexed.is_empty() {
            (scan, residual)
        } else {
            let unindexed = unindexed.into_iter().cloned().collect();
            let shares_budget = mask.as_ref().is_some_and(IndexMask::may_sort);
            let merged = self
                .merge_unindexed(
                    scan,
                    unindexed,
                    expr_filter,
                    residual.as_ref(),
                    shares_budget,
                    options,
                )
                .await
                .map_err(unreadable)?;
            (merged, None)
        };
        let residual = match residual {
            None => {
                filter_plan.disable_refine();
                None
            }
            Some(residual) => {
                filter_plan.expr_filter_plan = ExprFilterPlan::new_refine_only(residual);
                Some(ResidualRead { fetch })
            }
        };
        Ok(OrderedRead {
            scan,
            residual,
            fallback_limit,
        })
    }

    /// Merges `scan` with a top-k sort of the `unindexed` fragments, which
    /// reads them with `filter` as the sort plan does, so it keeps only the
    /// rows the scan's mask and filter let through. The scan's fetch bounds
    /// the sort and the merge.
    ///
    /// The rows of `scan` that a `residual` rejects count toward that fetch,
    /// as in the read of a fully covered index, so that sorted rows cannot
    /// fill it with keys past the one where the scan stops. Each of those rows
    /// carries through the merge whether it passes the residual, and the
    /// sorted rows, which passed the whole filter in their read, all pass, so
    /// no row meets the filter twice.
    async fn merge_unindexed(
        &self,
        scan: Arc<dyn ExecutionPlan>,
        unindexed: Vec<Fragment>,
        filter: &ExprFilterPlan,
        residual: Option<&Expr>,
        shares_budget: bool,
        options: SortOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, Error> {
        let fetch = scan.fetch();
        let key = scan.schema().field(0).name().clone();
        let projection = self
            .dataset
            .empty_projection()
            .union_column(&key, OnMissing::Error)?
            .with_row_addr();
        let read = self
            .new_filtered_read(
                filter,
                projection,
                false,
                Some(Arc::new(unindexed)),
                None,
                None,
            )
            .await?;
        let order = |schema: &SchemaRef| -> Result<LexOrdering, Error> {
            Ok(LexOrdering::from([
                PhysicalSortExpr::new(col(&key, schema)?, options),
                PhysicalSortExpr::new(col(ROW_ADDR, schema)?, SortOptions::default()),
            ]))
        };
        let sorted: Arc<dyn ExecutionPlan> =
            Arc::new(SortExec::new(order(&read.schema())?, read).with_fetch(fetch));
        // Named after the key, so that it differs from both other columns.
        let passes_column = format!("{key}_passes_filter");
        let project = |input: Arc<dyn ExecutionPlan>,
                       passes: Option<Arc<dyn PhysicalExpr>>|
         -> Result<Arc<dyn ExecutionPlan>, Error> {
            let schema = input.schema();
            let mut columns = vec![
                (col(&key, &schema)?, key.clone()),
                (col(ROW_ADDR, &schema)?, ROW_ADDR.to_owned()),
            ];
            columns.extend(passes.map(|passes| (passes, passes_column.clone())));
            Ok(Arc::new(ProjectionExec::try_new(columns, input)?))
        };
        let inputs = match residual {
            None => vec![scan, sorted],
            Some(residual) => {
                let columns = self
                    .dataset
                    .empty_projection()
                    .union_columns(Planner::column_names_in_expr(residual), OnMissing::Ignore)?;
                let scan = self.take(scan, columns)?;
                // `CASE` turns a null into false, so the column is non-null like the sorted
                // rows' `true`, and the union does not cast one input to match the other.
                let passes = when(residual.clone(), true.lit()).otherwise(false.lit())?;
                let passes = Planner::new(scan.schema()).create_physical_expr(&passes)?;
                vec![
                    project(scan, Some(passes))?,
                    project(sorted, Some(lit(true)))?,
                ]
            }
        };
        let union = UnionExec::try_new(inputs)?;
        // EnforceDistribution removes a merge without a fetch as a mere change of partitions,
        // so a read without a candidate budget bounds the merge by the live rows.
        let merge_fetch = fetch.unwrap_or_else(|| {
            live_rows_bound(self.dataset.fragments().iter().map(Fragment::num_rows))
        });
        let mut merged: Arc<dyn ExecutionPlan> = Arc::new(
            SortPreservingMergeExec::new(order(&union.schema())?, union)
                .with_fetch(Some(merge_fetch)),
        );
        if shares_budget {
            merged = Arc::new(SharedVisitBudgetExec::new(merged));
        }
        if residual.is_some() {
            let passes = Expr::Column(Column::new_unqualified(&passes_column));
            merged = project(Arc::new(LanceFilterExec::try_new(passes, merged)?), None)?;
        }
        Ok(merged)
    }

    /// The mask that applies `index_query` to an ordered read as the sort plan
    /// applies it, and whether the search is exact for every row it lets
    /// through.
    async fn index_mask(
        &self,
        index_query: &ScalarIndexExpr,
        read: MaskedRead,
    ) -> Result<(IndexMask, bool), Error> {
        let fragments = self.dataset.fragments();
        let search = self
            .scalar_index_search(index_query.clone(), fragments)
            .await?;
        let stale_rows = self
            .overlay_stale_index_rows(index_query, fragments)
            .await?;
        let is_fresh = stale_rows.is_empty();
        let stale_rows = stale_rows.result_rows().await?.clone();
        let (unindexed, is_every_fragment_answered) = if self.fast_search {
            (UnindexedFragments::Skip, true)
        } else {
            let scope = search.fragment_scope();
            let is_indexed =
                |fragment: &Fragment| scope.is_some_and(|scope| scope.contains(fragment.id as u32));
            (UnindexedFragments::Emit, fragments.iter().all(is_indexed))
        };
        let is_exact = !index_query.needs_recheck() && is_fresh && is_every_fragment_answered;
        let mask = IndexMask::new(search, stale_rows, unindexed, read);
        Ok((mask, is_exact))
    }

    pub(super) async fn with_sort_fallback(
        &self,
        ordered: Arc<dyn ExecutionPlan>,
        limit: usize,
        session: Option<&dyn Session>,
    ) -> Result<Arc<dyn ExecutionPlan>, Error> {
        let mut sorted = self.clone();
        sorted.ordered_index(OrderedIndexMode::Off);
        let sorted = match session {
            Some(session) => sorted.create_plan_with_session(session).await?,
            None => sorted.create_plan().await?,
        };
        Ok(Arc::new(OrderedFallbackExec::try_new(
            ordered, sorted, limit,
        )?))
    }

    async fn is_ordered_index(&self, column: &str, index: &IndexMetadata) -> crate::Result<bool> {
        let details = IndexDetails(fetch_index_details(&self.dataset, column, index).await?);
        Ok(!details.is_vector()
            && details.get_plugin()?.ordering_capability(&details.0)
                == IndexOrderingCapability::Bidirectional)
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

// rstest drops a parametrized test's own attributes, so the allow for
// test_ordered_residual's eight parameters sits on the module.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use arrow_array::{
        Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, Int64Array,
        RecordBatch, RecordBatchIterator, StringArray, StructArray, UInt64Array,
        cast::AsArray,
        types::{Int32Type, Int64Type, UInt64Type},
    };
    use arrow_schema::{DataType, Field, Schema, SortOptions};
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::logical_expr::{ColumnarValue, Expr, Volatility, col, create_udf};
    use datafusion::physical_expr::{PhysicalSortExpr, expressions::Column};
    use datafusion::physical_plan::{ExecutionPlan, displayable};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_ADDR, ROW_ADDR_FIELD, ROW_ID, utils::address::RowAddress};
    use lance_file::version::LanceFileVersion;
    use lance_index::IndexType;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::scalar::minhash_lsh::MinHashQuery;
    use lance_index::scalar::{
        BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams,
    };
    use lance_table::format::IndexMetadata;
    use roaring::RoaringBitmap;
    use rstest::rstest;

    use super::super::{
        ColumnOrdering, ExprFilterPlan, FilterPlan, RowAddrMask, RowAddrTreeMap, Scanner,
    };
    use super::OrderedIndexMode;
    use crate::dataset::transaction::{Operation, Transaction};
    use crate::dataset::{NewColumnTransform, UpdateBuilder, WriteDestination, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::io::exec::LanceFilterExec;
    use crate::utils::test::covering::declare_covering;
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

    /// Declares `v` as a covering column of every segment of `ts_idx`.
    async fn cover_v(dataset: &mut Dataset) {
        let segments = dataset.load_indices_by_name(INDEX).await.unwrap().len();
        for _ in 0..segments {
            declare_covering(dataset, "ts", "v").await;
        }
    }

    async fn scan(
        dataset: &Dataset,
        configure: &impl Fn(&mut Scanner),
        mode: OrderedIndexMode,
    ) -> (String, RecordBatch) {
        let mut scanner = dataset.scan();
        configure(&mut scanner);
        scanner.ordered_index(mode);
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
        let (plan, rows) = scan(dataset, &configure, OrderedIndexMode::Auto).await;
        let (sorted_plan, sorted_rows) = scan(dataset, &configure, OrderedIndexMode::Off).await;
        assert!(!sorted_plan.contains("OrderedIndexScan"), "{sorted_plan}");
        let mut scanner = dataset.scan();
        configure(&mut scanner);
        let verbose_plan = scanner.explain_plan(true).await.unwrap();
        let required_plan = scanner
            .ordered_index(OrderedIndexMode::Required)
            .explain_plan(true)
            .await;
        match expect {
            Expect::OrderedScan(node) => {
                assert!(plan.contains(node) && !plan.contains("SortExec"), "{plan}");
                assert!(!verbose_plan.contains("declined"), "{verbose_plan}");
                assert_eq!(required_plan.unwrap(), verbose_plan);
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
                let planning_error = scanner.create_plan().await.unwrap_err();
                assert_eq!(error.to_string(), planning_error.to_string());
            }
        }
        assert_eq!(rows, sorted_rows);
        rows
    }

    #[rstest]
    #[case::off("off", OrderedIndexMode::Off)]
    #[case::auto("auto", OrderedIndexMode::Auto)]
    #[case::upper_case("AUTO", OrderedIndexMode::Auto)]
    #[case::mixed_case("Off", OrderedIndexMode::Off)]
    #[case::required("required", OrderedIndexMode::Required)]
    #[case::upper_case_required("REQUIRED", OrderedIndexMode::Required)]
    fn test_ordered_index_mode_parses_names(#[case] name: &str, #[case] mode: OrderedIndexMode) {
        assert_eq!(name.parse::<OrderedIndexMode>().unwrap(), mode);
        assert_eq!(mode.to_string().parse::<OrderedIndexMode>().unwrap(), mode);
    }

    #[rstest]
    #[case::unknown("sorted")]
    #[case::empty("")]
    #[case::padded(" auto")]
    fn test_ordered_index_mode_rejects_unknown_names(#[case] name: &str) {
        let error = name.parse::<OrderedIndexMode>().unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let expected =
            format!("unknown ordered_index mode '{name}', expected one of: off, auto, required");
        assert!(error.to_string().contains(&expected), "{error}");
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
        #[values(false, true)] covered: bool,
    ) {
        let mut dataset = dataset(WriteParams::default()).await;
        if covered {
            cover_v(&mut dataset).await;
        }
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
        #[values(false, true)] covered: bool,
    ) {
        let mut dataset = dataset(WriteParams::default()).await;
        if covered {
            cover_v(&mut dataset).await;
        }
        let configure = |scanner: &mut Scanner| {
            decline(order_by_ts(scanner));
        };
        let expect = Expect::Sort {
            reason_prefix: reason,
        };
        assert_ordered_read(&dataset, configure, expect).await;
    }

    #[rstest]
    #[case::projection(|scanner: &mut Scanner| {
        scanner.project(&["id", "_row_last_updated_at_version"]).unwrap();
    }, "the scan reads a row version column, which the read cannot take")]
    #[case::filter(|scanner: &mut Scanner| {
        scanner.filter("_row_created_at_version = 1").unwrap();
    }, "the filter reads a row version column, which the read cannot take")]
    #[tokio::test]
    async fn test_ordered_decline_row_version_columns(
        #[case] decline: fn(&mut Scanner),
        #[case] reason: &str,
    ) {
        let dataset = dataset(WriteParams::default()).await;
        let configure = |scanner: &mut Scanner| {
            decline(order_by_ts(scanner));
        };
        let mut plans = Vec::new();
        for mode in [OrderedIndexMode::Auto, OrderedIndexMode::Off] {
            let mut scanner = dataset.scan();
            configure(&mut scanner);
            scanner.ordered_index(mode);
            plans.push(scanner.explain_plan(false).await.unwrap());
            // Reading a row version column needs stable row ids, which make the
            // ordered read decline before it checks the columns.
            let error = scanner.try_into_batch().await.unwrap_err();
            assert!(
                matches!(error, Error::InvalidInput { .. })
                    && error.to_string().contains("requires stable row ids"),
                "{error}"
            );
        }
        assert_eq!(plans[0], plans[1]);
        let declined = format!(
            "OrderedIndexScan declined: column 'ts' cannot be read in key order from an index \
             because {reason}"
        );
        let mut scanner = dataset.scan();
        configure(&mut scanner);
        let verbose_plan = scanner.explain_plan(true).await.unwrap();
        assert!(
            verbose_plan.lines().last().unwrap().starts_with(&declined),
            "{verbose_plan}"
        );
        let error = scanner
            .ordered_index(OrderedIndexMode::Required)
            .explain_plan(false)
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }) && error.to_string().contains(&declined),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_ordered_decline_with_ordered_index_off() {
        let dataset = dataset(WriteParams::default()).await;
        let mut scanner = dataset.scan();
        order_by_ts(&mut scanner).ordered_index(OrderedIndexMode::Off);
        let plan = scanner.explain_plan(true).await.unwrap();
        assert!(
            plan.ends_with(
                "OrderedIndexScan declined: ordered_index=off turns off ordered index reads \
                 for column 'ts'\n"
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
            .ordered_index(OrderedIndexMode::Required)
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

    const ONE_SEGMENT: &[&[u32]] = &[&[0, 1, 2]];
    const SEGMENT_PER_FRAGMENT: &[&[u32]] = &[&[0], &[1], &[2]];
    const IGNORED_TS_INDEX: &str =
        "every ordered index on column 'ts' is ignored by the scan: 'ts_idx'";

    #[rstest]
    #[case::order_column_index(
        INDEX,
        ONE_SEGMENT,
        Expect::Sort { reason_prefix: IGNORED_TS_INDEX }
    )]
    #[case::order_column_index_in_three_segments(
        INDEX,
        SEGMENT_PER_FRAGMENT,
        Expect::Sort { reason_prefix: IGNORED_TS_INDEX }
    )]
    #[case::unrelated_index(
        "id_idx",
        ONE_SEGMENT,
        Expect::OrderedScan("OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=10")
    )]
    #[tokio::test]
    async fn test_ordered_read_skips_ignored_indices(
        #[case] ignored: &str,
        #[case] segment_fragments: &[&[u32]],
        #[case] expect: Expect<'static>,
        #[values(false, true)] covered: bool,
    ) {
        let mut dataset = segmented_dataset("memory://", segment_fragments).await;
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".into()),
                &btree(),
                false,
            )
            .await
            .unwrap();
        if covered {
            cover_v(&mut dataset).await;
        }
        let configure = |scanner: &mut Scanner| {
            order_by_ts(scanner).with_ignored_scalar_indices([ignored]);
        };
        assert_ordered_read(&dataset, configure, expect).await;
    }

    #[rstest]
    #[case::first_ignored(INDEX, "ts_idx_2")]
    #[case::second_ignored("ts_idx_2", INDEX)]
    #[tokio::test]
    async fn test_ordered_read_serves_a_kept_index_beside_an_ignored_one(
        #[case] ignored: &str,
        #[case] kept: &str,
    ) {
        let mut dataset = dataset(WriteParams::default()).await;
        dataset
            .create_index(
                &["ts"],
                IndexType::BTree,
                Some("ts_idx_2".into()),
                &btree(),
                false,
            )
            .await
            .unwrap();
        let configure = |scanner: &mut Scanner| {
            order_by_ts(scanner).with_ignored_scalar_indices([ignored]);
        };
        let node = format!("OrderedIndexScan: index={kept}, order=[ts ASC NULLS LAST], fetch=10");
        assert_ordered_read(&dataset, configure, Expect::OrderedScan(&node)).await;
    }

    /// Overwrites `column` with `value` in the row whose `ts` is `key`.
    async fn overlay(dataset: Dataset, column: &str, key: i32, value: i32) -> Dataset {
        let batch = dataset
            .scan()
            .project(&["ts"])
            .unwrap()
            .with_row_address()
            .filter(&format!("ts = {key}"))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch["ts"].as_primitive::<Int32Type>().value(0), key);
        let addr = batch[ROW_ADDR].as_primitive::<UInt64Type>().value(0);
        overlay_at(
            dataset,
            addr,
            column,
            Arc::new(Int32Array::from(vec![value])),
        )
        .await
    }

    async fn overlay_at(dataset: Dataset, addr: u64, column: &str, value: ArrayRef) -> Dataset {
        let field = dataset.schema().field(column).unwrap();
        let mut overlay = dataset
            .get_fragment(RowAddress::from(addr).fragment_id() as usize)
            .unwrap()
            .write_overlay(&dataset.schema().project_by_ids(&[field.id], true))
            .await
            .unwrap();
        let values = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                ROW_ADDR_FIELD.clone(),
                Field::from(field),
            ])),
            vec![Arc::new(UInt64Array::from(vec![addr])), value],
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

    #[rstest]
    #[case::one_segment(
        ONE_SEGMENT,
        1,
        "OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=10"
    )]
    #[case::every_segment_of_three(
        SEGMENT_PER_FRAGMENT,
        3,
        "OrderedIndexScan: index=ts_idx, segments=3, order=[ts ASC NULLS LAST], fetch=10"
    )]
    #[case::one_segment_of_three(
        SEGMENT_PER_FRAGMENT,
        1,
        "OrderedIndexScan: index=ts_idx, segments=3, order=[ts ASC NULLS LAST], fetch=10"
    )]
    #[tokio::test]
    async fn test_ordered_read_serves_a_covered_index_by_its_key(
        #[case] segment_fragments: &[&[u32]],
        #[case] covered_segments: usize,
        #[case] node: &str,
    ) {
        let mut dataset = segmented_dataset("memory://", segment_fragments).await;
        let (ts_id, id_id) = declare_covering(&mut dataset, "ts", "id").await;
        for _ in 1..covered_segments {
            declare_covering(&mut dataset, "ts", "id").await;
        }
        let segments = dataset.load_indices_by_name(INDEX).await.unwrap();
        let covered = segments
            .iter()
            .filter(|segment| segment.fields == [ts_id, id_id]);
        assert_eq!(covered.count(), covered_segments);
        let configure = |scanner: &mut Scanner| {
            order_by_ts(scanner.project(&["id", "ts"]).unwrap());
        };
        assert_ordered_read(&dataset, configure, Expect::OrderedScan(node)).await;

        let dataset = overlay(dataset, "id", 4, -4).await;
        let rows = assert_ordered_read(&dataset, configure, Expect::OrderedScan(node)).await;
        assert!(
            rows["id"]
                .as_primitive::<Int32Type>()
                .values()
                .contains(&-4)
        );

        let dataset = overlay(dataset, "ts", 6, 1000).await;
        let expect = Expect::Sort {
            reason_prefix: IndexShape::NewerKeyOverlay.reason(),
        };
        assert_ordered_read(&dataset, configure, expect).await;
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
        FragmentSubsetOfPartialIndex,
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
                Self::FragmentSubsetOfPartialIndex => {
                    "column 'ts' cannot be read in key order from an index because the scan reads \
                     a list of 3 fragments that differs from the dataset's 4 fragments"
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
            IndexShape::FragmentSubsetOfPartialIndex,
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
        #[values(false, true)] covered: bool,
    ) {
        let mut order_column = "ts";
        let mut full_text_query = None;
        let tempdir = TempStrDir::default();
        let mut dataset = match shape {
            IndexShape::FragmentSubsetOfPartialIndex => {
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
                overlay(
                    segmented_dataset("memory://", &[&[0, 1], &[2]]).await,
                    "ts",
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
                overlay(dataset(WriteParams::default()).await, "ts", 150, -1).await
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
        if covered {
            cover_v(&mut dataset).await;
        }
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

    const MASK_ROWS: i32 = 10_000;
    const MASK_FRAGMENT_ROWS: usize = 2_500;
    const OLD_TS: i32 = 400;

    fn mask_rows(ids: Range<i32>) -> RecordBatch {
        let status = ids.clone().map(|id| match id {
            id if id % 2 == 0 => Some("p50"),
            id if id % 100 == 1 => Some("p1"),
            id if id % 10_000 == 3 => Some("p001"),
            id if id % 10 == 5 => None,
            _ => Some("other"),
        });
        let ts = ids
            .clone()
            .map(|id| {
                if id < MASK_ROWS {
                    id * 7919 % MASK_ROWS
                } else {
                    id
                }
            })
            .collect::<Vec<_>>();
        let age = StringArray::from_iter_values(
            ts.iter().map(|&ts| if ts < OLD_TS { "old" } else { "new" }),
        );
        RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(Int32Array::from_iter_values(ids.clone())) as ArrayRef,
            ),
            ("ts", Arc::new(Int32Array::from(ts))),
            (
                "v",
                Arc::new(Int64Array::from_iter_values(ids.map(i64::from))),
            ),
            ("status", Arc::new(StringArray::from_iter(status))),
            ("age", Arc::new(age)),
        ])
        .unwrap()
    }

    fn mask_addr(id: i32) -> u64 {
        let fragment = id as usize / MASK_FRAGMENT_ROWS;
        let offset = id as usize % MASK_FRAGMENT_ROWS;
        RowAddress::new_from_parts(fragment as u32, offset as u32).into()
    }

    #[derive(Clone, Copy, Debug)]
    enum MaskData {
        Indexed,
        IndexedId,
        Deleted,
        MostlyDeleted,
        Segmented,
        UnindexedLastFragment,
        StaleStatus,
    }

    impl MaskData {
        async fn dataset(self) -> Dataset {
            let batch = mask_rows(0..MASK_ROWS);
            let mut dataset = Dataset::write(
                RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                "memory://",
                Some(WriteParams {
                    max_rows_per_file: MASK_FRAGMENT_ROWS,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            for column in ["status", "age"] {
                dataset
                    .create_index(
                        &[column],
                        IndexType::Bitmap,
                        Some(format!("{column}_idx")),
                        &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
                        false,
                    )
                    .await
                    .unwrap();
            }
            let zone_map = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
                .with_params(&serde_json::json!({"rows_per_zone": 512}));
            dataset
                .create_index(
                    &["v"],
                    IndexType::ZoneMap,
                    Some("v_idx".into()),
                    &zone_map,
                    false,
                )
                .await
                .unwrap();
            if let Self::UnindexedLastFragment = self {
                let batch = mask_rows(MASK_ROWS..MASK_ROWS + 500);
                dataset
                    .append(
                        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                        None,
                    )
                    .await
                    .unwrap();
            }
            if let Self::Segmented = self {
                let fragment_ids = dataset
                    .fragments()
                    .iter()
                    .map(|fragment| fragment.id as u32)
                    .collect::<Vec<_>>();
                let mut segments = Vec::new();
                for fragment_id in fragment_ids {
                    let segment = dataset
                        .create_index_builder(&["ts"], IndexType::BTree, &btree())
                        .name(INDEX.into())
                        .fragments(vec![fragment_id])
                        .execute_uncommitted()
                        .await
                        .unwrap();
                    segments.push(segment);
                }
                dataset
                    .commit_existing_index_segments(INDEX, "ts", segments)
                    .await
                    .unwrap();
            } else {
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
            }
            match self {
                Self::IndexedId => {
                    dataset
                        .create_index(
                            &["id"],
                            IndexType::BTree,
                            Some("id_idx".into()),
                            &btree(),
                            false,
                        )
                        .await
                        .unwrap();
                }
                Self::Deleted => {
                    dataset.delete("id % 5 = 0").await.unwrap();
                }
                Self::MostlyDeleted => {
                    dataset
                        .delete("id % 2 = 0 OR (status = 'p1' AND id >= 2000)")
                        .await
                        .unwrap();
                }
                Self::StaleStatus => {
                    let (now_p50, no_longer_p50) = Self::newest_ids_outside_and_in_p50();
                    for (id, status) in [(now_p50, "p50"), (no_longer_p50, "other")] {
                        let value = Arc::new(StringArray::from(vec![status]));
                        dataset = overlay_at(dataset, mask_addr(id), "status", value).await;
                    }
                }
                _ => {}
            }
            dataset
        }

        fn newest_ids_outside_and_in_p50() -> (i32, i32) {
            let newest = |is_p50: bool| {
                (0..MASK_ROWS)
                    .filter(|id| (id % 2 == 0) == is_p50)
                    .max_by_key(|id| id * 7919 % MASK_ROWS)
                    .unwrap()
            };
            (newest(false), newest(true))
        }
    }

    #[derive(Debug)]
    struct MaskRow {
        id: i32,
        ts: i32,
        v: i64,
        status: Option<String>,
        fragment: u32,
    }

    impl MaskRow {
        fn status(&self) -> Option<&str> {
            self.status.as_deref()
        }
    }

    async fn mask_oracle(
        dataset: &Dataset,
        keep: fn(&MaskRow) -> bool,
        limit: usize,
        offset: usize,
    ) -> Vec<i32> {
        let batch = dataset
            .scan()
            .project(&["id", "ts", "v", "status"])
            .unwrap()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let ids = batch["id"].as_primitive::<Int32Type>();
        let ts = batch["ts"].as_primitive::<Int32Type>();
        let v = batch["v"].as_primitive::<Int64Type>();
        let status = batch["status"].as_string::<i32>();
        let addrs = batch[ROW_ADDR].as_primitive::<UInt64Type>();
        let mut kept = (0..batch.num_rows())
            .map(|row| MaskRow {
                id: ids.value(row),
                ts: ts.value(row),
                v: v.value(row),
                status: status.is_valid(row).then(|| status.value(row).to_owned()),
                fragment: RowAddress::from(addrs.value(row)).fragment_id(),
            })
            .filter(keep)
            .collect::<Vec<_>>();
        kept.sort_by_key(|row| std::cmp::Reverse(row.ts));
        kept.into_iter()
            .skip(offset)
            .take(limit)
            .map(|row| row.id)
            .collect()
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum MaskPath {
        Ordered,
        Sorted,
        SpendsBudget,
        Required,
    }

    const STATUS_MASK: &str = "mask=[status = p50]@status_idx(Bitmap)";

    #[rstest]
    #[case::half(MaskData::Indexed, "status = 'p50'", |row: &MaskRow| row.status() == Some("p50"), (10, 0), false, MaskPath::Ordered, STATUS_MASK)]
    #[case::one_percent(MaskData::Indexed, "status = 'p1'", |row: &MaskRow| row.status() == Some("p1"), (10, 0), false, MaskPath::Ordered, "mask=[status = p1]@status_idx(Bitmap)")]
    #[case::hundredth_percent(MaskData::Indexed, "status = 'p001'", |row: &MaskRow| row.status() == Some("p001"), (10, 0), false, MaskPath::Sorted, "mask=[status = p001]@status_idx(Bitmap)")]
    #[case::required_small_mask(MaskData::Indexed, "status = 'p001'", |row: &MaskRow| row.status() == Some("p001"), (10, 0), false, MaskPath::Required, "mask=[status = p001]@status_idx(Bitmap)")]
    #[case::inexact(MaskData::Indexed, "v < 700", |row: &MaskRow| row.v < 700, (10, 0), false, MaskPath::Ordered, "mask=[v < 700]@v_idx(ZoneMap), residual=v < Int64(700)")]
    #[case::search_of_another_btree(MaskData::IndexedId, "id < 5000", |row: &MaskRow| row.id < 5000, (10, 0), false, MaskPath::Ordered, "mask=[id < 5000]@id_idx(BTree)")]
    #[case::two_searches_of_the_order_index(MaskData::Indexed, "ts < 100 OR ts > 9900", |row: &MaskRow| row.ts < 100 || row.ts > 9900, (10, 0), false, MaskPath::Ordered, "mask=OR([ts < 100]@ts_idx(BTree),[ts > 9900]@ts_idx(BTree))")]
    #[case::mask_and_residual(MaskData::Indexed, "status = 'p50' AND id % 3 = 0", |row: &MaskRow| row.status() == Some("p50") && row.id % 3 == 0, (10, 0), false, MaskPath::Ordered, "mask=[status = p50]@status_idx(Bitmap), residual=CAST(id AS Int64) % Int64(3) = Int64(0)")]
    #[case::not_equal_with_nulls(MaskData::Indexed, "status != 'p50'", |row: &MaskRow| row.status().is_some_and(|status| status != "p50"), (10, 0), false, MaskPath::Ordered, "mask=NOT([status = p50]@status_idx(Bitmap))")]
    #[case::is_null(MaskData::Indexed, "status IS NULL", |row: &MaskRow| row.status().is_none(), (10, 0), false, MaskPath::Ordered, "mask=[status IS NULL]@status_idx(Bitmap)")]
    #[case::deletions(MaskData::Deleted, "status = 'p50'", |row: &MaskRow| row.status() == Some("p50"), (10, 0), false, MaskPath::Ordered, STATUS_MASK)]
    #[case::segments(MaskData::Segmented, "status = 'p50'", |row: &MaskRow| row.status() == Some("p50"), (10, 0), false, MaskPath::Ordered, STATUS_MASK)]
    #[case::unindexed_fragment(MaskData::UnindexedLastFragment, "status = 'p50'", |row: &MaskRow| row.status() == Some("p50"), (10, 0), false, MaskPath::Ordered, r#"mask=[status = p50]@status_idx(Bitmap), residual=status = Utf8("p50")"#)]
    #[case::unindexed_fragment_fast_search(MaskData::UnindexedLastFragment, "status = 'p50'", |row: &MaskRow| row.fragment < 4 && row.status() == Some("p50"), (10, 0), true, MaskPath::SpendsBudget, STATUS_MASK)]
    #[case::stale_overlay(MaskData::StaleStatus, "status = 'p50'", |row: &MaskRow| row.status() == Some("p50"), (10, 0), false, MaskPath::Ordered, r#"mask=[status = p50]@status_idx(Bitmap), residual=status = Utf8("p50")"#)]
    #[case::offset(MaskData::Indexed, "status = 'p50'", |row: &MaskRow| row.status() == Some("p50"), (5, 3), false, MaskPath::Ordered, STATUS_MASK)]
    #[case::limit_1023(MaskData::Indexed, "status = 'p50' AND id % 3 != 0", |row: &MaskRow| row.status() == Some("p50") && row.id % 3 != 0, (1023, 0), false, MaskPath::Ordered, "mask=[status = p50]@status_idx(Bitmap), residual=CAST(id AS Int64) % Int64(3) != Int64(0)")]
    #[case::limit_1024(MaskData::Indexed, "status = 'p50' AND id % 3 != 0", |row: &MaskRow| row.status() == Some("p50") && row.id % 3 != 0, (1024, 0), false, MaskPath::Ordered, "mask=[status = p50]@status_idx(Bitmap), residual=CAST(id AS Int64) % Int64(3) != Int64(0)")]
    #[case::limit_1025(MaskData::Indexed, "status = 'p50' AND id % 3 != 0", |row: &MaskRow| row.status() == Some("p50") && row.id % 3 != 0, (1025, 0), false, MaskPath::Ordered, "mask=[status = p50]@status_idx(Bitmap), residual=CAST(id AS Int64) % Int64(3) != Int64(0)")]
    #[case::limit_2100(MaskData::Indexed, "status = 'p50' AND id % 3 != 0", |row: &MaskRow| row.status() == Some("p50") && row.id % 3 != 0, (2100, 0), false, MaskPath::Ordered, "mask=[status = p50]@status_idx(Bitmap), residual=CAST(id AS Int64) % Int64(3) != Int64(0)")]
    #[case::oldest_spends_the_budget(MaskData::Indexed, "age = 'old'", |row: &MaskRow| row.ts < OLD_TS, (10, 0), false, MaskPath::SpendsBudget, "mask=[age = old]@age_idx(Bitmap)")]
    #[case::oldest_required(MaskData::Indexed, "age = 'old'", |row: &MaskRow| row.ts < OLD_TS, (10, 0), false, MaskPath::Required, "mask=[age = old]@age_idx(Bitmap)")]
    #[case::oldest_segments_share_the_budget(MaskData::Segmented, "age = 'old'", |row: &MaskRow| row.ts < OLD_TS, (20, 0), false, MaskPath::SpendsBudget, "mask=[age = old]@age_idx(Bitmap)")]
    #[case::mostly_deleted_sorts(MaskData::MostlyDeleted, "status = 'p1'", |row: &MaskRow| row.status() == Some("p1"), (10, 0), false, MaskPath::Sorted, "mask=[status = p1]@status_idx(Bitmap)")]
    #[tokio::test]
    async fn test_ordered_mask(
        #[case] data: MaskData,
        #[case] filter: &str,
        #[case] keep: fn(&MaskRow) -> bool,
        #[case] limit_offset: (i64, i64),
        #[case] fast_search: bool,
        #[case] path: MaskPath,
        #[case] scan_line_end: &str,
        #[values(false, true)] covered: bool,
    ) {
        let mut dataset = data.dataset().await;
        if covered {
            cover_v(&mut dataset).await;
            if let MaskData::IndexedId = data {
                // Carrying `ts` does not make `id_idx` an index on `ts`.
                declare_covering(&mut dataset, "id", "ts").await;
            }
        }
        let (limit, offset) = limit_offset;
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .filter(filter)
                .unwrap()
                .order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".into())]))
                .unwrap()
                .limit(Some(limit), Some(offset))
                .unwrap()
                .batch_size(1024)
                .target_parallelism(4);
            if fast_search {
                scanner.fast_search();
            }
        };
        let expected = mask_oracle(&dataset, keep, limit as usize, offset as usize).await;
        assert!(!expected.is_empty(), "{filter} keeps live rows");
        if let MaskData::StaleStatus = data {
            let (now_p50, no_longer_p50) = MaskData::newest_ids_outside_and_in_p50();
            assert!(
                expected.contains(&now_p50) && !expected.contains(&no_longer_p50),
                "{expected:?}"
            );
        }

        for required in [path == MaskPath::Required, true] {
            let mut scanner = dataset.scan();
            configure(&mut scanner);
            scanner.ordered_index(if required {
                OrderedIndexMode::Required
            } else {
                OrderedIndexMode::Auto
            });
            let plan = scanner.explain_plan(false).await.unwrap();
            let scan_line = plan
                .lines()
                .find(|line| line.trim_start().starts_with("OrderedIndexScan: "))
                .unwrap_or_else(|| panic!("{plan}"));
            assert!(scan_line.ends_with(scan_line_end), "{plan}");
            assert_eq!(plan.contains("OrderedFallback"), !required, "{plan}");
            assert!(!required || !plan.contains("SortExec"), "{plan}");
            if !required {
                let analyzed = scanner.analyze_plan().await.unwrap();
                let scan_metrics = analyzed
                    .lines()
                    .find(|line| line.contains("OrderedIndexScan: "))
                    .unwrap();
                let (fallback, walks) = match path {
                    MaskPath::Ordered => ("ordered_fallback=false", true),
                    MaskPath::SpendsBudget => ("ordered_fallback=true", true),
                    MaskPath::Sorted | MaskPath::Required => ("ordered_fallback=true", false),
                };
                assert!(
                    analyzed.contains(fallback) && scan_metrics.contains("mask_rows="),
                    "{analyzed}"
                );
                assert_eq!(scan_metrics.contains("visit_budget="), walks, "{analyzed}");
                assert_eq!(
                    !scan_metrics.contains("rows_scanned=0"),
                    walks,
                    "{analyzed}"
                );
            }
            let rows = scanner.try_into_batch().await.unwrap();
            assert_eq!(ids(&rows), expected, "{plan}");
        }

        let mut sorted = dataset.scan();
        configure(&mut sorted);
        sorted.ordered_index(OrderedIndexMode::Off);
        let sorted_plan = sorted.explain_plan(false).await.unwrap();
        assert!(!sorted_plan.contains("OrderedIndexScan"), "{sorted_plan}");
        assert_eq!(ids(&sorted.try_into_batch().await.unwrap()), expected);
    }

    #[rstest]
    #[case::large_mask_walks_the_index(
        MaskData::Indexed,
        "p50",
        "mask_rows=5.00 K",
        "ordered_fallback=false"
    )]
    #[case::small_mask_sorts(MaskData::Indexed, "p001", "mask_rows=1", "ordered_fallback=true")]
    #[case::live_rows_only(MaskData::MostlyDeleted, "p1", "mask_rows=20", "ordered_fallback=true")]
    #[tokio::test]
    async fn test_ordered_mask_reports_rows_and_path(
        #[case] data: MaskData,
        #[case] status: &str,
        #[case] mask_rows: &str,
        #[case] fallback: &str,
        #[values(false, true)] covered: bool,
    ) {
        let mut dataset = data.dataset().await;
        if covered {
            cover_v(&mut dataset).await;
        }
        let mut scanner = dataset.scan();
        scanner
            .project(&["id"])
            .unwrap()
            .filter(&format!("status = '{status}'"))
            .unwrap()
            .order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".into())]))
            .unwrap()
            .limit(Some(10), None)
            .unwrap();
        let analyzed = scanner.analyze_plan().await.unwrap();
        let scan_line = analyzed
            .lines()
            .find(|line| line.contains("OrderedIndexScan: "))
            .unwrap();
        assert!(
            scan_line.contains(&format!(
                "mask=[status = {status}]@status_idx(Bitmap), metrics=["
            )) && scan_line.contains(mask_rows)
                && analyzed.contains(fallback),
            "{analyzed}"
        );
    }

    #[rstest]
    #[case::mask_index("status = 'p50'", |row: &MaskRow| row.status() == Some("p50"), r#"residual=status = Utf8("p50")"#)]
    #[case::one_of_two_mask_indices("status = 'p50' AND age = 'old'", |row: &MaskRow| row.status() == Some("p50") && row.ts < OLD_TS, r#"mask=[age = old]@age_idx(Bitmap), residual=status = Utf8("p50")"#)]
    #[tokio::test]
    async fn test_ordered_mask_skips_ignored_indices(
        #[case] filter: &str,
        #[case] keep: fn(&MaskRow) -> bool,
        #[case] scan_line_end: &str,
    ) {
        let dataset = MaskData::Indexed.dataset().await;
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .filter(filter)
                .unwrap()
                .order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".into())]))
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .with_ignored_scalar_indices(["status_idx"]);
        };
        let expected = mask_oracle(&dataset, keep, 10, 0).await;
        assert!(!expected.is_empty(), "{filter} keeps live rows");
        for mode in [
            OrderedIndexMode::Auto,
            OrderedIndexMode::Required,
            OrderedIndexMode::Off,
        ] {
            let mut scanner = dataset.scan();
            configure(&mut scanner);
            scanner.ordered_index(mode);
            let plan = scanner.explain_plan(false).await.unwrap();
            assert!(!plan.contains("status_idx"), "{plan}");
            let scan_line = plan
                .lines()
                .find(|line| line.trim_start().starts_with("OrderedIndexScan: "));
            match mode {
                OrderedIndexMode::Off => assert!(scan_line.is_none(), "{plan}"),
                _ => assert!(
                    scan_line.is_some_and(|line| line.ends_with(scan_line_end)),
                    "{plan}"
                ),
            }
            assert_eq!(
                ids(&scanner.try_into_batch().await.unwrap()),
                expected,
                "{plan}"
            );
        }

        let ignore_ts_idx = |scanner: &mut Scanner| {
            configure(scanner);
            scanner.with_ignored_scalar_indices(["status_idx", INDEX]);
        };
        let expect = Expect::Sort {
            reason_prefix: IGNORED_TS_INDEX,
        };
        assert_ordered_read(&dataset, ignore_ts_idx, expect).await;
    }

    #[rstest]
    #[tokio::test]
    async fn test_ordered_mask_from_another_index_on_the_order_column(
        #[values(false, true)] covered: bool,
    ) {
        let mut dataset = write("memory://", WriteParams::default()).await;
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
        if covered {
            cover_v(&mut dataset).await;
        }
        dataset
            .create_index(
                &["ts"],
                IndexType::Bitmap,
                Some("ts_bitmap".into()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
                false,
            )
            .await
            .unwrap();
        let bitmap_segments = dataset.load_indices_by_name("ts_bitmap").await.unwrap();
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id"])
                .unwrap()
                .filter("ts = 42")
                .unwrap()
                .order_by(Some(vec![ColumnOrdering::asc_nulls_last("ts".into())]))
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .with_scalar_index_selection("ts_bitmap".into(), bitmap_segments.clone(), None);
        };
        let mut sorted = dataset.scan();
        configure(&mut sorted);
        sorted.ordered_index(OrderedIndexMode::Off);
        let expected = sorted.try_into_batch().await.unwrap();
        assert_eq!(expected.num_rows(), 1);

        for required in [false, true] {
            let mut scanner = dataset.scan();
            configure(&mut scanner);
            scanner.ordered_index(if required {
                OrderedIndexMode::Required
            } else {
                OrderedIndexMode::Auto
            });
            let plan = scanner.explain_plan(false).await.unwrap();
            assert!(
                plan.contains("OrderedIndexScan: index=ts_idx")
                    && plan.contains("mask=[ts = 42]@ts_bitmap(Bitmap)")
                    && plan.contains("ScalarIndexQuery: query=[ts = 42]@ts_bitmap(Bitmap)")
                    && plan.contains("OrderedFallback") != required,
                "{plan}"
            );
            assert_eq!(scanner.try_into_batch().await.unwrap(), expected, "{plan}");
        }
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
        AllNullColumn,
        Many,
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
                Self::AllNullColumn => {
                    let mut dataset = dataset(WriteParams::default()).await;
                    let column = Field::new("w", DataType::Int64, true);
                    dataset
                        .add_columns(
                            NewColumnTransform::AllNulls(Arc::new(Schema::new(vec![column]))),
                            None,
                            None,
                        )
                        .await
                        .unwrap();
                    dataset
                }
                Self::Many => {
                    let ids = 0..8192;
                    let batch = RecordBatch::try_from_iter([
                        (
                            "id",
                            Arc::new(Int32Array::from_iter_values(ids.clone())) as ArrayRef,
                        ),
                        (
                            "ts",
                            Arc::new(Int32Array::from_iter_values(
                                ids.clone().map(|id| id * 37 % 8192),
                            )),
                        ),
                        (
                            "v",
                            Arc::new(Int64Array::from_iter_values(ids.clone().map(i64::from))),
                        ),
                        (
                            "text",
                            Arc::new(StringArray::from_iter_values(
                                ids.map(|id| if id % 2 == 0 { "even" } else { "odd" }),
                            )),
                        ),
                    ])
                    .unwrap();
                    let mut dataset = Dataset::write(
                        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                        "memory://",
                        Some(WriteParams {
                            max_rows_per_file: 4096,
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
                    dataset
                }
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
    #[case::required_search_and_residual(
        Rows::Indexed,
        "ts > 100 AND text = 'odd'",
        |row: &Row| row.ts.is_some_and(|ts| ts > 100) && row.text == "odd",
        ColumnOrdering::asc_nulls_last("ts".into()),
        (10, 0),
        Read::Required,
        r#"OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], query=ts > 100, residual=text = Utf8("odd")"#
    )]
    #[case::all_null_column(
        Rows::AllNullColumn,
        "w IS NULL",
        |_: &Row| true,
        ColumnOrdering::desc_nulls_last("ts".into()),
        (10, 0),
        Read::Ordered,
        "OrderedIndexScan: index=ts_idx, order=[ts DESC NULLS LAST], fetch=40, residual=w IS NULL"
    )]
    #[case::row_share(
        Rows::Many,
        "ts % 8 = 6",
        |row: &Row| row.ts.is_some_and(|ts| ts % 8 == 6),
        ColumnOrdering::asc_nulls_last("ts".into()),
        (1, 0),
        Read::Ordered,
        "OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=8, residual=CAST(ts AS Int64) % Int64(8) = Int64(6)"
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
        #[values(false, true)] covered: bool,
    ) {
        let mut dataset = rows.dataset().await;
        if covered {
            cover_v(&mut dataset).await;
        }
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
                .ordered_index(match read {
                    Read::Required => OrderedIndexMode::Required,
                    Read::Ordered | Read::FallsBack => OrderedIndexMode::Auto,
                });
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
        let mut filters = Vec::new();
        scanner
            .create_plan()
            .await
            .unwrap()
            .apply(|node| {
                if let Some(filter) = node.downcast_ref::<LanceFilterExec>() {
                    filters.push((filter.expr().to_string(), filter.filter.fetch()));
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
        let (_, residual) = scan_line.split_once("residual=").unwrap();
        assert!(
            filters.contains(&(residual.to_owned(), Some((limit + offset) as usize))),
            "{filters:?}"
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
        sorted.ordered_index(OrderedIndexMode::Off);
        let sorted_plan = sorted.explain_plan(false).await.unwrap();
        assert!(!sorted_plan.contains("OrderedIndexScan"), "{sorted_plan}");
        assert_eq!(ids(&sorted.try_into_batch().await.unwrap()), expected);
    }

    #[rstest]
    #[case::filtered_limit(|scanner: &mut Scanner| {
        scanner.filter("text = 'even'").unwrap().limit(Some(10), None).unwrap();
    }, &[
        "ProjectionExec: expr=[id@0 as id, ts@1 as ts]",
        r#"  LanceRead: uri=data, projection=[id, ts, text], num_fragments=3, range_before=None, range_after=Some(0..10), row_id=true, row_addr=false, full_filter=text = Utf8("even"), refine_filter=text = Utf8("even")"#,
    ])]
    #[case::filtered_take_limit(|scanner: &mut Scanner| {
        scanner
            .filter("_rowid IN (1, 2, 3, 4) AND text = 'even'")
            .unwrap()
            .limit(Some(2), None)
            .unwrap();
    }, &[
        "ProjectionExec: expr=[id@0 as id, ts@1 as ts]",
        "  GlobalLimitExec: skip=0, fetch=2",
        "    FilterExec: text@3 = even",
        "      LanceRead: uri=data, projection=[text], source=stream(_rowaddr)",
        "        LanceRead: uri=data, projection=[id, ts], num_fragments=3, range_before=None, range_after=None, row_id=false, row_addr=true, full_filter=--, refine_filter=--",
        "          RecordBatchExec: rows=2, columns=3",
    ])]
    #[case::ordered_read(|scanner: &mut Scanner| {
        order_by_ts(scanner);
    }, &[
        "ProjectionExec: expr=[id@2 as id, ts@0 as ts]",
        "  LanceRead: uri=data, projection=[id], source=stream(_rowaddr)",
        "    OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=10",
    ])]
    #[case::ordered_search(|scanner: &mut Scanner| {
        order_by_ts(scanner).filter("ts BETWEEN 20 AND 80").unwrap();
    }, &[
        "ProjectionExec: expr=[id@2 as id, ts@0 as ts]",
        "  LanceRead: uri=data, projection=[id], source=stream(_rowaddr)",
        "    OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=10, query=ts >= 20 && ts <= 80",
    ])]
    #[tokio::test]
    async fn test_ordered_residual_absent_keeps_plans(
        #[case] configure: fn(&mut Scanner),
        #[case] expected: &[&str],
    ) {
        let dataset = dataset(WriteParams::default()).await;
        let mut scanner = dataset.scan();
        scanner
            .project(&["id", "ts"])
            .unwrap()
            .target_parallelism(1);
        configure(&mut scanner);
        let plan = scanner.explain_plan(false).await.unwrap();
        assert_eq!(plan.lines().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_ordered_residual_budget_saturates() {
        assert_eq!(
            super::residual_budget([usize::MAX, usize::MAX], 1),
            usize::try_from(u64::MAX / 1024).unwrap_or(usize::MAX)
        );
    }

    #[rstest]
    #[case::saturates(&[Some(usize::MAX - 1), Some(2)], usize::MAX)]
    #[case::missing_count(&[Some(1), None, Some(2)], usize::MAX)]
    #[case::sums(&[Some(3), Some(4)], 7)]
    fn test_ordered_partial_merge_fetch_covers_the_live_rows(
        #[case] row_counts: &[Option<usize>],
        #[case] expected: usize,
    ) {
        assert_eq!(super::live_rows_bound(row_counts.iter().copied()), expected);
    }

    #[tokio::test]
    async fn test_ordered_residual_limits_rows_across_partitions() {
        let dataset = Rows::Many.dataset().await;
        let mut scanner = dataset.scan();
        scanner
            .project(&["id", "ts"])
            .unwrap()
            .filter("text = 'even'")
            .unwrap()
            .order_by(Some(vec![ColumnOrdering::asc_nulls_last("ts".into())]))
            .unwrap()
            .limit(Some(2100), None)
            .unwrap()
            .batch_size(1024)
            .target_parallelism(4);
        let mut sorted = scanner.clone();
        sorted.ordered_index(OrderedIndexMode::Off);
        let analyzed = scanner.analyze_plan().await.unwrap();
        assert!(
            analyzed.contains("RoundRobinBatch(4)") && analyzed.contains("ordered_fallback=false"),
            "{analyzed}"
        );
        let rows = scanner.try_into_batch().await.unwrap();
        assert_eq!(rows.num_rows(), 2100);
        assert_eq!(rows, sorted.try_into_batch().await.unwrap());
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
            .ordered_index(OrderedIndexMode::Off)
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(expected.schema().metadata()["format"], "test");
        assert_eq!(scanner.try_into_batch().await.unwrap(), expected);
    }

    #[tokio::test]
    async fn test_ordered_residual_with_ignored_indices() {
        let mut dataset = dataset(WriteParams::default()).await;
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".into()),
                &btree(),
                false,
            )
            .await
            .unwrap();
        let configure = |scanner: &mut Scanner| {
            order_by_ts(scanner)
                .filter("id < 100")
                .unwrap()
                .with_ignored_scalar_indices(["id_idx"]);
        };
        let (plan, rows) = scan(&dataset, &configure, OrderedIndexMode::Auto).await;
        let (sorted_plan, sorted_rows) = scan(&dataset, &configure, OrderedIndexMode::Off).await;
        let fallback = sorted_plan
            .lines()
            .map(|line| format!("  {line}\n"))
            .collect::<String>();
        assert!(
            plan.starts_with("OrderedFallback: limit=10\n")
                && plan.contains(
                    "OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], fetch=40, \
                     residual=id < Int32(100)"
                )
                && plan.ends_with(&fallback),
            "{plan}"
        );
        let (required_plan, required_rows) =
            scan(&dataset, &configure, OrderedIndexMode::Required).await;
        assert!(
            required_plan.contains(
                "OrderedIndexScan: index=ts_idx, order=[ts ASC NULLS LAST], \
                 residual=id < Int32(100)"
            ) && !required_plan.contains("OrderedFallback"),
            "{required_plan}"
        );
        assert_eq!(rows, sorted_rows);
        assert_eq!(required_rows, sorted_rows);

        let configure = |scanner: &mut Scanner| {
            order_by_ts(scanner)
                .filter("ts > 5")
                .unwrap()
                .with_ignored_scalar_indices([INDEX]);
        };
        let expect = Expect::Sort {
            reason_prefix: IGNORED_TS_INDEX,
        };
        assert_ordered_read(&dataset, configure, expect).await;
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
        expected
            .use_scalar_index(false)
            .ordered_index(OrderedIndexMode::Off);
        let ordered = limit.is_some() && shape != DatasetShape::StableRowIds;
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
        expected
            .use_scalar_index(false)
            .ordered_index(OrderedIndexMode::Off);
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

    const PARTIAL_ROWS: i32 = 4_000;
    const PARTIAL_FRAGMENT_ROWS: usize = 1_000;
    const PARTIAL_APPENDED: i32 = 445;

    // 7919 is coprime to 20,000, so every id below 20,000 gets its own key.
    fn interleaved_key(id: i32) -> Option<i32> {
        Some(id * 7919 % 20_000)
    }

    // The 10 indexed rows with interleaved keys of 19,900 and up, and the 221 appended rows
    // below them.
    fn is_few(id: i32) -> bool {
        id % 2 == 1 && (id < PARTIAL_ROWS) == interleaved_key(id).is_some_and(|key| key >= 19_900)
    }

    fn partial_rows(ids: Range<i32>, key: fn(i32) -> Option<i32>) -> RecordBatch {
        let status = ids.clone().map(|id| match id {
            id if id % 2 == 0 => "p50",
            id if is_few(id) => "few",
            _ => "other",
        });
        RecordBatch::try_from_iter_with_nullable([
            (
                "id",
                Arc::new(Int32Array::from_iter_values(ids.clone())) as ArrayRef,
                false,
            ),
            ("ts", Arc::new(Int32Array::from_iter(ids.map(key))), true),
            (
                "status",
                Arc::new(StringArray::from_iter_values(status)),
                false,
            ),
        ])
        .unwrap()
    }

    async fn append(dataset: &mut Dataset, batch: RecordBatch) {
        dataset
            .append(
                RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                Some(WriteParams {
                    max_rows_per_file: PARTIAL_FRAGMENT_ROWS,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
    }

    async fn index_status(dataset: &mut Dataset) {
        dataset
            .create_index(
                &["status"],
                IndexType::Bitmap,
                Some("status_idx".into()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
                false,
            )
            .await
            .unwrap();
    }

    async fn index_ts_segments(dataset: &mut Dataset, segment_fragments: &[&[u32]]) {
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
    }

    #[derive(Clone, Copy, Debug)]
    enum PartialData {
        Appended(i32),
        Deleted,
        NewNulls,
        NewKeysAboveTheIndex,
        Segmented,
        MaskMissesNewRows,
        MaskCoversNewRows,
        /// The mask selects too few indexed rows to walk, and the appended
        /// rows it selects fill the limit with keys below them.
        SmallMask,
        UnknownCoverage,
        NoRowCount,
        CoversNoRows,
    }

    impl PartialData {
        async fn dataset(self, tempdir: &TempStrDir) -> Dataset {
            let batch = partial_rows(0..PARTIAL_ROWS, interleaved_key);
            let mut dataset = Dataset::write(
                RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                tempdir.as_str(),
                Some(WriteParams {
                    max_rows_per_file: PARTIAL_FRAGMENT_ROWS,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            match self {
                Self::Segmented | Self::UnknownCoverage => {
                    index_ts_segments(&mut dataset, &[&[0, 1], &[2, 3]]).await;
                }
                Self::CoversNoRows => index_ts_segments(&mut dataset, &[&[0]]).await,
                _ => {
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
                }
            }
            if let Self::MaskMissesNewRows = self {
                index_status(&mut dataset).await;
            }
            if let Self::UnknownCoverage = self {
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
                dataset = Dataset::open(tempdir.as_str()).await.unwrap();
                recommit_segment(&mut dataset, segment, None).await;
            }
            let (appended, key): (i32, fn(i32) -> Option<i32>) = match self {
                Self::Appended(rows) => (rows, interleaved_key),
                Self::NewNulls => (PARTIAL_APPENDED, |id| {
                    interleaved_key(id).filter(|_| id % 5 != 0)
                }),
                Self::NewKeysAboveTheIndex => (PARTIAL_APPENDED, |id| Some(20_000 + id)),
                Self::CoversNoRows => (0, interleaved_key),
                _ => (PARTIAL_APPENDED, interleaved_key),
            };
            if appended > 0 {
                let batch = partial_rows(PARTIAL_ROWS..PARTIAL_ROWS + appended, key);
                append(&mut dataset, batch).await;
            }
            match self {
                Self::Deleted => {
                    dataset.delete("id % 7 = 3").await.unwrap();
                }
                Self::MaskCoversNewRows | Self::SmallMask => index_status(&mut dataset).await,
                Self::NoRowCount => {
                    dataset
                        .delete(&format!("id >= {PARTIAL_ROWS} AND id % 9 = 0"))
                        .await
                        .unwrap();
                    let mut manifest = dataset.manifest.as_ref().clone();
                    let mut fragments = manifest.fragments.as_ref().clone();
                    let appended = fragments.last_mut().unwrap();
                    appended.deletion_file.as_mut().unwrap().num_deleted_rows = None;
                    assert_eq!(appended.num_rows(), None);
                    manifest.fragments = Arc::new(fragments);
                    dataset.manifest = Arc::new(manifest);
                }
                Self::CoversNoRows => {
                    dataset.delete("id < 1000").await.unwrap();
                }
                _ => {}
            }
            dataset
        }
    }

    #[derive(Debug)]
    struct PartialRow {
        id: i32,
        ts: Option<i32>,
        status: String,
    }

    async fn partial_oracle(
        dataset: &Dataset,
        keep: fn(&PartialRow) -> bool,
        ordering: &ColumnOrdering,
    ) -> Vec<(Option<i32>, i32)> {
        let batch = dataset
            .scan()
            .project(&["id", "ts", "status"])
            .unwrap()
            .use_scalar_index(false)
            .try_into_batch()
            .await
            .unwrap();
        let (ids, ts, status) = (
            batch["id"].as_primitive::<Int32Type>(),
            batch["ts"].as_primitive::<Int32Type>(),
            batch["status"].as_string::<i32>(),
        );
        let mut rows = (0..batch.num_rows())
            .map(|row| PartialRow {
                id: ids.value(row),
                ts: ts.is_valid(row).then(|| ts.value(row)),
                status: status.value(row).to_owned(),
            })
            .filter(keep)
            .map(|row| (row.ts, row.id))
            .collect::<Vec<_>>();
        rows.sort_by(|(a, _), (b, _)| match (a, b) {
            (Some(a), Some(b)) if ordering.ascending => a.cmp(b),
            (Some(a), Some(b)) => b.cmp(a),
            (a, b) if ordering.nulls_first => a.is_some().cmp(&b.is_some()),
            (a, b) => b.is_some().cmp(&a.is_some()),
        });
        rows
    }

    /// Asserts that `batch` holds rows `offset..offset + limit` of `sorted`,
    /// where rows with equal keys may come in any order.
    fn assert_window(
        batch: &RecordBatch,
        sorted: &[(Option<i32>, i32)],
        offset: usize,
        limit: usize,
        plan: &str,
    ) {
        let ts = batch["ts"].as_primitive::<Int32Type>();
        let actual = ts.iter().zip(ids(batch)).collect::<Vec<_>>();
        let window = &sorted[offset.min(sorted.len())..(offset + limit).min(sorted.len())];
        let keys =
            |rows: &[(Option<i32>, i32)]| rows.iter().map(|(key, _)| *key).collect::<Vec<_>>();
        assert_eq!(keys(&actual), keys(window), "{plan}");
        let mut ids_by_key = HashMap::<_, HashSet<_>>::new();
        for (key, id) in sorted {
            ids_by_key.entry(*key).or_default().insert(*id);
        }
        let mut seen = HashSet::new();
        for (key, id) in actual {
            assert!(
                ids_by_key[&key].contains(&id) && seen.insert(id),
                "{id} {plan}"
            );
        }
    }

    fn plan_children<'a>(lines: &[&'a str], parent: usize) -> Vec<(usize, &'a str)> {
        let depth = indent(lines[parent]);
        lines[parent + 1..]
            .iter()
            .take_while(|line| indent(line) > depth)
            .enumerate()
            .filter(|(_, line)| indent(line) == depth + 2)
            .map(|(offset, line)| (parent + 1 + offset, line.trim()))
            .collect()
    }

    fn indent(line: &str) -> usize {
        line.len() - line.trim_start().len()
    }

    fn assert_merge(
        plan: &str,
        sorted_plan: &str,
        ordering: &ColumnOrdering,
        unindexed: usize,
        live_rows: usize,
        shares_budget: bool,
    ) {
        let lines = plan.lines().collect::<Vec<_>>();
        let parent = |child: usize| {
            lines[..child]
                .iter()
                .rposition(|line| indent(line) < indent(lines[child]))
                .unwrap_or_else(|| panic!("{plan}"))
        };
        let union = lines
            .iter()
            .position(|line| line.trim() == "UnionExec")
            .unwrap_or_else(|| panic!("{plan}"));
        let merge = parent(union);
        assert_eq!(plan_children(&lines, merge).len(), 1, "{plan}");
        assert_eq!(
            lines[parent(merge)].trim() == "SharedVisitBudget",
            shares_budget,
            "{plan}"
        );
        let scan = lines
            .iter()
            .position(|line| line.trim().starts_with("OrderedIndexScan: index=ts_idx, "))
            .unwrap_or_else(|| panic!("{plan}"));
        let fetch = lines[scan]
            .split(", ")
            .find_map(|part| part.strip_prefix("fetch="))
            .map(|fetch| format!("fetch={fetch}"));
        let has_residual = lines[scan].contains(", residual=");
        // The residual runs once on each row: on the scan's rows before the merge, which
        // carries the result, and on the sorted rows in their read.
        let mut ancestors = vec![];
        let mut node = merge;
        while indent(lines[node]) > 0 {
            node = parent(node);
            ancestors.push(lines[node].trim());
        }
        let filters = ancestors
            .into_iter()
            .filter(|line| line.starts_with("FilterExec: "))
            .collect::<Vec<_>>();
        let expected_filters: &[&str] = if has_residual {
            &["FilterExec: ts_passes_filter@2"]
        } else {
            &[]
        };
        assert_eq!(filters, expected_filters, "{plan}");
        let inputs = plan_children(&lines, union);
        let [(covered, covered_line), (sort, sorted)] = inputs[..] else {
            panic!("{plan}");
        };
        let (sort, sorted) = if has_residual {
            let (prefix, suffix) = (
                "ProjectionExec: expr=[ts@0 as ts, _rowaddr@1 as _rowaddr, ",
                " as ts_passes_filter]",
            );
            assert!(
                covered_line.starts_with(prefix)
                    && covered_line.ends_with(suffix)
                    && covered < scan
                    && scan < sort,
                "{plan}"
            );
            assert_eq!(sorted, format!("{prefix}true{suffix}"), "{plan}");
            let [sort] = plan_children(&lines, sort)[..] else {
                panic!("{plan}");
            };
            sort
        } else {
            assert_eq!(covered, scan, "{plan}");
            (sort, sorted)
        };
        let key = PhysicalSortExpr::new(
            Arc::new(Column::new("ts", 0)),
            SortOptions {
                descending: !ordering.ascending,
                nulls_first: ordering.nulls_first,
            },
        );
        let order = format!("[{key}, _rowaddr@1 ASC]");
        let merge_fetch = fetch.clone().unwrap_or(format!("fetch={live_rows}"));
        let expected_merge = format!("SortPreservingMergeExec: {order}, {merge_fetch}");
        assert_eq!(lines[merge].trim(), expected_merge, "{plan}");
        let expected_sort = match &fetch {
            Some(fetch) => {
                format!("SortExec: TopK({fetch}), expr={order}, preserve_partitioning=[false]")
            }
            None => format!("SortExec: expr={order}, preserve_partitioning=[false]"),
        };
        assert_eq!(sorted, expected_sort, "{plan}");
        let [(read, line)] = plan_children(&lines, sort)[..] else {
            panic!("{plan}");
        };
        assert!(
            line.starts_with("LanceRead: ")
                && line.contains(&format!("projection=[ts], num_fragments={unindexed}, "))
                && line.contains("row_addr=true, "),
            "{plan}"
        );
        // The read applies the sort plan's filter and index search.
        let sorted_lines = sorted_plan.lines().collect::<Vec<_>>();
        let sorted_read = sorted_lines
            .iter()
            .position(|line| {
                line.trim().starts_with("LanceRead: ") && line.contains("full_filter=")
            })
            .unwrap_or_else(|| panic!("{sorted_plan}"));
        let filter = |line: &str| {
            line.split_once(", full_filter=")
                .map(|(_, filter)| filter.to_owned())
        };
        assert_eq!(
            filter(line),
            filter(sorted_lines[sorted_read]),
            "{plan}\n{sorted_plan}"
        );
        let index_input = |lines: &[&str], read| {
            plan_children(lines, read)
                .into_iter()
                .map(|(_, line)| line.to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            index_input(&lines, read),
            index_input(&sorted_lines, sorted_read),
            "{plan}\n{sorted_plan}"
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Partial {
        Merges,
        /// As `Merges`, but under `auto` the residual keeps too few of the
        /// merged rows within its budget, or the mask stops the read, so the
        /// scan sorts.
        MergesThenSorts,
        DeclinesUnderAuto(&'static str),
        Declines(&'static str),
    }

    const P50: fn(&PartialRow) -> bool = |row| row.status == "p50";
    const ALL: fn(&PartialRow) -> bool = |_| true;

    #[rstest]
    #[case::one_percent(PartialData::Appended(40), None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::Merges)]
    #[case::ten_percent(PartialData::Appended(PARTIAL_APPENDED), None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::Merges)]
    #[case::forty_nine_percent(PartialData::Appended(3_843), None, ALL, ColumnOrdering::asc_nulls_last("ts".into()), (100, 0), Partial::Merges)]
    #[case::sixty_percent(PartialData::Appended(6_000), None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::DeclinesUnderAuto("index 'ts_idx' on column 'ts' covers 4 of the dataset's 10 fragments, and the 6 it misses hold 60.0% of its live rows, more than the 50% that the read sorts and merges in"))]
    #[case::deletions_on_both_sides(PartialData::Deleted, None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::Merges)]
    #[case::residual(PartialData::Appended(PARTIAL_APPENDED), Some("id % 3 = 0"), |row: &PartialRow| row.id % 3 == 0, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::Merges)]
    #[case::residual_budget_covers_the_merge(PartialData::Appended(PARTIAL_APPENDED), Some("id % 5 = 1"), |row: &PartialRow| row.id % 5 == 1, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::MergesThenSorts)]
    #[case::search_of_the_order_index(PartialData::Appended(PARTIAL_APPENDED), Some("ts < 19000"), |row: &PartialRow| row.ts.is_some_and(|ts| ts < 19_000), ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::Merges)]
    #[case::new_nulls_first(PartialData::NewNulls, None, ALL, ColumnOrdering::asc_nulls_first("ts".into()), (100, 0), Partial::Merges)]
    #[case::new_nulls_last(PartialData::NewNulls, None, ALL, ColumnOrdering::asc_nulls_last("ts".into()), (100, 4_400), Partial::Merges)]
    #[case::new_keys_above_the_index(PartialData::NewKeysAboveTheIndex, None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::Merges)]
    #[case::segments(PartialData::Segmented, None, ALL, ColumnOrdering::desc_nulls_first("ts".into()), (100, 0), Partial::Merges)]
    #[case::offset(PartialData::Appended(PARTIAL_APPENDED), None, ALL, ColumnOrdering::asc_nulls_last("ts".into()), (50, 37), Partial::Merges)]
    #[case::limit_1023(PartialData::Appended(PARTIAL_APPENDED), None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (1023, 0), Partial::Merges)]
    #[case::limit_1024(PartialData::Appended(PARTIAL_APPENDED), None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (1024, 0), Partial::Merges)]
    #[case::limit_1025(PartialData::Appended(PARTIAL_APPENDED), None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (1025, 0), Partial::Merges)]
    #[case::limit_2100(PartialData::Appended(PARTIAL_APPENDED), None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (2100, 0), Partial::Merges)]
    #[case::mask(PartialData::MaskCoversNewRows, Some("status = 'p50'"), P50, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::Merges)]
    #[case::mask_misses_new_rows(PartialData::MaskMissesNewRows, Some("status = 'p50'"), P50, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::Merges)]
    #[case::small_mask(PartialData::SmallMask, Some("status = 'few'"), |row: &PartialRow| row.status == "few", ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::MergesThenSorts)]
    #[case::unknown_coverage(PartialData::UnknownCoverage, None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::Declines("index 'ts_idx' on column 'ts' has a segment that records no fragment coverage, so it cannot prove that it covers the dataset's 5 fragments"))]
    #[case::covers_no_rows(PartialData::CoversNoRows, None, ALL, ColumnOrdering::desc_nulls_last("ts".into()), (100, 0), Partial::Declines("index 'ts_idx' on column 'ts' covers 0 of the dataset's 3 fragments, which hold none of its live rows"))]
    #[tokio::test]
    async fn test_ordered_partial(
        #[case] data: PartialData,
        #[case] filter: Option<&str>,
        #[case] keep: fn(&PartialRow) -> bool,
        #[case] ordering: ColumnOrdering,
        #[case] limit_offset: (i64, i64),
        #[case] expect: Partial,
    ) {
        let tempdir = TempStrDir::default();
        let dataset = data.dataset(&tempdir).await;
        let (limit, offset) = limit_offset;
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .order_by(Some(vec![ordering.clone()]))
                .unwrap()
                .limit(Some(limit), Some(offset))
                .unwrap()
                .batch_size(1024)
                .target_parallelism(4);
            if let Some(filter) = filter {
                scanner.filter(filter).unwrap();
            }
        };
        let sorted_rows = partial_oracle(&dataset, keep, &ordering).await;
        let (limit, offset) = (limit as usize, offset as usize);
        assert!(
            sorted_rows.len() > offset,
            "{data:?} keeps rows past the offset"
        );
        let mut sorted = dataset.scan();
        configure(&mut sorted);
        sorted.ordered_index(OrderedIndexMode::Off);
        let sorted_plan = sorted.explain_plan(false).await.unwrap();
        assert!(!sorted_plan.contains("OrderedIndexScan"), "{sorted_plan}");
        let rows = sorted.try_into_batch().await.unwrap();
        assert_window(&rows, &sorted_rows, offset, limit, &sorted_plan);

        let coverage = dataset
            .load_indices_by_name(INDEX)
            .await
            .unwrap()
            .iter()
            .filter_map(|segment| segment.fragment_bitmap.clone())
            .fold(RoaringBitmap::new(), |coverage, segment| coverage | segment);
        let unindexed = dataset
            .fragments()
            .iter()
            .filter(|fragment| !coverage.contains(fragment.id as u32))
            .count();
        let live_rows = dataset.count_rows(None).await.unwrap();
        let is_masked = matches!(
            data,
            PartialData::MaskCoversNewRows
                | PartialData::MaskMissesNewRows
                | PartialData::SmallMask
        );
        for mode in [OrderedIndexMode::Auto, OrderedIndexMode::Required] {
            let mut scanner = dataset.scan();
            configure(&mut scanner);
            scanner.ordered_index(mode);
            let declined = match (expect, mode) {
                (Partial::Declines(reason), _)
                | (Partial::DeclinesUnderAuto(reason), OrderedIndexMode::Auto) => Some(reason),
                _ => None,
            };
            let plan = match declined {
                Some(reason) => {
                    let declined = format!("OrderedIndexScan declined: {reason}");
                    if mode == OrderedIndexMode::Required {
                        let error = scanner.create_plan().await.unwrap_err();
                        assert!(
                            matches!(error, Error::InvalidInput { .. })
                                && error.to_string().contains(&declined),
                            "{error}"
                        );
                        continue;
                    }
                    let plan = scanner.explain_plan(false).await.unwrap();
                    assert_eq!(plan, sorted_plan);
                    let verbose = scanner.explain_plan(true).await.unwrap();
                    assert_eq!(verbose.lines().last().unwrap(), declined, "{verbose}");
                    plan
                }
                None => {
                    let plan = scanner.explain_plan(false).await.unwrap();
                    let is_auto = mode == OrderedIndexMode::Auto;
                    let shares_budget = is_auto && is_masked;
                    assert_merge(
                        &plan,
                        &sorted_plan,
                        &ordering,
                        unindexed,
                        live_rows,
                        shares_budget,
                    );
                    let has_fallback = is_auto && (filter.is_some() || is_masked);
                    assert_eq!(
                        plan.starts_with("OrderedFallback: "),
                        has_fallback,
                        "{plan}"
                    );
                    if has_fallback {
                        let analyzed = scanner.analyze_plan().await.unwrap();
                        let sorts = matches!(expect, Partial::MergesThenSorts);
                        assert!(
                            analyzed.contains(&format!("ordered_fallback={sorts}")),
                            "{analyzed}"
                        );
                        let counts_rows = analyzed.lines().any(|line| {
                            line.trim_start().starts_with("SharedVisitBudget, ")
                                && line.contains("output_rows=")
                        });
                        assert_eq!(counts_rows, shares_budget, "{analyzed}");
                    }
                    plan
                }
            };
            let rows = scanner.try_into_batch().await.unwrap();
            assert_window(&rows, &sorted_rows, offset, limit, &plan);
            if let PartialData::NewKeysAboveTheIndex = data {
                assert!(ids(&rows).iter().all(|&id| id >= PARTIAL_ROWS), "{plan}");
            }
        }
    }

    /// Under `fast_search`, the sort plan reads only the fragments that the
    /// filter's index searches cover, so the read merges in none of the
    /// others, whatever shape the search has. An index that covers none of
    /// the live rows still declines; the filter then searches no index, so
    /// the sort reads every fragment.
    #[rstest]
    #[case::search_of_the_order_index(PartialData::Appended(PARTIAL_APPENDED), "ts < 19000", |row: &PartialRow| row.id < PARTIAL_ROWS && row.ts.is_some_and(|ts| ts < 19_000), Ok(("OrderedIndexScan: index=ts_idx, order=[ts DESC NULLS LAST], fetch=100, query=ts < 19000", false)))]
    #[case::negated_search(PartialData::Appended(6_000), "ts != 7", |row: &PartialRow| row.id < PARTIAL_ROWS && row.ts.is_some_and(|ts| ts != 7), Ok(("OrderedIndexScan: index=ts_idx, order=[ts DESC NULLS LAST], fetch=", true)))]
    #[case::index_covers_no_live_rows(PartialData::CoversNoRows, "ts = 7", |row: &PartialRow| row.ts == Some(7), Err("index 'ts_idx' on column 'ts' covers 0 of the dataset's 3 fragments, which hold none of its live rows"))]
    #[tokio::test]
    async fn test_ordered_partial_fast_search_reads_the_covered_fragments(
        #[case] data: PartialData,
        #[case] filter: &str,
        #[case] keep: fn(&PartialRow) -> bool,
        #[case] expect: Result<(&str, bool), &str>,
        #[values(OrderedIndexMode::Auto, OrderedIndexMode::Required)] mode: OrderedIndexMode,
    ) {
        let tempdir = TempStrDir::default();
        let dataset = data.dataset(&tempdir).await;
        let ordering = ColumnOrdering::desc_nulls_last("ts".into());
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .filter(filter)
                .unwrap()
                .order_by(Some(vec![ordering.clone()]))
                .unwrap()
                .limit(Some(100), None)
                .unwrap()
                .fast_search();
        };
        let sorted_rows = partial_oracle(&dataset, keep, &ordering).await;
        let mut sorted = dataset.scan();
        configure(&mut sorted);
        sorted.ordered_index(OrderedIndexMode::Off);
        let sorted_plan = sorted.explain_plan(false).await.unwrap();
        assert_window(
            &sorted.try_into_batch().await.unwrap(),
            &sorted_rows,
            0,
            100,
            &sorted_plan,
        );
        let mut scanner = dataset.scan();
        configure(&mut scanner);
        scanner.ordered_index(mode);
        let plan = match expect {
            Ok((scan, has_fallback_under_auto)) => {
                let plan = scanner.explain_plan(false).await.unwrap();
                assert!(
                    plan.contains(scan)
                        && !plan.contains("UnionExec")
                        && plan.contains("OrderedFallback")
                            == (has_fallback_under_auto && mode == OrderedIndexMode::Auto),
                    "{plan}"
                );
                plan
            }
            Err(reason) => {
                let declined = format!("OrderedIndexScan declined: {reason}");
                if mode == OrderedIndexMode::Required {
                    let error = scanner.create_plan().await.unwrap_err();
                    assert!(
                        matches!(error, Error::InvalidInput { .. })
                            && error.to_string().contains(&declined),
                        "{error}"
                    );
                    return;
                }
                let plan = scanner.explain_plan(false).await.unwrap();
                assert_eq!(plan, sorted_plan);
                let verbose = scanner.explain_plan(true).await.unwrap();
                assert_eq!(verbose.lines().last().unwrap(), declined, "{verbose}");
                plan
            }
        };
        assert_window(
            &scanner.try_into_batch().await.unwrap(),
            &sorted_rows,
            0,
            100,
            &plan,
        );
    }

    /// The sort plan and the merged read both fail to plan when a fragment
    /// records no row count, because DataFusion asks the read of that
    /// fragment for its row count, so no rows can be compared. This checks
    /// the ordered read's decision, and that the scan fails as
    /// `ordered_index("off")` does.
    #[rstest]
    #[tokio::test]
    async fn test_ordered_partial_fragment_without_row_count(
        #[values(OrderedIndexMode::Auto, OrderedIndexMode::Required)] mode: OrderedIndexMode,
    ) {
        let tempdir = TempStrDir::default();
        let dataset = PartialData::NoRowCount.dataset(&tempdir).await;
        let ordering = [ColumnOrdering::desc_nulls_last("ts".into())];
        let scan = |mode| {
            let mut scanner = dataset.scan();
            scanner
                .order_by(Some(ordering.to_vec()))
                .unwrap()
                .limit(Some(100), None)
                .unwrap()
                .ordered_index(mode);
            scanner
        };
        for mode in [mode, OrderedIndexMode::Off] {
            let error = scan(mode).try_into_batch().await.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("Fragments are missing row count stats"),
                "{mode}: {error}"
            );
        }
        let mut filter_plan = FilterPlan::new(None, ExprFilterPlan::default());
        let read = scan(mode)
            .try_ordered_read(&ordering, &mut filter_plan)
            .await;
        match (mode, read) {
            (OrderedIndexMode::Auto, Err(decline)) => assert_eq!(
                decline.to_string(),
                "OrderedIndexScan declined: index 'ts_idx' on column 'ts' covers 4 of the \
                 dataset's 5 fragments, and a fragment records no row count, so the share of \
                 live rows in the 1 it misses is unknown, and the read sorts and merges them \
                 in only when they hold at most 50%"
            ),
            (OrderedIndexMode::Required, Ok(read)) => {
                let plan = displayable(read.scan.as_ref()).indent(true).to_string();
                assert!(
                    plan.starts_with("SortPreservingMergeExec: ")
                        && plan.contains("fetch=100")
                        && plan.contains("UnionExec"),
                    "{plan}"
                );
            }
            (mode, read) => panic!("{mode}: {:?}", read.err()),
        }
    }

    /// The masked read visits the rejected newest keys of the indexed rows,
    /// stops short of the matches below them, and must not let the matches in
    /// the appended rows, older than all of those, fill the limit.
    #[rstest]
    #[tokio::test]
    async fn test_ordered_partial_mask_stops_short(
        #[values(false, true)] mask_covers_new_rows: bool,
    ) {
        let key_and_status = |id: i32| match id {
            0..6_000 => (200 + id, "other"),
            6_000..6_100 => (id - 5_900, "match"),
            6_100..7_000 => (6_099 - id, "match"),
            _ => (id - 7_000, "match"),
        };
        let batch = |ids: Range<i32>| {
            RecordBatch::try_from_iter([
                (
                    "id",
                    Arc::new(Int32Array::from_iter_values(ids.clone())) as ArrayRef,
                ),
                (
                    "ts",
                    Arc::new(Int32Array::from_iter_values(
                        ids.clone().map(|id| key_and_status(id).0),
                    )),
                ),
                (
                    "status",
                    Arc::new(StringArray::from_iter_values(
                        ids.map(|id| key_and_status(id).1),
                    )),
                ),
            ])
            .unwrap()
        };
        let indexed = batch(0..7_000);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(indexed.clone())], indexed.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: PARTIAL_FRAGMENT_ROWS,
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
        if !mask_covers_new_rows {
            index_status(&mut dataset).await;
        }
        append(&mut dataset, batch(7_000..7_100)).await;
        if mask_covers_new_rows {
            index_status(&mut dataset).await;
        }
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .filter("status = 'match'")
                .unwrap()
                .order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".into())]))
                .unwrap()
                .limit(Some(100), None)
                .unwrap();
        };
        let expected = (100..200).rev().collect::<Vec<_>>();
        for mode in [
            OrderedIndexMode::Auto,
            OrderedIndexMode::Required,
            OrderedIndexMode::Off,
        ] {
            let mut scanner = dataset.scan();
            configure(&mut scanner);
            scanner.ordered_index(mode);
            let plan = scanner.explain_plan(false).await.unwrap();
            assert_eq!(
                plan.contains("SharedVisitBudget"),
                mode == OrderedIndexMode::Auto,
                "{plan}"
            );
            if mode == OrderedIndexMode::Auto {
                let analyzed = scanner.analyze_plan().await.unwrap();
                assert!(
                    analyzed.contains("ordered_fallback=true")
                        && analyzed.contains("visit_budget="),
                    "{analyzed}"
                );
            }
            let rows = scanner.try_into_batch().await.unwrap();
            let ts = rows["ts"].as_primitive::<Int32Type>().values().to_vec();
            assert_eq!(ts, expected, "{plan}");
        }
    }

    /// A filter on `_rowid` that holds the first time it sees a row and fails
    /// after that, so a row that meets it twice drops out. `seen` records
    /// each row id it sees.
    fn true_once(seen: Arc<Mutex<Vec<u64>>>) -> Expr {
        let udf = create_udf(
            "true_once",
            vec![DataType::UInt64],
            DataType::Boolean,
            Volatility::Volatile,
            Arc::new(move |args: &[ColumnarValue]| {
                let ids = ColumnarValue::values_to_arrays(args)?;
                let mut seen = seen.lock().unwrap();
                let passes = ids[0]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .map(|&id| {
                        let is_first = !seen.contains(&id);
                        seen.push(id);
                        is_first
                    })
                    .collect::<Vec<_>>();
                Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(passes))))
            }),
        );
        udf.call(vec![col(ROW_ID)])
    }

    /// The read evaluates a volatile filter once on each row, as the sort plan
    /// does: on the indexed rows before the merge and on the appended rows in
    /// their read, whichever holds the newest row.
    #[rstest]
    #[case::appended_row_first(|id| Some(id))]
    #[case::indexed_row_first(|id| Some(id - 200))]
    #[tokio::test]
    async fn test_ordered_partial_evaluates_a_volatile_filter_once(
        #[case] appended_key: fn(i32) -> Option<i32>,
    ) {
        let indexed = partial_rows(0..100, Some);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(indexed.clone())], indexed.schema()),
            "memory://",
            None,
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
        append(&mut dataset, partial_rows(100..200, appended_key)).await;
        let ordering = ColumnOrdering::desc_nulls_last("ts".into());
        let newest = partial_oracle(&dataset, ALL, &ordering).await[0];
        for mode in [
            OrderedIndexMode::Auto,
            OrderedIndexMode::Required,
            OrderedIndexMode::Off,
        ] {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let mut scanner = dataset.scan();
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .filter_expr(true_once(seen.clone()))
                .order_by(Some(vec![ordering.clone()]))
                .unwrap()
                .limit(Some(1), None)
                .unwrap()
                .ordered_index(mode);
            let plan = scanner.explain_plan(false).await.unwrap();
            assert_eq!(
                plan.contains("UnionExec"),
                mode != OrderedIndexMode::Off,
                "{plan}"
            );
            let rows = scanner.try_into_batch().await.unwrap();
            let ts = rows["ts"].as_primitive::<Int32Type>();
            assert_eq!(
                ts.iter().zip(ids(&rows)).collect::<Vec<_>>(),
                [newest],
                "{mode}: {plan}"
            );
            let seen = seen.lock().unwrap();
            assert_eq!(
                seen.iter().collect::<HashSet<_>>().len(),
                seen.len(),
                "{mode} evaluates the filter twice on a row: {seen:?}"
            );
            let fragments = seen
                .iter()
                .map(|&id| RowAddress::from(id).fragment_id())
                .collect::<HashSet<_>>();
            assert_eq!(fragments, HashSet::from([0, 1]), "{mode}: {seen:?}");
        }
    }

    /// Under `fast_search`, the read sorts and merges in only the fragments
    /// its index misses that the filter's index covers, so a decline names
    /// those and the share they hold of the rows the scan reads.
    #[tokio::test]
    async fn test_ordered_partial_fast_search_decline_names_the_fragments_it_reads() {
        let batch = partial_rows(0..90, interleaved_key);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        index_ts_segments(&mut dataset, &[&[0, 1, 2, 3]]).await;
        index_status(&mut dataset).await;
        append(&mut dataset, partial_rows(90..100, interleaved_key)).await;
        assert_eq!(dataset.fragments().len(), 10);
        let ordering = ColumnOrdering::desc_nulls_last("ts".into());
        let configure = |scanner: &mut Scanner| {
            scanner
                .project(&["id", "ts"])
                .unwrap()
                .filter("status = 'p50'")
                .unwrap()
                .order_by(Some(vec![ordering.clone()]))
                .unwrap()
                .limit(Some(100), None)
                .unwrap()
                .fast_search();
        };
        let sorted_rows = partial_oracle(
            &dataset,
            |row| row.id < 90 && row.status == "p50",
            &ordering,
        )
        .await;
        let mut sorted = dataset.scan();
        configure(&mut sorted);
        sorted.ordered_index(OrderedIndexMode::Off);
        let sorted_plan = sorted.explain_plan(false).await.unwrap();
        let rows = sorted.try_into_batch().await.unwrap();
        assert_window(&rows, &sorted_rows, 0, 100, &sorted_plan);
        for mode in [OrderedIndexMode::Auto, OrderedIndexMode::Required] {
            let mut scanner = dataset.scan();
            configure(&mut scanner);
            scanner.ordered_index(mode);
            let plan = scanner.explain_plan(false).await.unwrap();
            if mode == OrderedIndexMode::Auto {
                assert_eq!(plan, sorted_plan);
                let verbose = scanner.explain_plan(true).await.unwrap();
                assert_eq!(
                    verbose.lines().last().unwrap(),
                    "OrderedIndexScan declined: index 'ts_idx' on column 'ts' covers 4 of the \
                     dataset's 10 fragments, and the 5 it misses that the fast search reads hold \
                     55.6% of the live rows in those and the 4 it covers, more than the 50% that \
                     the read sorts and merges in",
                    "{verbose}"
                );
            } else {
                assert!(plan.contains("UnionExec"), "{plan}");
            }
            let rows = scanner.try_into_batch().await.unwrap();
            assert_window(&rows, &sorted_rows, 0, 100, &plan);
        }
    }
}
