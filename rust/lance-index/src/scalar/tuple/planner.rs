// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plan a conjunctive filter as a [`TupleQuery`] over ordered key columns.
//!
//! [`plan_tuple_predicate`] reads each conjunct on its own. It returns `None`
//! when no conjunct binds the first key, and the caller keeps every conjunct.
//! Otherwise it returns a [`TuplePlan`] that sorts each conjunct by these rules:
//!
//! - `key = v`, `key IN (...)`, `key IS NULL`, and an OR whose terms are all
//!   `key = v` or `key IN (...)` on one key fix a key to a set of values. A
//!   bare Boolean key means `key = true`, and `NOT key` means `key = false`,
//!   alone or as an OR term. [`Planner::optimize_expr`] writes a short
//!   `key IN (...)` list as such an OR, `key = true` as `key`, `key = false`
//!   as `NOT key`, and `x = 0.0` on a float key as `x IN (-0.0, 0.0)`. These
//!   conjuncts bind a leading prefix of keys: key 0, then key 1, and so on, up
//!   to the first key that no conjunct fixes or whose smallest set would grow
//!   the prefix past [`MAX_PREFIX_TUPLES`]. When several conjuncts fix one
//!   key, the one with the smallest set binds it.
//! - `key < v`, `key <= v`, `key > v`, `key >= v`, `key BETWEEN v AND w`, and
//!   `key IS NOT NULL` bound the key after the prefix. In input order, each one
//!   binds only when every finite bound it carries is still open, because a
//!   conjunct is consumed whole or not at all. So in an unoptimized
//!   `a < 20 AND a BETWEEN 5 AND 10`, the `BETWEEN` stays in the key residual.
//!   [`Planner::optimize_expr`] splits `BETWEEN` into two bounds, so in
//!   `a > 0 AND a BETWEEN 1 AND 10`, `a > 0` binds the lower bound, `a <= 10`
//!   binds the upper bound, and `a >= 1` stays in the key residual.
//! - Every other conjunct that reads only key columns joins the key residual,
//!   which the index evaluates on decoded keys. This includes any other OR or
//!   NOT, `!=`, `NOT IN`, expressions over keys, conjuncts on keys after the
//!   range, and a bound on a side that an earlier conjunct already bounds.
//! - A conjunct that reads any other column goes back to the caller untouched,
//!   in the row residual. A column qualified by a relation, such as `l.tenant`,
//!   counts as outside the key, because the keys carry no relation to match it.
//!
//! Literals, casts of literals, and `arrow_cast` of literals are coerced to the
//! key type when the coercion is lossless. In a returned plan, any other literal
//! leaves its conjunct in the key residual.
//!
//! NULL literals follow SQL, where a comparison with NULL is never TRUE.
//! `key = NULL`, alone or as an OR term, and the NULL items of `key IN (...)`
//! match no row, so the planner drops them from the key's set. In a filter,
//! `x OR NULL` selects the rows that `x` selects, so the planner drops a NULL
//! literal OR term too. `key = NULL` and `key IN (NULL)` leave an empty set,
//! which selects no rows. In a returned plan, a conjunct with a NULL range
//! bound stays in the key residual. Only `key IS NULL` matches NULL keys.
//!
//! [`Planner::optimize_expr`]: lance_datafusion::planner::Planner::optimize_expr

use std::collections::HashSet;
use std::ops::Bound;

use arrow_schema::{DataType, FieldRef, Fields};
use datafusion_common::ScalarValue;
use datafusion_expr::expr::{Between, BinaryExpr, Cast, InList, ScalarFunction};
use datafusion_expr::utils::split_binary;
use datafusion_expr::{Expr, Operator};
use lance_datafusion::expr::safe_coerce_scalar;

use super::{KeyRange, KeyValues, TupleQuery};

/// The most prefix tuples one planned [`TupleQuery`] may cover.
///
/// The prefix covers the cross product of its keys' sets, and an index runs one
/// seek per prefix tuple, so this caps the seeks one query costs. A key whose
/// set would grow the product past the cap ends the prefix. In a returned plan,
/// its conjunct stays in the key residual.
pub const MAX_PREFIX_TUPLES: usize = 8192;

/// A conjunctive filter, split by what a [`TupleQuery`] answers.
///
/// Each input conjunct is in exactly one of `consumed`, `key_residual`, and
/// `row_residual`, in input order.
#[derive(Clone, Debug, PartialEq)]
pub struct TuplePlan<'a> {
    /// The key tuples to search.
    pub query: TupleQuery,
    /// Conjuncts that select exactly the rows `query` selects.
    pub consumed: Vec<&'a Expr>,
    /// Conjuncts that read only key columns, constants included, but that
    /// `query` does not express.
    pub key_residual: Vec<&'a Expr>,
    /// Conjuncts that read a column outside the key, including any column
    /// qualified by a relation.
    pub row_residual: Vec<&'a Expr>,
}

enum Constraint {
    Values { key: usize, values: KeyValues },
    Range { key: usize, range: KeyRange },
}

/// Plan `conjuncts`, the terms of an AND, as one query over `keys`, the index's
/// key columns in key order.
///
/// Pass conjuncts that went through [`Planner::optimize_expr`], as
/// [`create_filter_plan`] does before it plans per-column indices. That pass
/// coerces literal types, so `at >= '2024-01-01'` on a timestamp key arrives
/// here with a timestamp literal. This function never parses a string into a
/// timestamp, so without that pass the raw string leaves its conjunct unplanned.
///
/// Returns `None` when no conjunct binds the first key, and the caller keeps
/// every conjunct. When the smallest set on the first key holds more than
/// [`MAX_PREFIX_TUPLES`] values, no set binds it, so the result is `None`
/// unless a range binds that key.
///
/// [`Planner::optimize_expr`]: lance_datafusion::planner::Planner::optimize_expr
/// [`create_filter_plan`]: crate::scalar::expression::PlannerIndexExt::create_filter_plan
///
/// ```
/// use arrow_schema::{DataType, Field, Fields};
/// use datafusion_expr::{col, lit};
/// use lance_index::scalar::tuple::planner::plan_tuple_predicate;
///
/// let keys = Fields::from(vec![
///     Field::new("tenant", DataType::Int32, true),
///     Field::new("ts", DataType::Int64, true),
/// ]);
/// let conjuncts = [
///     col("tenant").in_list(vec![lit(1), lit(2)], false),
///     col("ts").gt_eq(lit(10_i64)),
///     col("ts").lt(lit(20_i64)),
///     col("status").eq(lit("a")),
/// ];
/// let plan = plan_tuple_predicate(&conjuncts, &keys).expect("tenant binds the first key");
/// assert_eq!(
///     plan.query.to_string(),
///     "TupleQuery[tenant IN (1, 2), ts >= 10 AND ts < 20]"
/// );
/// assert_eq!(plan.row_residual, vec![&conjuncts[3]]);
/// ```
pub fn plan_tuple_predicate<'a>(conjuncts: &'a [Expr], keys: &Fields) -> Option<TuplePlan<'a>> {
    let mut row_residual = Vec::new();
    let mut key_conjuncts = Vec::new();
    for conjunct in conjuncts {
        let reads_only_keys = conjunct
            .column_refs()
            .iter()
            .all(|column| column.relation.is_none() && keys.find(&column.name).is_some());
        if reads_only_keys {
            key_conjuncts.push((conjunct, constraint(conjunct, keys)));
        } else {
            row_residual.push(conjunct);
        }
    }
    let mut is_consumed = vec![false; key_conjuncts.len()];

    let mut prefix = Vec::new();
    let mut tuples = 1_usize;
    for index in 0..keys.len() {
        let smallest = key_conjuncts
            .iter()
            .enumerate()
            .filter_map(|(position, (_, constraint))| match constraint {
                Some(Constraint::Values { key, values }) if *key == index => {
                    Some((position, values))
                }
                _ => None,
            })
            .min_by_key(|(_, values)| set_size(values));
        let Some((position, values)) = smallest else {
            break;
        };
        let Some(product) = tuples
            .checked_mul(set_size(values))
            .filter(|product| *product <= MAX_PREFIX_TUPLES)
        else {
            break;
        };
        tuples = product;
        is_consumed[position] = true;
        prefix.push(values.clone());
    }

    let mut range: Option<KeyRange> = None;
    for (position, (_, constraint)) in key_conjuncts.iter().enumerate() {
        let Some(Constraint::Range {
            key,
            range: (lower, upper),
        }) = constraint
        else {
            continue;
        };
        if *key != prefix.len() {
            continue;
        }
        let bound = range.get_or_insert((Bound::Unbounded, Bound::Unbounded));
        let lower_fits = matches!(lower, Bound::Unbounded) || matches!(bound.0, Bound::Unbounded);
        let upper_fits = matches!(upper, Bound::Unbounded) || matches!(bound.1, Bound::Unbounded);
        if lower_fits && upper_fits {
            if matches!(bound.0, Bound::Unbounded) {
                bound.0 = lower.clone();
            }
            if matches!(bound.1, Bound::Unbounded) {
                bound.1 = upper.clone();
            }
            is_consumed[position] = true;
        }
    }

    if prefix.is_empty() && range.is_none() {
        return None;
    }
    let mut consumed = Vec::new();
    let mut key_residual = Vec::new();
    for ((conjunct, _), is_consumed) in key_conjuncts.into_iter().zip(is_consumed) {
        if is_consumed {
            consumed.push(conjunct);
        } else {
            key_residual.push(conjunct);
        }
    }
    Some(TuplePlan {
        query: TupleQuery {
            keys: keys.clone(),
            prefix,
            range,
        },
        consumed,
        key_residual,
        row_residual,
    })
}

fn set_size(values: &KeyValues) -> usize {
    match values {
        KeyValues::Null => 1,
        KeyValues::In(values) => values.len(),
    }
}

fn constraint(conjunct: &Expr, keys: &Fields) -> Option<Constraint> {
    match conjunct {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: op @ (Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq),
            right,
        }) => {
            let ((key, field), op, literal) = match key_column(left, keys) {
                Some(column) => (column, *op, right),
                None => (key_column(right, keys)?, op.swap()?, left),
            };
            let value = bound(literal, field.data_type())?;
            let range = match op {
                Operator::Lt => (Bound::Unbounded, Bound::Excluded(value)),
                Operator::LtEq => (Bound::Unbounded, Bound::Included(value)),
                Operator::Gt => (Bound::Excluded(value), Bound::Unbounded),
                Operator::GtEq => (Bound::Included(value), Bound::Unbounded),
                _ => return None,
            };
            Some(Constraint::Range { key, range })
        }
        Expr::Between(Between {
            expr,
            negated: false,
            low,
            high,
        }) => {
            let (key, field) = key_column(expr, keys)?;
            let low = bound(low, field.data_type())?;
            let high = bound(high, field.data_type())?;
            let range = (Bound::Included(low), Bound::Included(high));
            Some(Constraint::Range { key, range })
        }
        Expr::IsNull(expr) => Some(Constraint::Values {
            key: key_column(expr, keys)?.0,
            values: KeyValues::Null,
        }),
        Expr::IsNotNull(expr) => Some(Constraint::Range {
            key: key_column(expr, keys)?.0,
            range: (Bound::Unbounded, Bound::Unbounded),
        }),
        Expr::BinaryExpr(BinaryExpr {
            op: Operator::Eq | Operator::Or,
            ..
        })
        | Expr::InList(_)
        | Expr::Column(_)
        | Expr::Not(_) => key_set(conjunct, keys),
        _ => None,
    }
}

fn key_column<'k>(expr: &Expr, keys: &'k Fields) -> Option<(usize, &'k FieldRef)> {
    match expr {
        Expr::Column(column) => keys.find(&column.name),
        _ => None,
    }
}

fn key_set(conjunct: &Expr, keys: &Fields) -> Option<Constraint> {
    let boolean_key = |expr: &Expr| {
        key_column(expr, keys)
            .filter(|(_, field)| field.data_type() == &DataType::Boolean)
            .map(|(key, _)| key)
    };
    let mut key = None;
    let mut values = Vec::new();
    for term in split_binary(conjunct, Operator::Or) {
        let term_key = match term {
            Expr::Literal(value, _) if value.is_null() => continue,
            Expr::BinaryExpr(BinaryExpr {
                left,
                op: Operator::Eq,
                right,
            }) => {
                let (term_key, value) = match key_column(left, keys) {
                    Some((term_key, _)) => (term_key, right),
                    None => (key_column(right, keys)?.0, left),
                };
                values.push(literal(value)?);
                term_key
            }
            Expr::InList(InList {
                expr,
                list,
                negated: false,
            }) => {
                for item in list {
                    values.push(literal(item)?);
                }
                key_column(expr, keys)?.0
            }
            Expr::Column(_) => {
                values.push(ScalarValue::Boolean(Some(true)));
                boolean_key(term)?
            }
            Expr::Not(expr) => {
                values.push(ScalarValue::Boolean(Some(false)));
                boolean_key(expr)?
            }
            _ => return None,
        };
        if *key.get_or_insert(term_key) != term_key {
            return None;
        }
    }
    let key = key?;
    let values = key_values(values, keys[key].data_type())?;
    Some(Constraint::Values { key, values })
}

fn key_values(
    items: impl IntoIterator<Item = ScalarValue>,
    data_type: &DataType,
) -> Option<KeyValues> {
    let mut values = Vec::new();
    let mut seen = HashSet::new();
    for value in items {
        if value.is_null() {
            continue;
        }
        let value = coerce_lossless(value, data_type)?;
        if seen.insert(value.clone()) {
            values.push(value);
        }
    }
    Some(KeyValues::In(values))
}

fn bound(expr: &Expr, data_type: &DataType) -> Option<ScalarValue> {
    literal(expr)
        .filter(|value| !value.is_null())
        .and_then(|value| coerce_lossless(value, data_type))
}

fn literal(expr: &Expr) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) => Some(value.clone()),
        Expr::Cast(Cast { expr, field }) => match expr.as_ref() {
            Expr::Literal(value, _) => value.cast_to(field.data_type()).ok(),
            _ => None,
        },
        Expr::ScalarFunction(ScalarFunction { func, args }) if func.name() == "arrow_cast" => {
            match args.as_slice() {
                [Expr::Literal(value, _), Expr::Literal(data_type, _)] => {
                    let data_type = data_type.try_as_str()??.parse::<DataType>().ok()?;
                    value.cast_to(&data_type).ok()
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn coerce_lossless(value: ScalarValue, data_type: &DataType) -> Option<ScalarValue> {
    let coerced = safe_coerce_scalar(&value, data_type)?;
    let is_lossless =
        coerced.data_type() == *data_type && coerced.cast_to(&value.data_type()).ok()? == value;
    is_lossless.then_some(coerced)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, LazyLock};

    use arrow_array::cast::AsArray;
    use arrow_array::{
        BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray, TimestampMillisecondArray,
    };
    use arrow_schema::{Field, Schema, TimeUnit};
    use datafusion::prelude::SessionContext;
    use datafusion_common::{Column, DFSchema};
    use datafusion_expr::utils::split_conjunction;
    use datafusion_expr::{col, lit};
    use itertools::{Itertools, iproduct};
    use lance_datafusion::planner::Planner;
    use rstest::rstest;

    use super::*;
    use crate::scalar::AnyQuery;

    const NEW_YEAR_2024_MS: i64 = 1_704_067_200_000;

    struct Fixture {
        keys: Fields,
        boolean_keys: Fields,
        schema: DFSchema,
        rows: RecordBatch,
        session: SessionContext,
    }

    static FIXTURE: LazyLock<Fixture> = LazyLock::new(|| {
        let schema = Arc::new(Schema::new(vec![
            Field::new("tenant", DataType::Int32, true),
            Field::new("ts", DataType::Int64, true),
            Field::new("at", DataType::Timestamp(TimeUnit::Millisecond, None), true),
            Field::new("status", DataType::Utf8, true),
            Field::new("flag", DataType::Boolean, true),
        ]));
        let tenants = [None, Some(0), Some(1), Some(2), Some(3), Some(7)]
            .into_iter()
            .chain([63, 64, 8190, 8191, 8192].map(Some));
        let ts = [None, Some(0), Some(1), Some(5), Some(9), Some(10), Some(11)]
            .into_iter()
            .chain([19, 20, 21, 127, 128, 2730, 2731].map(Some));
        let at = [
            None,
            Some(NEW_YEAR_2024_MS),
            Some(NEW_YEAR_2024_MS + 1),
            Some(NEW_YEAR_2024_MS + 86_400_000),
        ];
        let status = [None, Some("a"), Some("b")];
        let flag = [None, Some(false), Some(true)];
        let (tenants, ts, at, status, flag): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            iproduct!(tenants, ts, at, status, flag).multiunzip();
        let rows = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(tenants)),
                Arc::new(Int64Array::from(ts)),
                Arc::new(TimestampMillisecondArray::from(at)),
                Arc::new(StringArray::from(status)),
                Arc::new(BooleanArray::from(flag)),
            ],
        )
        .unwrap();
        Fixture {
            keys: schema.fields()[..3].into(),
            boolean_keys: vec![schema.field(4).clone(), schema.field(0).clone()].into(),
            schema: DFSchema::try_from(schema.as_ref().clone()).unwrap(),
            rows,
            session: SessionContext::new(),
        }
    });

    fn selected_rows(predicate: Expr) -> Vec<bool> {
        let fixture = &*FIXTURE;
        let predicate = Planner::new(fixture.rows.schema())
            .optimize_expr(predicate)
            .unwrap();
        let predicate = fixture
            .session
            .create_physical_expr(predicate, &fixture.schema)
            .unwrap();
        let selected = predicate
            .evaluate(&fixture.rows)
            .unwrap()
            .into_array(fixture.rows.num_rows())
            .unwrap();
        selected
            .as_boolean()
            .iter()
            .map(|is_selected| is_selected == Some(true))
            .collect()
    }

    struct Expected {
        query: String,
        key_residual: &'static [usize],
        row_residual: &'static [usize],
    }

    fn planned(
        query: impl Into<String>,
        key_residual: &'static [usize],
        row_residual: &'static [usize],
    ) -> Option<Expected> {
        Some(Expected {
            query: query.into(),
            key_residual,
            row_residual,
        })
    }

    fn check_plan(conjuncts: &[Expr], keys: &Fields, expected: Option<Expected>) {
        let plan = plan_tuple_predicate(conjuncts, keys);
        let Some(expected) = expected else {
            assert_eq!(plan, None);
            return;
        };
        let plan = plan.expect("predicate should plan");

        let query = &plan.query;
        let revalidated = TupleQuery::try_new(
            query.keys().clone(),
            query.prefix().to_vec(),
            query.range().cloned(),
        )
        .unwrap();
        assert_eq!(&revalidated, query);

        let conjuncts_at =
            |positions: &[usize]| positions.iter().map(|&p| &conjuncts[p]).collect::<Vec<_>>();
        let consumed = (0..conjuncts.len())
            .filter(|p| !expected.key_residual.contains(p) && !expected.row_residual.contains(p))
            .collect::<Vec<_>>();
        assert_eq!(query.to_string(), expected.query);
        assert_eq!(plan.consumed, conjuncts_at(&consumed));
        assert_eq!(plan.key_residual, conjuncts_at(expected.key_residual));
        assert_eq!(plan.row_residual, conjuncts_at(expected.row_residual));

        let planned = std::iter::once(query.to_expr(String::new()))
            .chain(plan.key_residual.into_iter().cloned())
            .chain(plan.row_residual.into_iter().cloned())
            .reduce(Expr::and)
            .unwrap();
        let predicate = conjuncts.iter().cloned().reduce(Expr::and).unwrap();
        assert_eq!(selected_rows(planned), selected_rows(predicate));
    }

    fn in_list(key: &str, len: usize) -> String {
        format!("{key} IN ({})", (0..len).join(", "))
    }

    // A SQL filter reaches the planner after two optimizer passes, one when the
    // scanner parses it and one in `create_filter_plan`.
    fn conjuncts(predicate: &str, passes: usize) -> Vec<Expr> {
        let mut predicate = FIXTURE
            .session
            .state()
            .create_logical_expr(predicate, &FIXTURE.schema)
            .unwrap();
        for _ in 0..passes {
            predicate = Planner::new(FIXTURE.rows.schema())
                .optimize_expr(predicate)
                .unwrap();
        }
        split_conjunction(&predicate).into_iter().cloned().collect()
    }

    #[rstest]
    #[case::you_see(
        "tenant IN (1, 2) AND ts >= 10 AND ts < 20 AND status = 'a'",
        planned("TupleQuery[tenant IN (1, 2), ts >= 10 AND ts < 20]", &[], &[3])
    )]
    #[case::equality_on_one_key("tenant = 1", planned("TupleQuery[tenant = 1]", &[], &[]))]
    #[case::equality_on_two_keys(
        "tenant = 1 AND ts = 10",
        planned("TupleQuery[tenant = 1, ts = 10]", &[], &[])
    )]
    #[case::equality_on_three_keys(
        "tenant = 1 AND ts = 10 AND at = TIMESTAMP '2024-01-01 00:00:00'",
        planned("TupleQuery[tenant = 1, ts = 10, at = 1704067200000]", &[], &[])
    )]
    #[case::equality_out_of_key_order(
        "ts = 10 AND tenant = 1",
        planned("TupleQuery[tenant = 1, ts = 10]", &[], &[])
    )]
    #[case::in_on_two_keys(
        "tenant IN (1, 2) AND ts IN (10, 20)",
        planned("TupleQuery[tenant IN (1, 2), ts IN (10, 20)]", &[], &[])
    )]
    #[case::in_three_values(
        "tenant IN (1, 2, 3) AND ts = 10",
        planned("TupleQuery[tenant IN (1, 2, 3), ts = 10]", &[], &[])
    )]
    #[case::in_four_values(
        "tenant IN (1, 2, 3, 4)",
        planned("TupleQuery[tenant IN (1, 2, 3, 4)]", &[], &[])
    )]
    #[case::range_gt(
        "tenant = 1 AND ts > 10",
        planned("TupleQuery[tenant = 1, ts > 10]", &[], &[])
    )]
    #[case::range_gt_eq("tenant >= 2", planned("TupleQuery[tenant >= 2]", &[], &[]))]
    #[case::range_lt(
        "tenant = 1 AND ts < 10",
        planned("TupleQuery[tenant = 1, ts < 10]", &[], &[])
    )]
    #[case::range_lt_eq(
        "tenant = 1 AND ts <= 10",
        planned("TupleQuery[tenant = 1, ts <= 10]", &[], &[])
    )]
    #[case::range_on_first_key(
        "tenant > 1 AND tenant < 7",
        planned("TupleQuery[tenant > 1 AND tenant < 7]", &[], &[])
    )]
    #[case::second_lower_bound(
        "tenant = 1 AND ts > 5 AND ts >= 10",
        planned("TupleQuery[tenant = 1, ts > 5]", &[2], &[])
    )]
    #[case::between_after_lower_bound(
        "tenant > 10 AND tenant BETWEEN 0 AND 100",
        planned("TupleQuery[tenant > 10 AND tenant <= 100]", &[1], &[])
    )]
    #[case::is_null_in_prefix(
        "tenant IS NULL AND ts = 10",
        planned("TupleQuery[tenant IS NULL, ts = 10]", &[], &[])
    )]
    #[case::is_null_on_last_bound_key(
        "tenant = 1 AND ts IS NULL",
        planned("TupleQuery[tenant = 1, ts IS NULL]", &[], &[])
    )]
    #[case::is_not_null(
        "tenant = 1 AND ts IS NOT NULL",
        planned("TupleQuery[tenant = 1, ts IS NOT NULL]", &[], &[])
    )]
    #[case::is_not_null_with_bound(
        "tenant = 1 AND ts IS NOT NULL AND ts < 10",
        planned("TupleQuery[tenant = 1, ts < 10]", &[], &[])
    )]
    #[case::in_with_null("tenant IN (1, NULL)", planned("TupleQuery[tenant = 1]", &[], &[]))]
    #[case::duplicate_in_values(
        "tenant IN (2, 1, 2)",
        planned("TupleQuery[tenant IN (2, 1)]", &[], &[])
    )]
    #[case::smallest_set_binds(
        "tenant IN (1, 2, 3) AND tenant = 2",
        planned("TupleQuery[tenant = 2]", &[0], &[])
    )]
    #[case::contradiction(
        "tenant = 1 AND tenant = 2",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::contradiction_with_is_null(
        "tenant = 1 AND tenant IS NULL",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::expression_over_key(
        "tenant = 1 AND ts % 7 = 0",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::range_on_bound_key(
        "tenant = 1 AND tenant > 0",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::equality_after_range(
        "tenant > 1 AND ts = 10",
        planned("TupleQuery[tenant > 1]", &[1], &[])
    )]
    #[case::skipped_key(
        "tenant = 1 AND at = TIMESTAMP '2024-01-01 00:00:00'",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::or_on_bound_prefix(
        "tenant = 1 AND (ts = 10 OR ts = 20)",
        planned("TupleQuery[tenant = 1, ts IN (10, 20)]", &[], &[])
    )]
    #[case::or_mixed_with_range(
        "tenant = 1 AND (ts = 10 OR ts > 20)",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::or_mixed_with_is_null(
        "tenant = 1 AND (ts = 10 OR ts IS NULL)",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::not_on_bound_prefix(
        "tenant = 1 AND NOT ts = 10",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::non_key_column(
        "tenant = 1 AND status = 'a'",
        planned("TupleQuery[tenant = 1]", &[], &[1])
    )]
    #[case::non_key_column_inside_or(
        "tenant = 1 AND (ts = 10 OR status = 'a')",
        planned("TupleQuery[tenant = 1]", &[], &[1])
    )]
    #[case::timestamp_literal_coerced(
        "tenant = 1 AND ts = 10 AND at >= TIMESTAMP '2024-01-01 00:00:00'",
        planned("TupleQuery[tenant = 1, ts = 10, at >= 1704067200000]", &[], &[])
    )]
    #[case::timestamp_string_literal(
        "tenant = 1 AND ts = 10 AND at >= '2024-01-01'",
        planned("TupleQuery[tenant = 1, ts = 10, at >= 1704067200000]", &[], &[])
    )]
    #[case::literal_overflows_key("tenant = 3000000000", None)]
    #[case::float_literal_on_integer_key("tenant = 1.5", None)]
    #[case::or_across_keys("tenant = 1 OR ts = 2", None)]
    #[case::or_on_one_key(
        "tenant = 1 OR tenant = 2",
        planned("TupleQuery[tenant IN (1, 2)]", &[], &[])
    )]
    #[case::not_equality("NOT tenant = 1", None)]
    #[case::not_equal("tenant != 1", None)]
    #[case::not_in("tenant NOT IN (1, 2)", None)]
    #[case::not_between("tenant NOT BETWEEN 1 AND 2", None)]
    #[case::equality_on_second_key_only("ts = 10", None)]
    #[case::equality_on_later_keys_only("ts = 10 AND at = TIMESTAMP '2024-01-01 00:00:00'", None)]
    #[case::arrow_cast_of_key("tenant = arrow_cast(ts, 'Int32')", None)]
    #[case::key_compared_to_key("tenant = ts", None)]
    #[case::non_key_only("status = 'a'", None)]
    #[case::cap_8191_tuples(
        in_list("tenant", 8191),
        planned(format!("TupleQuery[{}]", in_list("tenant", 8191)), &[], &[])
    )]
    #[case::cap_8192_tuples(
        format!("{} AND {}", in_list("tenant", 64), in_list("ts", 128)),
        planned(
            format!("TupleQuery[{}, {}]", in_list("tenant", 64), in_list("ts", 128)),
            &[],
            &[]
        )
    )]
    #[case::cap_8193_tuples(
        format!("{} AND {}", in_list("tenant", 3), in_list("ts", 2731)),
        planned(format!("TupleQuery[{}]", in_list("tenant", 3)), &[1], &[])
    )]
    #[case::cap_8193_values_on_one_key(in_list("tenant", 8193), None)]
    #[case::cap_8193_values_beside_range(
        format!("{} AND tenant > 5", in_list("tenant", 8193)),
        planned("TupleQuery[tenant > 5]", &[0], &[])
    )]
    fn plans_conjunctive_predicate(
        #[case] predicate: String,
        #[case] expected: Option<Expected>,
        #[values(1, 2)] passes: usize,
    ) {
        check_plan(&conjuncts(&predicate, passes), &FIXTURE.keys, expected);
    }

    #[rstest]
    #[case::equals_true(
        "flag = true AND tenant = 1",
        planned("TupleQuery[flag = true, tenant = 1]", &[], &[])
    )]
    #[case::equals_false(
        "flag = false AND tenant = 1",
        planned("TupleQuery[flag = false, tenant = 1]", &[], &[])
    )]
    #[case::or_of_both_values(
        "(flag = true OR flag = false) AND tenant = 1",
        planned("TupleQuery[flag IN (true, false), tenant = 1]", &[], &[])
    )]
    #[case::or_across_keys("flag OR tenant = 1", None)]
    fn plans_predicate_on_boolean_key(
        #[case] predicate: &str,
        #[case] expected: Option<Expected>,
        #[values(1, 2)] passes: usize,
    ) {
        check_plan(
            &conjuncts(predicate, passes),
            &FIXTURE.boolean_keys,
            expected,
        );
    }

    #[rstest]
    #[case::literal_on_left("1 = tenant", planned("TupleQuery[tenant = 1]", &[], &[]))]
    #[case::arrow_cast_on_first_key(
        "tenant = arrow_cast(1, 'Int32')",
        planned("TupleQuery[tenant = 1]", &[], &[])
    )]
    #[case::range_literal_on_left(
        "tenant = 1 AND 10 < ts",
        planned("TupleQuery[tenant = 1, ts > 10]", &[], &[])
    )]
    #[case::range_between(
        "tenant = 1 AND ts BETWEEN 5 AND 10",
        planned("TupleQuery[tenant = 1, ts >= 5 AND ts <= 10]", &[], &[])
    )]
    #[case::between_after_upper_bound(
        "tenant = 1 AND ts < 20 AND ts BETWEEN 5 AND 10",
        planned("TupleQuery[tenant = 1, ts < 20]", &[2], &[])
    )]
    #[case::equals_null("tenant = NULL", planned("TupleQuery[FALSE]", &[], &[]))]
    #[case::null_range_bound(
        "tenant = 1 AND ts > NULL",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::constant_conjunct(
        "tenant = 1 AND FALSE",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::timestamp_literal_lossy(
        "tenant = 1 AND ts = 10 AND at = TIMESTAMP '2024-01-01 00:00:00.000000100'",
        planned("TupleQuery[tenant = 1, ts = 10]", &[2], &[])
    )]
    #[case::timestamp_string_literal(
        "tenant = 1 AND ts = 10 AND at >= '2024-01-01'",
        planned("TupleQuery[tenant = 1, ts = 10]", &[2], &[])
    )]
    #[case::arrow_cast_string_to_timestamp(
        "tenant = 1 AND ts = 10 AND at >= arrow_cast('2024-01-01T00:00:00', 'Timestamp(Millisecond, None)')",
        planned("TupleQuery[tenant = 1, ts = 10, at >= 1704067200000]", &[], &[])
    )]
    #[case::cast_key("CAST(tenant AS BIGINT) = 1", None)]
    #[case::in_only_null("tenant IN (NULL)", planned("TupleQuery[FALSE]", &[], &[]))]
    #[case::not_equality("NOT tenant = 1", None)]
    #[case::not_between("tenant NOT BETWEEN 1 AND 2", None)]
    #[case::or_with_null_literal(
        "tenant = 1 OR NULL",
        planned("TupleQuery[tenant = 1]", &[], &[])
    )]
    #[case::or_with_true("tenant = 1 OR TRUE", None)]
    #[case::or_with_lossy_value("tenant = 1 OR tenant = 1.5", None)]
    #[case::or_with_in_term(
        "tenant IN (1, 2, 3, 4) OR tenant = 5",
        planned("TupleQuery[tenant IN (1, 2, 3, 4, 5)]", &[], &[])
    )]
    #[case::or_with_not_in_term("tenant NOT IN (1, 2, 3, 4) OR tenant = 5", None)]
    fn plans_unoptimized_predicate(#[case] predicate: &str, #[case] expected: Option<Expected>) {
        check_plan(&conjuncts(predicate, 0), &FIXTURE.keys, expected);
    }

    #[rstest]
    fn signed_zero_in_list_binds_float_key(#[values(1, 2)] passes: usize) {
        let keys = Fields::from(vec![Field::new("x", DataType::Float64, true)]);
        let planner = Planner::new(Arc::new(Schema::new(keys.clone())));
        let mut filter = planner.parse_filter("x IN (0.0, 1.0)").unwrap();
        for _ in 0..passes {
            filter = planner.optimize_expr(filter).unwrap();
        }
        let plan = plan_tuple_predicate(std::slice::from_ref(&filter), &keys).expect("x binds");
        assert_eq!(plan.query.to_string(), "TupleQuery[x IN (-0, 0, 1)]");
    }

    #[test]
    fn qualified_column_never_binds_a_key() {
        let other_tenant = || Expr::Column(Column::new(Some("l"), "tenant"));
        check_plan(&[other_tenant().eq(lit(1))], &FIXTURE.keys, None);

        let conjuncts = [col("tenant").eq(lit(1)), other_tenant().eq(lit(2))];
        let plan = plan_tuple_predicate(&conjuncts, &FIXTURE.keys).expect("tenant binds key 0");
        assert_eq!(plan.consumed, vec![&conjuncts[0]]);
        assert_eq!(plan.row_residual, vec![&conjuncts[1]]);
    }

    #[test]
    fn literal_in_other_time_zone_stays_in_key_residual() {
        let new_year_utc =
            ScalarValue::TimestampMillisecond(Some(NEW_YEAR_2024_MS), Some("UTC".into()));
        let conjuncts = [
            col("tenant").eq(lit(1)),
            col("ts").eq(lit(10_i64)),
            col("at").eq(lit(new_year_utc)),
        ];
        check_plan(
            &conjuncts,
            &FIXTURE.keys,
            planned("TupleQuery[tenant = 1, ts = 10]", &[2], &[]),
        );
    }
}
