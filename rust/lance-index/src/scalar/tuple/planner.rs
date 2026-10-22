// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plan a conjunctive filter as a [`TupleQuery`] over ordered key columns.
//!
//! [`plan_tuple_predicate`] reads each conjunct on its own.
//!
//! - `key = v`, `key IN (...)`, and `key IS NULL` fix a key to a set of values.
//!   They bind a leading prefix of keys: key 0, then key 1, and so on, up to the
//!   first key that no conjunct fixes. When several conjuncts fix one key, the
//!   one with the smallest set binds it.
//! - `key < v`, `key <= v`, `key > v`, `key >= v`, `key BETWEEN v AND w`, and
//!   `key IS NOT NULL` bound the key after the prefix. The first conjunct with a
//!   lower bound and the first with an upper bound bind it.
//! - Every other conjunct that reads only key columns joins the key residual,
//!   which the index evaluates on decoded keys. This includes OR, NOT, `!=`,
//!   `NOT IN`, expressions over keys, conjuncts on keys after the range, and a
//!   second bound on a key that already has one.
//! - A conjunct that reads any other column goes back to the caller untouched.
//!
//! Literals, and casts of literals, are coerced to the key type when the
//! coercion is lossless. Any other literal leaves its conjunct in the key
//! residual.
//!
//! NULL literals follow SQL, where a comparison with NULL is never TRUE.
//! `key = NULL` and the NULL items of `key IN (...)` match no row, so the
//! planner drops them from the key's set. `key = NULL` and `key IN (NULL)`
//! leave an empty set, which selects no rows. A conjunct with a NULL range
//! bound stays in the key residual. Only `key IS NULL` matches NULL keys.

use std::collections::HashSet;
use std::ops::Bound;

use arrow_schema::{DataType, Fields};
use datafusion_common::ScalarValue;
use datafusion_expr::expr::{Between, BinaryExpr, Cast, InList};
use datafusion_expr::{Expr, Operator};
use lance_datafusion::expr::safe_coerce_scalar;

use super::{KeyRange, KeyValues, TupleQuery};

/// The most prefix tuples one planned [`TupleQuery`] may cover.
///
/// The prefix covers the cross product of its keys' sets, and an index runs one
/// seek per prefix tuple, so this caps the seeks one query costs. A key whose
/// set would grow the product past the cap ends the prefix, and its conjunct
/// stays in the key residual.
pub const MAX_PREFIX_TUPLES: usize = 8192;

/// A conjunctive filter, split by what a [`TupleQuery`] answers.
///
/// Each input conjunct is in exactly one of `consumed`, `key_residual`, and
/// `row_residual`, in input order.
#[derive(Clone, Debug, PartialEq)]
pub struct TuplePlan<'a> {
    /// The key tuples to search.
    pub query: TupleQuery,
    /// Conjuncts that `query` expresses exactly.
    pub consumed: Vec<&'a Expr>,
    /// Conjuncts that read only key columns, constants included, but that
    /// `query` does not express.
    pub key_residual: Vec<&'a Expr>,
    /// Conjuncts that read a column outside the key.
    pub row_residual: Vec<&'a Expr>,
}

enum Constraint {
    Values { key: usize, values: KeyValues },
    Range { key: usize, range: KeyRange },
}

/// Plan `conjuncts`, the terms of an AND, as one query over `keys`, the index's
/// key columns in key order.
///
/// Returns `None` when no conjunct binds the first key.
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
            .all(|column| keys.find(&column.name).is_some());
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
    let key_column = |expr: &Expr| match expr {
        Expr::Column(column) => keys.find(&column.name),
        _ => None,
    };
    match conjunct {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            let ((key, field), op, literal) = match key_column(left) {
                Some(column) => (column, *op, right),
                None => (key_column(right)?, op.swap()?, left),
            };
            if op == Operator::Eq {
                let values = key_values([literal.as_ref()], field.data_type())?;
                return Some(Constraint::Values { key, values });
            }
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
        Expr::InList(InList {
            expr,
            list,
            negated: false,
        }) => {
            let (key, field) = key_column(expr)?;
            let values = key_values(list, field.data_type())?;
            Some(Constraint::Values { key, values })
        }
        Expr::Between(Between {
            expr,
            negated: false,
            low,
            high,
        }) => {
            let (key, field) = key_column(expr)?;
            let low = bound(low, field.data_type())?;
            let high = bound(high, field.data_type())?;
            let range = (Bound::Included(low), Bound::Included(high));
            Some(Constraint::Range { key, range })
        }
        Expr::IsNull(expr) => Some(Constraint::Values {
            key: key_column(expr)?.0,
            values: KeyValues::Null,
        }),
        Expr::IsNotNull(expr) => Some(Constraint::Range {
            key: key_column(expr)?.0,
            range: (Bound::Unbounded, Bound::Unbounded),
        }),
        _ => None,
    }
}

fn key_values<'e>(
    items: impl IntoIterator<Item = &'e Expr>,
    data_type: &DataType,
) -> Option<KeyValues> {
    let mut values = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        let value = literal(item)?;
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
        Int32Array, Int64Array, RecordBatch, StringArray, TimestampMillisecondArray,
    };
    use arrow_schema::{Field, Schema, TimeUnit};
    use datafusion::prelude::SessionContext;
    use datafusion_common::DFSchema;
    use datafusion_expr::utils::split_conjunction;
    use datafusion_expr::{col, lit};
    use itertools::{Itertools, iproduct};
    use rstest::rstest;

    use super::*;
    use crate::scalar::AnyQuery;

    const NEW_YEAR_2024_MS: i64 = 1_704_067_200_000;

    struct Fixture {
        keys: Fields,
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
        let (tenants, ts, at, status): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            iproduct!(tenants, ts, at, status).multiunzip();
        let rows = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(tenants)),
                Arc::new(Int64Array::from(ts)),
                Arc::new(TimestampMillisecondArray::from(at)),
                Arc::new(StringArray::from(status)),
            ],
        )
        .unwrap();
        Fixture {
            keys: schema.fields()[..3].into(),
            schema: DFSchema::try_from(schema.as_ref().clone()).unwrap(),
            rows,
            session: SessionContext::new(),
        }
    });

    fn selected_rows(predicate: Expr) -> Vec<bool> {
        let fixture = &*FIXTURE;
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

    fn check_plan(conjuncts: &[Expr], expected: Option<Expected>) {
        let plan = plan_tuple_predicate(conjuncts, &FIXTURE.keys);
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
    #[case::literal_on_left("1 = tenant", planned("TupleQuery[tenant = 1]", &[], &[]))]
    #[case::in_on_two_keys(
        "tenant IN (1, 2) AND ts IN (10, 20)",
        planned("TupleQuery[tenant IN (1, 2), ts IN (10, 20)]", &[], &[])
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
    #[case::range_literal_on_left(
        "tenant = 1 AND 10 < ts",
        planned("TupleQuery[tenant = 1, ts > 10]", &[], &[])
    )]
    #[case::range_between(
        "tenant = 1 AND ts BETWEEN 5 AND 10",
        planned("TupleQuery[tenant = 1, ts >= 5 AND ts <= 10]", &[], &[])
    )]
    #[case::range_on_first_key(
        "tenant > 1 AND tenant < 7",
        planned("TupleQuery[tenant > 1 AND tenant < 7]", &[], &[])
    )]
    #[case::second_lower_bound(
        "tenant = 1 AND ts > 5 AND ts >= 10",
        planned("TupleQuery[tenant = 1, ts > 5]", &[2], &[])
    )]
    #[case::between_after_upper_bound(
        "tenant = 1 AND ts < 20 AND ts BETWEEN 5 AND 10",
        planned("TupleQuery[tenant = 1, ts < 20]", &[2], &[])
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
    #[case::in_only_null("tenant IN (NULL)", planned("TupleQuery[tenant IN ()]", &[], &[]))]
    #[case::equals_null("tenant = NULL", planned("TupleQuery[tenant IN ()]", &[], &[]))]
    #[case::null_range_bound(
        "tenant = 1 AND ts > NULL",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
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
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::not_on_bound_prefix(
        "tenant = 1 AND NOT ts = 10",
        planned("TupleQuery[tenant = 1]", &[1], &[])
    )]
    #[case::constant_conjunct(
        "tenant = 1 AND FALSE",
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
    #[case::timestamp_literal_lossy(
        "tenant = 1 AND ts = 10 AND at = TIMESTAMP '2024-01-01 00:00:00.000000100'",
        planned("TupleQuery[tenant = 1, ts = 10]", &[2], &[])
    )]
    #[case::timestamp_string_literal(
        "tenant = 1 AND ts = 10 AND at >= '2024-01-01'",
        planned("TupleQuery[tenant = 1, ts = 10]", &[2], &[])
    )]
    #[case::literal_overflows_key("tenant = 3000000000", None)]
    #[case::float_literal_on_integer_key("tenant = 1.5", None)]
    #[case::or_across_keys("tenant = 1 OR ts = 2", None)]
    #[case::or_on_one_key("tenant = 1 OR tenant = 2", None)]
    #[case::not_equality("NOT tenant = 1", None)]
    #[case::not_equal("tenant != 1", None)]
    #[case::not_in("tenant NOT IN (1, 2)", None)]
    #[case::not_between("tenant NOT BETWEEN 1 AND 2", None)]
    #[case::equality_on_second_key_only("ts = 10", None)]
    #[case::equality_on_later_keys_only("ts = 10 AND at = TIMESTAMP '2024-01-01 00:00:00'", None)]
    #[case::cast_key("CAST(tenant AS BIGINT) = 1", None)]
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
    fn plans_conjunctive_predicate(#[case] predicate: String, #[case] expected: Option<Expected>) {
        let predicate = FIXTURE
            .session
            .state()
            .create_logical_expr(&predicate, &FIXTURE.schema)
            .unwrap();
        let conjuncts = split_conjunction(&predicate)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        check_plan(&conjuncts, expected);
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
            planned("TupleQuery[tenant = 1, ts = 10]", &[2], &[]),
        );
    }
}
