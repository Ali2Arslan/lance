// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Logical queries over an ordered list of key columns.
//!
//! A [`TupleQuery`] says which key tuples match, not how an index stores them.
//! A multi-column index lowers it to its own representation, such as encoded
//! bytes or typed comparisons. [`planner`] builds one from a conjunctive filter.
//!
//! ```
//! use std::ops::Bound;
//!
//! use arrow_schema::{DataType, Field, Fields};
//! use datafusion_common::ScalarValue;
//! use lance_index::scalar::tuple::{KeyValues, TupleQuery};
//!
//! # fn main() -> lance_core::Result<()> {
//! let keys = Fields::from(vec![
//!     Field::new("tenant", DataType::Int32, true),
//!     Field::new("ts", DataType::Int64, true),
//! ]);
//! let tenants = vec![ScalarValue::Int32(Some(1)), ScalarValue::Int32(Some(2))];
//! let ts = (
//!     Bound::Included(ScalarValue::Int64(Some(10))),
//!     Bound::Excluded(ScalarValue::Int64(Some(20))),
//! );
//! let query = TupleQuery::try_new(keys, vec![KeyValues::In(tenants)], Some(ts))?;
//! assert_eq!(
//!     query.to_string(),
//!     "TupleQuery[tenant IN (1, 2), ts >= 10 AND ts < 20]"
//! );
//! # Ok(())
//! # }
//! ```

use std::any::Any;
use std::fmt;
use std::ops::Bound;

use arrow_schema::{Field, Fields};
use datafusion_common::{Column, ScalarValue};
use datafusion_expr::{Expr, lit};
use lance_core::{Error, Result};

use super::AnyQuery;

pub mod planner;

/// The values one key of a [`TupleQuery`] prefix may take.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KeyValues {
    /// The key is NULL.
    Null,
    /// The key equals one of these non-null values. An empty list matches no key.
    In(Vec<ScalarValue>),
}

/// Lower and upper bounds on one key. A NULL key is never within them.
pub type KeyRange = (Bound<ScalarValue>, Bound<ScalarValue>);

/// A search over the leading columns of an ordered key.
///
/// The query fixes keys `0..prefix.len()`, each to its [`KeyValues`], so it
/// covers the cross product of those sets. When `range` is set, the next key is
/// non-null and within the bounds, so an unbounded range means `IS NOT NULL`.
/// Later keys are unconstrained.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TupleQuery {
    keys: Fields,
    prefix: Vec<KeyValues>,
    range: Option<KeyRange>,
}

impl TupleQuery {
    /// Build a query over `keys`, the index's key columns in key order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when `prefix` and `range` constrain more
    /// keys than `keys` holds, or when a value or bound is NULL or does not have
    /// its key's type.
    pub fn try_new(keys: Fields, prefix: Vec<KeyValues>, range: Option<KeyRange>) -> Result<Self> {
        let constrained = prefix.len() + usize::from(range.is_some());
        if constrained > keys.len() {
            return Err(Error::invalid_input(format!(
                "TupleQuery constrains {constrained} keys ({} in the prefix, range: {}) but the index has {} keys",
                prefix.len(),
                range.is_some(),
                keys.len()
            )));
        }
        for (values, key) in prefix.iter().zip(keys.iter()) {
            if let KeyValues::In(values) = values {
                for value in values {
                    check_value(value, key)?;
                }
            }
        }
        if let Some((lower, upper)) = &range {
            for bound in [lower, upper] {
                if let Bound::Included(value) | Bound::Excluded(value) = bound {
                    check_value(value, &keys[prefix.len()])?;
                }
            }
        }
        Ok(Self {
            keys,
            prefix,
            range,
        })
    }

    /// The index's key columns, in key order.
    pub fn keys(&self) -> &Fields {
        &self.keys
    }

    /// The values each leading key may take.
    pub fn prefix(&self) -> &[KeyValues] {
        &self.prefix
    }

    /// The bounds on the key after the prefix, if it is constrained.
    pub fn range(&self) -> Option<&KeyRange> {
        self.range.as_ref()
    }
}

fn check_value(value: &ScalarValue, key: &Field) -> Result<()> {
    if value.is_null() || value.data_type() != *key.data_type() {
        return Err(Error::invalid_input(format!(
            "TupleQuery value {value:?} for key '{}' must be a non-null {}",
            key.name(),
            key.data_type()
        )));
    }
    Ok(())
}

impl fmt::Display for TupleQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::with_capacity(self.prefix.len() + 1);
        for (values, key) in self.prefix.iter().zip(self.keys.iter()) {
            let name = key.name();
            terms.push(match values {
                KeyValues::Null => format!("{name} IS NULL"),
                KeyValues::In(values) if values.len() == 1 => format!("{name} = {}", values[0]),
                KeyValues::In(values) => {
                    let values = values.iter().map(ToString::to_string).collect::<Vec<_>>();
                    format!("{name} IN ({})", values.join(", "))
                }
            });
        }
        if let Some((lower, upper)) = &self.range {
            let name = self.keys[self.prefix.len()].name();
            let lower = match lower {
                Bound::Included(value) => Some(format!("{name} >= {value}")),
                Bound::Excluded(value) => Some(format!("{name} > {value}")),
                Bound::Unbounded => None,
            };
            let upper = match upper {
                Bound::Included(value) => Some(format!("{name} <= {value}")),
                Bound::Excluded(value) => Some(format!("{name} < {value}")),
                Bound::Unbounded => None,
            };
            terms.push(match (lower, upper) {
                (Some(lower), Some(upper)) => format!("{lower} AND {upper}"),
                (Some(bound), None) | (None, Some(bound)) => bound,
                (None, None) => format!("{name} IS NOT NULL"),
            });
        }
        write!(f, "TupleQuery[{}]", terms.join(", "))
    }
}

impl AnyQuery for TupleQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, _col: &str) -> String {
        self.to_string()
    }

    fn to_expr(&self, _col: String) -> Expr {
        let column = |key: &Field| Expr::Column(Column::new_unqualified(key.name()));
        let mut terms = Vec::with_capacity(self.prefix.len() + 2);
        for (values, key) in self.prefix.iter().zip(self.keys.iter()) {
            terms.push(match values {
                KeyValues::Null => column(key).is_null(),
                KeyValues::In(values) if values.is_empty() => lit(false),
                KeyValues::In(values) if values.len() == 1 => {
                    column(key).eq(lit(values[0].clone()))
                }
                KeyValues::In(values) => {
                    column(key).in_list(values.iter().cloned().map(lit).collect(), false)
                }
            });
        }
        if let Some((lower, upper)) = &self.range {
            let key = column(&self.keys[self.prefix.len()]);
            match lower {
                Bound::Included(value) => terms.push(key.clone().gt_eq(lit(value.clone()))),
                Bound::Excluded(value) => terms.push(key.clone().gt(lit(value.clone()))),
                Bound::Unbounded => {}
            }
            match upper {
                Bound::Included(value) => terms.push(key.clone().lt_eq(lit(value.clone()))),
                Bound::Excluded(value) => terms.push(key.clone().lt(lit(value.clone()))),
                Bound::Unbounded => {}
            }
            if matches!((lower, upper), (Bound::Unbounded, Bound::Unbounded)) {
                terms.push(key.is_not_null());
            }
        }
        terms.into_iter().reduce(Expr::and).unwrap_or(lit(true))
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use rstest::rstest;

    use super::*;
    use crate::scalar::SargableQuery;

    fn keys() -> Fields {
        Fields::from(vec![
            Field::new("tenant", DataType::Int32, true),
            Field::new("ts", DataType::Int64, true),
        ])
    }

    #[rstest]
    #[case::prefix_longer_than_keys(vec![KeyValues::Null; 3], None)]
    #[case::range_after_last_key(
        vec![KeyValues::Null; 2],
        Some((Bound::Unbounded, Bound::Unbounded))
    )]
    #[case::value_of_other_type(vec![KeyValues::In(vec![ScalarValue::Int64(Some(1))])], None)]
    #[case::null_value(vec![KeyValues::In(vec![ScalarValue::Int32(None)])], None)]
    #[case::null_bound(
        vec![KeyValues::Null],
        Some((Bound::Included(ScalarValue::Int64(None)), Bound::Unbounded))
    )]
    #[case::bound_of_other_type(
        vec![KeyValues::Null],
        Some((Bound::Unbounded, Bound::Excluded(ScalarValue::Int32(Some(1)))))
    )]
    fn try_new_rejects_invalid_query(
        #[case] prefix: Vec<KeyValues>,
        #[case] range: Option<KeyRange>,
    ) {
        let error = TupleQuery::try_new(keys(), prefix, range).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }

    #[test]
    fn any_query_compares_and_formats_by_value() {
        let query = TupleQuery::try_new(
            keys(),
            vec![KeyValues::Null],
            Some((
                Bound::Excluded(ScalarValue::Int64(Some(5))),
                Bound::Unbounded,
            )),
        )
        .unwrap();
        let unbounded = TupleQuery::try_new(
            keys(),
            vec![KeyValues::Null],
            Some((Bound::Unbounded, Bound::Unbounded)),
        )
        .unwrap();
        assert!(query.dyn_eq(&query.clone()));
        assert!(!query.dyn_eq(&unbounded));
        assert!(!query.dyn_eq(&SargableQuery::IsNull()));
        assert_eq!(query.format("unused"), "TupleQuery[tenant IS NULL, ts > 5]");
        assert_eq!(
            unbounded.format("unused"),
            "TupleQuery[tenant IS NULL, ts IS NOT NULL]"
        );
    }
}
