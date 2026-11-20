// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The key layout of a BTree index whose pages store encoded key tuples.
//!
//! [`BTreeKeyLayout::try_new`] applies the Key Layout Reader Rules of the
//! format spec (`docs/src/format/index/scalar/btree.md`) to a manifest's index
//! segment. The layout then lowers key predicates to ranges of encoded keys.

use std::ops::Bound;

use arrow_schema::{DataType, SortOptions};
use datafusion_common::ScalarValue;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_table::feature_flags::FLAG_BTREE_KEY_LAYOUT;
use lance_table::format::{IndexMetadata, Manifest};

use super::arrow_row::{ArrowRowV1, KeyBytes, next_prefix};
use crate::pbold::BTreeIndexDetails;
use crate::pbold::b_tree_key_layout::{ArrowRow, key_encoding, key_ordering};
use crate::scalar::tuple::KeyRange;

/// The BTree index version of a segment that sets `key_layout`.
pub const BTREE_KEY_LAYOUT_VERSION: u32 = 2;

/// One key column of a BTree index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BTreeKey {
    /// The path of the key column in the dataset schema, for messages.
    pub name: String,
    /// The type of the key column in the dataset schema.
    pub data_type: DataType,
    /// The direction and the null placement of the key.
    pub options: SortOptions,
}

/// The encoded keys `start <= key < end`, compared as unsigned byte strings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedRange {
    pub start: Vec<u8>,
    /// `None` when the range has no upper bound.
    pub end: Option<Vec<u8>>,
}

/// The validated `key_layout` of a BTree index segment.
///
/// Only the `arrow_row` version 1 encoding and the `lexicographic` ordering
/// pass validation, so a layout always encodes with [`ArrowRowV1`].
#[derive(Debug)]
pub struct BTreeKeyLayout {
    codec: ArrowRowV1,
}

impl BTreeKeyLayout {
    /// Reads the key layout of `index`, a BTree segment of `manifest`.
    ///
    /// Returns `None` when the segment's details have no `key_layout`. Its
    /// pages then store raw values, as in index version 1, however many
    /// entries `fields` has.
    ///
    /// # Errors
    ///
    /// [`Error::NotSupported`] when the segment is unusable under the Key
    /// Layout Reader Rules: the manifest's writer feature flags lack
    /// `FLAG_BTREE_KEY_LAYOUT`, `index_version` is below
    /// [`BTREE_KEY_LAYOUT_VERSION`], `fields` is empty, repeats a field, or
    /// names a field missing from the schema, `covering_fields` is not empty,
    /// `key_layout.keys` and `fields` differ in length, the encoding is not
    /// `arrow_row` version 1, the ordering is not `lexicographic`, or a key
    /// type is not admitted. The message names the index and every key.
    /// [`Error::InvalidInput`] when the details are not BTree details.
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use std::sync::Arc;
    ///
    /// use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    /// use lance_core::datatypes::Schema;
    /// use lance_index::pbold::b_tree_key_layout::{
    ///     ArrowRow, KeyEncoding, KeyField, KeyOrdering, Lexicographic, key_encoding, key_ordering,
    /// };
    /// use lance_index::pbold::{BTreeIndexDetails, BTreeKeyLayout as KeyLayoutProto};
    /// use lance_index::scalar::btree::key::BTreeKeyLayout;
    /// use lance_table::feature_flags::FLAG_BTREE_KEY_LAYOUT;
    /// use lance_table::format::{DataStorageFormat, IndexMetadata, Manifest};
    ///
    /// # fn main() -> lance_core::Result<()> {
    /// let schema = Schema::try_from(&ArrowSchema::new(vec![
    ///     Field::new("tenant", DataType::Int64, true),
    ///     Field::new("name", DataType::Utf8, true),
    /// ]))?;
    /// let mut manifest = Manifest::new(
    ///     schema,
    ///     Arc::new(vec![]),
    ///     DataStorageFormat::default(),
    ///     HashMap::new(),
    /// );
    /// manifest.writer_feature_flags |= FLAG_BTREE_KEY_LAYOUT;
    /// let details = BTreeIndexDetails {
    ///     key_layout: Some(KeyLayoutProto {
    ///         encoding: Some(KeyEncoding {
    ///             encoding: Some(key_encoding::Encoding::ArrowRow(ArrowRow { version: 1 })),
    ///         }),
    ///         ordering: Some(KeyOrdering {
    ///             ordering: Some(key_ordering::Ordering::Lexicographic(Lexicographic {})),
    ///         }),
    ///         keys: vec![
    ///             KeyField { descending: false, nulls_first: true },
    ///             KeyField { descending: true, nulls_first: false },
    ///         ],
    ///     }),
    /// };
    /// let index = IndexMetadata {
    ///     uuid: uuid::Uuid::new_v4(),
    ///     fields: vec![0, 1],
    ///     covering_fields: vec![],
    ///     name: "tenant_name".to_string(),
    ///     dataset_version: 1,
    ///     fragment_bitmap: None,
    ///     index_details: Some(Arc::new(prost_types::Any::from_msg(&details).unwrap())),
    ///     index_version: 2,
    ///     created_at: None,
    ///     base_id: None,
    ///     files: None,
    ///     clustering_id: None,
    ///     metadata: HashMap::new(),
    /// };
    /// let layout = BTreeKeyLayout::try_new(&index, &manifest)?.expect("details set key_layout");
    /// let names: Vec<_> = layout.keys().iter().map(|key| key.name.as_str()).collect();
    /// assert_eq!(names, ["tenant", "name"]);
    /// assert!(layout.keys()[1].options.descending);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(index: &IndexMetadata, manifest: &Manifest) -> Result<Option<Self>> {
        let Some(details) = &index.index_details else {
            return Ok(None);
        };
        let details = details.to_msg::<BTreeIndexDetails>().map_err(|error| {
            Error::invalid_input(format!(
                "index '{}' has details of type {}, not BTree details: {error}",
                index.name, details.type_url
            ))
        })?;
        let Some(layout) = details.key_layout else {
            return Ok(None);
        };
        let schema = &manifest.schema;
        let unusable = |problem: String| unusable(index, schema, problem);

        if manifest.writer_feature_flags & FLAG_BTREE_KEY_LAYOUT == 0 {
            return Err(unusable(format!(
                "the manifest's writer feature flags {:#x} lack FLAG_BTREE_KEY_LAYOUT",
                manifest.writer_feature_flags
            )));
        }
        if !u32::try_from(index.index_version).is_ok_and(|v| v >= BTREE_KEY_LAYOUT_VERSION) {
            return Err(unusable(format!(
                "index_version {} is below {BTREE_KEY_LAYOUT_VERSION}",
                index.index_version
            )));
        }
        if index.fields.is_empty() {
            return Err(unusable("fields is empty".to_string()));
        }
        if !index.covering_fields.is_empty() {
            return Err(unusable(format!(
                "covering_fields is {:?}, but a key layout defines no covering fields",
                index.covering_fields
            )));
        }
        if layout.keys.len() != index.fields.len() {
            return Err(unusable(format!(
                "key_layout has {} keys for {} fields",
                layout.keys.len(),
                index.fields.len()
            )));
        }
        match layout.encoding.and_then(|encoding| encoding.encoding) {
            Some(key_encoding::Encoding::ArrowRow(ArrowRow { version: 1 })) => {}
            Some(key_encoding::Encoding::ArrowRow(ArrowRow { version })) => {
                return Err(unusable(format!(
                    "encoding arrow_row version {version} is not supported, only version 1"
                )));
            }
            None => {
                return Err(unusable(
                    "encoding is missing or of a kind this reader does not know".to_string(),
                ));
            }
        }
        match layout.ordering.and_then(|ordering| ordering.ordering) {
            Some(key_ordering::Ordering::Lexicographic(_)) => {}
            None => {
                return Err(unusable(
                    "ordering is missing or of a kind this reader does not know".to_string(),
                ));
            }
        }

        let mut keys = Vec::with_capacity(index.fields.len());
        for (position, (&field_id, key)) in index.fields.iter().zip(&layout.keys).enumerate() {
            if let Some(first) = index.fields[..position]
                .iter()
                .position(|id| *id == field_id)
            {
                return Err(unusable(format!(
                    "key {position} repeats field id {field_id} of key {first}"
                )));
            }
            let Some((name, data_type)) = key_column(schema, field_id) else {
                return Err(unusable(format!(
                    "key {position} names field id {field_id}, which is not in the dataset schema"
                )));
            };
            if KeyBytes::of(&data_type).is_none() {
                return Err(unusable(format!(
                    "key {position} '{name}' has type {data_type}, which the arrow_row version 1 \
                     encoding does not admit"
                )));
            }
            keys.push(BTreeKey {
                name,
                data_type,
                options: SortOptions {
                    descending: key.descending,
                    nulls_first: key.nulls_first,
                },
            });
        }
        Ok(Some(Self {
            codec: ArrowRowV1::try_new(keys)?,
        }))
    }

    /// The keys, in key order.
    pub fn keys(&self) -> &[BTreeKey] {
        self.codec.keys()
    }

    /// The codec that encodes and decodes this layout's key tuples.
    pub fn codec(&self) -> &ArrowRowV1 {
        &self.codec
    }

    /// The encoded keys of the tuples whose first `prefix.len()` keys equal
    /// `prefix`. A typed null in `prefix` matches a null key, and an empty
    /// prefix matches every tuple.
    ///
    /// # Errors
    ///
    /// As [`ArrowRowV1::encode_prefix`].
    pub fn prefix_range(&self, prefix: &[ScalarValue]) -> Result<EncodedRange> {
        let start = self.codec.encode_prefix(prefix)?;
        let end = next_prefix(&start);
        Ok(EncodedRange { start, end })
    }

    /// The encoded keys of the tuples whose first `prefix.len()` keys equal
    /// `prefix` and whose next key is within `range`. As in
    /// [`TupleQuery`](crate::scalar::tuple::TupleQuery), a null key is never
    /// within a range, so an unbounded range selects the non-null keys.
    ///
    /// Returns `None` when no encoded key can be within the range.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when no key follows the prefix, or a bound is
    /// null, and otherwise as [`ArrowRowV1::encode_prefix`].
    pub fn range_after_prefix(
        &self,
        prefix: &[ScalarValue],
        (lower, upper): &KeyRange,
    ) -> Result<Option<EncodedRange>> {
        let Some(key) = self.keys().get(prefix.len()) else {
            return Err(Error::invalid_input(format!(
                "a range after a prefix of {} values needs another key, but there are {} keys",
                prefix.len(),
                self.keys().len()
            )));
        };
        let encode_with = |value: &ScalarValue| {
            let mut tuple = prefix.to_vec();
            tuple.push(value.clone());
            self.codec.encode_prefix(&tuple)
        };
        let encode_bound = |value: &ScalarValue| {
            if value.is_null() {
                return Err(Error::invalid_input(format!(
                    "a range bound on key {} '{}' is null",
                    prefix.len(),
                    key.name
                )));
            }
            encode_with(value)
        };
        let (start_bound, end_bound) = if key.options.descending {
            (upper, lower)
        } else {
            (lower, upper)
        };
        // The null sentinel puts the key's nulls at the low end of the bytes
        // when `nulls_first` is set, whatever the direction.
        let null = ScalarValue::try_new_null(&key.data_type)?;
        let start = match start_bound {
            Bound::Included(value) => encode_bound(value)?,
            Bound::Excluded(value) => match next_prefix(&encode_bound(value)?) {
                Some(start) => start,
                None => return Ok(None),
            },
            Bound::Unbounded if key.options.nulls_first => {
                match next_prefix(&encode_with(&null)?) {
                    Some(start) => start,
                    None => return Ok(None),
                }
            }
            Bound::Unbounded => self.codec.encode_prefix(prefix)?,
        };
        let end = match end_bound {
            Bound::Included(value) => next_prefix(&encode_bound(value)?),
            Bound::Excluded(value) => Some(encode_bound(value)?),
            Bound::Unbounded if key.options.nulls_first => {
                next_prefix(&self.codec.encode_prefix(prefix)?)
            }
            Bound::Unbounded => Some(encode_with(&null)?),
        };
        if end.as_ref().is_some_and(|end| start >= *end) {
            return Ok(None);
        }
        Ok(Some(EncodedRange { start, end }))
    }
}

fn key_column(schema: &Schema, field_id: i32) -> Option<(String, DataType)> {
    let field = schema.field_by_id(field_id)?;
    Some((schema.field_path(field_id).ok()?, field.data_type()))
}

fn unusable(index: &IndexMetadata, schema: &Schema, problem: String) -> Error {
    let keys = index
        .fields
        .iter()
        .enumerate()
        .map(|(position, &field_id)| match key_column(schema, field_id) {
            Some((name, data_type)) => format!("{position}: '{name}' {data_type}"),
            None => format!("{position}: field id {field_id}, not in the schema"),
        })
        .collect::<Vec<_>>()
        .join(", ");
    Error::not_supported(format!(
        "BTree index '{}' with keys [{keys}] is unusable: {problem}",
        index.name
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use arrow_schema::{Field, Fields, Schema as ArrowSchema, TimeUnit};
    use lance_table::format::DataStorageFormat;
    use prost::Message;
    use rstest::rstest;

    use super::*;
    use crate::pbold::b_tree_key_layout::{KeyEncoding, KeyField, KeyOrdering, Lexicographic};
    use crate::pbold::{BTreeKeyLayout as KeyLayoutProto, BitmapIndexDetails};

    const OPTIONS: [SortOptions; 4] = [
        SortOptions {
            descending: false,
            nulls_first: true,
        },
        SortOptions {
            descending: false,
            nulls_first: false,
        },
        SortOptions {
            descending: true,
            nulls_first: true,
        },
        SortOptions {
            descending: true,
            nulls_first: false,
        },
    ];

    fn timestamp() -> DataType {
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    }

    const TENANT: i32 = 0;
    const NAME: i32 = 1;
    const S_TS: i32 = 3;
    const AMOUNT: i32 = 4;
    const NOT_IN_SCHEMA: i32 = 9;

    fn manifest() -> Manifest {
        let schema = Schema::try_from(&ArrowSchema::new(vec![
            Field::new("tenant", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
            Field::new(
                "s",
                DataType::Struct(Fields::from(vec![Field::new("ts", timestamp(), true)])),
                true,
            ),
            Field::new("amount", DataType::Decimal128(10, 2), true),
        ]))
        .unwrap();
        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest.writer_feature_flags |= FLAG_BTREE_KEY_LAYOUT;
        manifest
    }

    fn layout_proto(options: &[SortOptions]) -> KeyLayoutProto {
        KeyLayoutProto {
            encoding: Some(KeyEncoding {
                encoding: Some(key_encoding::Encoding::ArrowRow(ArrowRow { version: 1 })),
            }),
            ordering: Some(KeyOrdering {
                ordering: Some(key_ordering::Ordering::Lexicographic(Lexicographic {})),
            }),
            keys: options
                .iter()
                .map(|options| KeyField {
                    descending: options.descending,
                    nulls_first: options.nulls_first,
                })
                .collect(),
        }
    }

    fn index(fields: Vec<i32>, key_layout: Option<KeyLayoutProto>) -> IndexMetadata {
        let details = BTreeIndexDetails { key_layout };
        IndexMetadata {
            uuid: uuid::Uuid::nil(),
            fields,
            covering_fields: vec![],
            name: "tenant_name".to_string(),
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: Some(Arc::new(prost_types::Any::from_msg(&details).unwrap())),
            index_version: BTREE_KEY_LAYOUT_VERSION as i32,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: HashMap::new(),
        }
    }

    fn edit_layout(index: &mut IndexMetadata, edit: fn(&mut KeyLayoutProto)) {
        let details = index.index_details.as_ref().unwrap();
        let mut details = details.to_msg::<BTreeIndexDetails>().unwrap();
        edit(details.key_layout.as_mut().unwrap());
        index.index_details = Some(Arc::new(prost_types::Any::from_msg(&details).unwrap()));
    }

    #[rstest]
    #[case::index_version_below_2(
        |index: &mut IndexMetadata| index.index_version = 1,
        "index_version 1 is below 2"
    )]
    #[case::fields_empty(
        |index: &mut IndexMetadata| {
            index.fields.clear();
            edit_layout(index, |layout| layout.keys.clear());
        },
        "keys [] is unusable: fields is empty"
    )]
    #[case::repeated_field(
        |index: &mut IndexMetadata| index.fields = vec![TENANT, TENANT],
        "key 1 repeats field id 0 of key 0"
    )]
    #[case::field_missing_from_schema(
        |index: &mut IndexMetadata| index.fields = vec![TENANT, NOT_IN_SCHEMA],
        "keys [0: 'tenant' Int64, 1: field id 9, not in the schema] is unusable: key 1 names field \
         id 9, which is not in the dataset schema"
    )]
    #[case::covering_fields_not_empty(
        |index: &mut IndexMetadata| index.covering_fields = vec![NAME],
        "covering_fields is [1], but a key layout defines no covering fields"
    )]
    #[case::key_count_mismatch(
        |index: &mut IndexMetadata| {
            edit_layout(index, |layout| layout.keys.push(KeyField::default()));
        },
        "key_layout has 3 keys for 2 fields"
    )]
    #[case::encoding_missing(
        |index: &mut IndexMetadata| edit_layout(index, |layout| layout.encoding = None),
        "encoding is missing or of a kind this reader does not know"
    )]
    #[case::encoding_unknown(
        |index: &mut IndexMetadata| edit_layout(index, |layout| {
            layout.encoding = Some(KeyEncoding::decode([0x12, 0x00].as_slice()).unwrap());
        }),
        "encoding is missing or of a kind this reader does not know"
    )]
    #[case::encoding_version_2(
        |index: &mut IndexMetadata| edit_layout(index, |layout| {
            layout.encoding = Some(KeyEncoding {
                encoding: Some(key_encoding::Encoding::ArrowRow(ArrowRow { version: 2 })),
            });
        }),
        "encoding arrow_row version 2 is not supported, only version 1"
    )]
    #[case::ordering_missing(
        |index: &mut IndexMetadata| edit_layout(index, |layout| layout.ordering = None),
        "ordering is missing or of a kind this reader does not know"
    )]
    #[case::ordering_unknown(
        |index: &mut IndexMetadata| edit_layout(index, |layout| {
            layout.ordering = Some(KeyOrdering::decode([0x12, 0x00].as_slice()).unwrap());
        }),
        "ordering is missing or of a kind this reader does not know"
    )]
    #[case::type_not_admitted(
        |index: &mut IndexMetadata| index.fields = vec![TENANT, AMOUNT],
        "key 1 'amount' has type Decimal128(10, 2), which the arrow_row version 1 encoding does \
         not admit"
    )]
    fn unusable_layouts_are_rejected(
        #[case] tamper: fn(&mut IndexMetadata),
        #[case] message: &str,
    ) {
        let manifest = manifest();
        let mut index = index(vec![TENANT, NAME], Some(layout_proto(&OPTIONS[..2])));
        assert!(
            BTreeKeyLayout::try_new(&index, &manifest)
                .unwrap()
                .is_some()
        );
        tamper(&mut index);
        let error = BTreeKeyLayout::try_new(&index, &manifest).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        let error = error.to_string();
        assert!(
            error.contains("BTree index 'tenant_name' with keys ["),
            "{error}"
        );
        assert!(error.contains(message), "{error}");
    }

    #[test]
    fn layout_is_unusable_without_the_writer_flag() {
        let index = index(vec![TENANT, NAME], Some(layout_proto(&OPTIONS[..2])));
        let mut manifest = manifest();
        manifest.writer_feature_flags &= !FLAG_BTREE_KEY_LAYOUT;
        let error = BTreeKeyLayout::try_new(&index, &manifest).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        let keys = "BTree index 'tenant_name' with keys [0: 'tenant' Int64, 1: 'name' Utf8]";
        assert!(error.to_string().contains(keys), "{error}");
        assert!(
            error.to_string().contains("lack FLAG_BTREE_KEY_LAYOUT"),
            "{error}"
        );
    }

    #[test]
    fn unusable_layout_names_every_key_with_its_type() {
        let mut index = index(vec![S_TS, AMOUNT], Some(layout_proto(&OPTIONS[..2])));
        index.index_version = 1;
        let error = BTreeKeyLayout::try_new(&index, &manifest()).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        let keys = format!("[0: 's.ts' {}, 1: 'amount' Decimal128(10, 2)]", timestamp());
        assert!(error.to_string().contains(&keys), "{error}");
    }

    #[rstest]
    #[case::one_key(vec![NAME], vec![("name", DataType::Utf8)])]
    #[case::nested_keys_in_field_order(
        vec![S_TS, TENANT, NAME],
        vec![("s.ts", timestamp()), ("tenant", DataType::Int64), ("name", DataType::Utf8)]
    )]
    fn layout_reads_keys_from_the_schema_in_field_order(
        #[case] fields: Vec<i32>,
        #[case] expected: Vec<(&str, DataType)>,
    ) {
        let options = &OPTIONS[OPTIONS.len() - fields.len()..];
        let index = index(fields, Some(layout_proto(options)));
        let layout = BTreeKeyLayout::try_new(&index, &manifest())
            .unwrap()
            .unwrap();
        let expected = expected
            .into_iter()
            .zip(options)
            .map(|((name, data_type), options)| BTreeKey {
                name: name.to_string(),
                data_type,
                options: *options,
            })
            .collect::<Vec<_>>();
        assert_eq!(layout.keys(), expected);
    }

    #[rstest]
    #[case::version_1(1, vec![TENANT], true)]
    #[case::version_2_without_key_layout(2, vec![TENANT], true)]
    #[case::two_fields_without_key_layout(2, vec![TENANT, NAME], true)]
    #[case::no_index_details(2, vec![TENANT, NAME], false)]
    fn segment_without_key_layout_has_plain_pages(
        #[case] index_version: i32,
        #[case] fields: Vec<i32>,
        #[case] has_details: bool,
    ) {
        let mut index = index(fields, None);
        index.index_version = index_version;
        if !has_details {
            index.index_details = None;
        }
        let mut manifest = manifest();
        manifest.writer_feature_flags = 0;
        assert!(
            BTreeKeyLayout::try_new(&index, &manifest)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn try_new_rejects_details_of_another_index_type() {
        let mut index = index(vec![TENANT], None);
        index.index_details = Some(Arc::new(
            prost_types::Any::from_msg(&BitmapIndexDetails::default()).unwrap(),
        ));
        let error = BTreeKeyLayout::try_new(&index, &manifest()).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("BitmapIndexDetails"), "{error}");
    }

    fn tenant_name_layout(tenant: SortOptions, name: SortOptions) -> BTreeKeyLayout {
        let index = index(vec![TENANT, NAME], Some(layout_proto(&[tenant, name])));
        BTreeKeyLayout::try_new(&index, &manifest())
            .unwrap()
            .unwrap()
    }

    fn tenants() -> Vec<ScalarValue> {
        [None, Some(-1), Some(0), Some(1)]
            .map(ScalarValue::Int64)
            .to_vec()
    }

    fn names() -> Vec<ScalarValue> {
        let long = "x".repeat(40);
        [
            None,
            Some(""),
            Some("a"),
            Some("ab"),
            Some("b"),
            Some("\u{10FFFF}"),
            Some(long.as_str()),
        ]
        .map(|name| ScalarValue::Utf8(name.map(str::to_string)))
        .to_vec()
    }

    fn every_tuple_with_its_key(layout: &BTreeKeyLayout) -> Vec<([ScalarValue; 2], Vec<u8>)> {
        let tuples = tenants()
            .into_iter()
            .flat_map(|tenant| names().into_iter().map(move |name| [tenant.clone(), name]))
            .collect::<Vec<_>>();
        let tenant: ArrayRef = Arc::new(Int64Array::from_iter(tuples.iter().map(|[t, _]| {
            let ScalarValue::Int64(t) = t else {
                unreachable!()
            };
            *t
        })));
        let name: ArrayRef = Arc::new(StringArray::from_iter(tuples.iter().map(|[_, n]| {
            let ScalarValue::Utf8(n) = n else {
                unreachable!()
            };
            n.clone()
        })));
        let encoded = layout.codec().encode(&[tenant, name]).unwrap();
        tuples
            .into_iter()
            .zip(encoded.iter().flatten().map(<[u8]>::to_vec))
            .collect()
    }

    fn contains(range: &EncodedRange, key: &[u8]) -> bool {
        key >= range.start.as_slice() && range.end.as_ref().is_none_or(|end| key < end.as_slice())
    }

    fn bounds(values: &[ScalarValue]) -> Vec<Bound<ScalarValue>> {
        let values = values.iter().filter(|value| !value.is_null());
        std::iter::once(Bound::Unbounded)
            .chain(values.clone().cloned().map(Bound::Included))
            .chain(values.cloned().map(Bound::Excluded))
            .collect()
    }

    fn within(value: &ScalarValue, (lower, upper): &KeyRange) -> bool {
        let above = match lower {
            Bound::Included(lower) => value >= lower,
            Bound::Excluded(lower) => value > lower,
            Bound::Unbounded => true,
        };
        let below = match upper {
            Bound::Included(upper) => value <= upper,
            Bound::Excluded(upper) => value < upper,
            Bound::Unbounded => true,
        };
        !value.is_null() && above && below
    }

    #[rstest]
    fn prefix_ranges_hold_exactly_the_tuples_with_the_prefix(
        #[values(OPTIONS[0], OPTIONS[1], OPTIONS[2], OPTIONS[3])] tenant: SortOptions,
        #[values(OPTIONS[0], OPTIONS[1], OPTIONS[2], OPTIONS[3])] name: SortOptions,
    ) {
        let layout = tenant_name_layout(tenant, name);
        let every_tuple = every_tuple_with_its_key(&layout);
        let prefixes = std::iter::once(vec![])
            .chain(tenants().into_iter().map(|tenant| vec![tenant]))
            .chain(every_tuple.iter().map(|(tuple, _)| tuple.to_vec()));
        for prefix in prefixes {
            let range = layout.prefix_range(&prefix).unwrap();
            for (tuple, key) in &every_tuple {
                assert_eq!(
                    contains(&range, key),
                    tuple.starts_with(&prefix),
                    "prefix {prefix:?}, tuple {tuple:?}"
                );
            }
        }
    }

    #[rstest]
    fn ranges_after_a_prefix_hold_exactly_the_tuples_within(
        #[values(OPTIONS[0], OPTIONS[1], OPTIONS[2], OPTIONS[3])] tenant: SortOptions,
        #[values(OPTIONS[0], OPTIONS[1], OPTIONS[2], OPTIONS[3])] name: SortOptions,
    ) {
        let layout = tenant_name_layout(tenant, name);
        let every_tuple = every_tuple_with_its_key(&layout);
        let mut cases = bounds(&tenants())
            .into_iter()
            .flat_map(|lower| {
                bounds(&tenants())
                    .into_iter()
                    .map(move |upper| (vec![], (lower.clone(), upper)))
            })
            .collect::<Vec<_>>();
        for prefix in tenants() {
            for lower in bounds(&names()) {
                for upper in bounds(&names()) {
                    cases.push((vec![prefix.clone()], (lower.clone(), upper)));
                }
            }
        }
        for (prefix, range) in cases {
            let encoded = layout.range_after_prefix(&prefix, &range).unwrap();
            for (tuple, key) in &every_tuple {
                let expected = tuple.starts_with(&prefix) && within(&tuple[prefix.len()], &range);
                let actual = encoded
                    .as_ref()
                    .is_some_and(|encoded| contains(encoded, key));
                assert_eq!(
                    actual, expected,
                    "prefix {prefix:?}, range {range:?}, tuple {tuple:?}"
                );
            }
        }
    }

    #[test]
    fn range_after_prefix_rejects_a_missing_key_and_a_null_bound() {
        let layout = tenant_name_layout(OPTIONS[0], OPTIONS[3]);
        let unbounded = (Bound::Unbounded, Bound::Unbounded);
        let prefix = [ScalarValue::Int64(Some(1)), ScalarValue::Utf8(None)];
        let error = layout.range_after_prefix(&prefix, &unbounded).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("needs another key, but there are 2 keys"),
            "{error}"
        );

        let null_bound = (Bound::Included(ScalarValue::Utf8(None)), Bound::Unbounded);
        let error = layout
            .range_after_prefix(&prefix[..1], &null_bound)
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("bound on key 1 'name' is null"),
            "{error}"
        );
    }
}
