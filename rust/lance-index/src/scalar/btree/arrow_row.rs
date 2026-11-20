// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The `arrow_row` version 1 encoding of BTree key tuples.
//!
//! The format spec (`docs/src/format/index/scalar/btree.md`) defines these
//! bytes. They are the bytes that `RowConverter` in `arrow-row` 59.3.3 writes,
//! so [`ArrowRowV1`] wraps it. The goldens in this module's tests pin the
//! bytes. If an Arrow upgrade changes them for an admitted type, the Encoding
//! Stability section of the spec requires vendoring the version 1 encoder for
//! that type rather than writing different bytes.

use std::sync::Arc;

use arrow::row::{RowConverter, SortField};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BinaryArray, LargeStringArray, StringArray};
use arrow_schema::{DataType, SortOptions};
use datafusion_common::ScalarValue;
use lance_core::{Error, Result};

use super::key::BTreeKey;

const MINI_BLOCKS: usize = 4;
const MINI_BLOCK_WIDTH: usize = 8;
const BLOCK_WIDTH: usize = 32;
const BLOCK_CONTINUATION: u8 = 0xFF;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KeyBytes {
    Boolean,
    Fixed { value_width: usize },
    Variable,
}

impl KeyBytes {
    pub(super) fn of(data_type: &DataType) -> Option<Self> {
        let value_width = match data_type {
            DataType::Boolean => return Some(Self::Boolean),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
                return Some(Self::Variable);
            }
            DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => 4,
            DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Timestamp(_, _) => 8,
            _ => return None,
        };
        Some(Self::Fixed { value_width })
    }

    fn canonical_len(self, bytes: &[u8], options: SortOptions) -> Option<usize> {
        let null = if options.nulls_first { 0x00 } else { 0xFF };
        let invert = if options.descending { 0xFF } else { 0x00 };
        let (&first, rest) = bytes.split_first()?;
        let value_width = match self {
            Self::Boolean => 1,
            Self::Fixed { value_width } => value_width,
            Self::Variable => {
                if first == null {
                    return Some(1);
                }
                return match first ^ invert {
                    0x01 => Some(1),
                    0x02 => canonical_blocks_len(rest, invert).map(|len| 1 + len),
                    _ => None,
                };
            }
        };
        let value = rest.get(..value_width)?;
        let is_canonical = if first == null {
            value.iter().all(|byte| *byte == 0)
        } else {
            first == 0x01 && (self != Self::Boolean || (value[0] ^ invert) <= 1)
        };
        is_canonical.then_some(1 + value_width)
    }
}

fn canonical_blocks_len(bytes: &[u8], invert: u8) -> Option<usize> {
    let mut len = 0;
    for block in 0.. {
        let width = if block < MINI_BLOCKS {
            MINI_BLOCK_WIDTH
        } else {
            BLOCK_WIDTH
        };
        let value = bytes.get(len..len + width)?;
        let marker = bytes.get(len + width)? ^ invert;
        len += width + 1;
        if marker != BLOCK_CONTINUATION {
            let used = usize::from(marker);
            let is_canonical =
                (1..=width).contains(&used) && value[used..].iter().all(|byte| *byte == invert);
            return is_canonical.then_some(len);
        }
    }
    None
}

/// The smallest byte string that sorts after every byte string starting with
/// `prefix`, or `None` when there is none because `prefix` is empty or all
/// `0xFF`.
///
/// `[prefix, next_prefix(prefix))` holds exactly the byte strings that start
/// with `prefix`.
pub fn next_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut next = prefix.to_vec();
    while let Some(last) = next.pop() {
        if last < u8::MAX {
            next.push(last + 1);
            return Some(next);
        }
    }
    None
}

/// Encodes key tuples as `arrow_row` version 1 bytes and decodes them.
///
/// Comparing two encoded tuples as unsigned byte strings orders them as the
/// `lexicographic` ordering orders the tuples, under each key's direction and
/// null placement.
///
/// ```
/// use std::sync::Arc;
///
/// use arrow_array::{ArrayRef, Int32Array, StringArray};
/// use arrow_schema::{DataType, SortOptions};
/// use lance_index::scalar::btree::arrow_row::ArrowRowV1;
/// use lance_index::scalar::btree::key::BTreeKey;
///
/// # fn main() -> lance_core::Result<()> {
/// let key = |name: &str, data_type, descending, nulls_first| BTreeKey {
///     name: name.to_string(),
///     data_type,
///     options: SortOptions { descending, nulls_first },
/// };
/// let codec = ArrowRowV1::try_new(vec![
///     key("tenant", DataType::Int32, false, true),
///     key("name", DataType::Utf8, true, false),
/// ])?;
/// let columns: Vec<ArrayRef> = vec![
///     Arc::new(Int32Array::from(vec![Some(-1), None])),
///     Arc::new(StringArray::from(vec![Some("ab"), None])),
/// ];
/// let encoded = codec.encode(&columns)?;
/// assert_eq!(encoded.value(0), b"\x01\x7f\xff\xff\xff\xfd\x9e\x9d\xff\xff\xff\xff\xff\xff\xfd");
/// assert_eq!(encoded.value(1), b"\x00\x00\x00\x00\x00\xff");
/// assert_eq!(codec.decode(&encoded)?, columns);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ArrowRowV1 {
    keys: Vec<BTreeKey>,
    layouts: Vec<KeyBytes>,
    encoder: RowConverter,
    /// Reads Utf8 keys as Binary, which has the same bytes, so that invalid
    /// UTF-8 in a stored key is an error instead of a panic in `arrow-row`.
    decoder: RowConverter,
}

impl ArrowRowV1 {
    /// Builds the codec for `keys`, in key order.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when `keys` is empty, and
    /// [`Error::NotSupported`] when the encoding does not admit the type of a
    /// key.
    pub fn try_new(keys: Vec<BTreeKey>) -> Result<Self> {
        if keys.is_empty() {
            return Err(Error::invalid_input(
                "an arrow_row version 1 codec needs at least one key",
            ));
        }
        let layouts = keys
            .iter()
            .enumerate()
            .map(|(position, key)| {
                KeyBytes::of(&key.data_type).ok_or_else(|| {
                    Error::not_supported(format!(
                        "key {position} '{}' has type {}, which the arrow_row version 1 \
                         encoding does not admit",
                        key.name, key.data_type
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let decoded_type = |data_type: &DataType| match data_type {
            DataType::Utf8 => DataType::Binary,
            DataType::LargeUtf8 => DataType::LargeBinary,
            other => other.clone(),
        };
        let encoder = RowConverter::new(sort_fields(&keys, DataType::clone))?;
        let decoder = RowConverter::new(sort_fields(&keys, decoded_type))?;
        Ok(Self {
            keys,
            layouts,
            encoder,
            decoder,
        })
    }

    /// The keys, in key order.
    pub fn keys(&self) -> &[BTreeKey] {
        &self.keys
    }

    /// Encodes the tuple in each row of `columns`, which hold the keys in key
    /// order.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when `columns` does not have one column per
    /// key, or a column does not have its key's type, and
    /// [`Error::Arrow`] when the columns differ in length or the encoded
    /// tuples exceed 2 GiB.
    pub fn encode(&self, columns: &[ArrayRef]) -> Result<BinaryArray> {
        if columns.len() != self.keys.len() {
            return Err(Error::invalid_input(format!(
                "encode takes one column for each of the {} keys, but got {} columns",
                self.keys.len(),
                columns.len()
            )));
        }
        self.check_types(columns)?;
        Ok(self.encoder.convert_columns(columns)?.try_into_binary()?)
    }

    /// Encodes one tuple prefix: a value for each of the first
    /// `values.len()` keys. A typed null is a null key, and an empty prefix
    /// encodes to no bytes.
    ///
    /// The result is a byte prefix of the encoding of every tuple that starts
    /// with these values.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when there are more values than keys, or a
    /// value does not have its key's type.
    pub fn encode_prefix(&self, values: &[ScalarValue]) -> Result<Vec<u8>> {
        let keys = self.keys.get(..values.len()).ok_or_else(|| {
            Error::invalid_input(format!(
                "a prefix of {} values is longer than the {} keys",
                values.len(),
                self.keys.len()
            ))
        })?;
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let columns = values
            .iter()
            .map(ScalarValue::to_array)
            .collect::<datafusion_common::Result<Vec<_>>>()?;
        self.check_types(&columns)?;
        let converter = RowConverter::new(sort_fields(keys, DataType::clone))?;
        Ok(converter
            .convert_columns(&columns)?
            .row(0)
            .as_ref()
            .to_vec())
    }

    /// Decodes each encoded tuple of `keys` into one column per key, in key
    /// order.
    ///
    /// # Errors
    ///
    /// [`Error::CorruptFile`] when a row is null or is not the canonical
    /// encoding of a tuple of these keys.
    pub fn decode(&self, keys: &BinaryArray) -> Result<Vec<ArrayRef>> {
        for (row, key) in keys.iter().enumerate() {
            let Some(key) = key else {
                return Err(corrupt_key(format!(
                    "row {row} is null, but an encoded key is never null"
                )));
            };
            self.check_canonical(row, key)?;
        }
        let parser = self.decoder.parser();
        let columns = self
            .decoder
            .convert_rows(keys.iter().flatten().map(|key| parser.parse(key)))?;
        columns
            .into_iter()
            .zip(&self.keys)
            .enumerate()
            .map(|(position, (column, key))| {
                let column: ArrayRef = match key.data_type {
                    DataType::Utf8 => Arc::new(
                        StringArray::try_from_binary(column.as_binary::<i32>().clone())
                            .map_err(|error| invalid_utf8(position, key, error))?,
                    ),
                    DataType::LargeUtf8 => Arc::new(
                        LargeStringArray::try_from_binary(column.as_binary::<i64>().clone())
                            .map_err(|error| invalid_utf8(position, key, error))?,
                    ),
                    _ => column,
                };
                Ok(column)
            })
            .collect()
    }

    fn check_types(&self, columns: &[ArrayRef]) -> Result<()> {
        for (position, (column, key)) in columns.iter().zip(&self.keys).enumerate() {
            if column.data_type() != &key.data_type {
                return Err(Error::invalid_input(format!(
                    "key {position} '{}' has type {}, but its column has type {}",
                    key.name,
                    key.data_type,
                    column.data_type()
                )));
            }
        }
        Ok(())
    }

    /// Checks that `bytes` is the canonical encoding of one tuple, because
    /// `arrow-row` panics on some malformed rows and decodes others to values
    /// that encode differently.
    fn check_canonical(&self, row: usize, mut bytes: &[u8]) -> Result<()> {
        for (position, (key, layout)) in self.keys.iter().zip(&self.layouts).enumerate() {
            let len = layout.canonical_len(bytes, key.options).ok_or_else(|| {
                corrupt_key(format!(
                    "row {row} key {position} '{}' ({}, {:?}) is not a canonical arrow_row \
                     version 1 encoding",
                    key.name, key.data_type, key.options
                ))
            })?;
            bytes = &bytes[len..];
        }
        if !bytes.is_empty() {
            return Err(corrupt_key(format!(
                "row {row} has {} bytes after its {} keys",
                bytes.len(),
                self.keys.len()
            )));
        }
        Ok(())
    }
}

fn sort_fields(keys: &[BTreeKey], data_type: impl Fn(&DataType) -> DataType) -> Vec<SortField> {
    keys.iter()
        .map(|key| SortField::new_with_options(data_type(&key.data_type), key.options))
        .collect()
}

fn corrupt_key(message: String) -> Error {
    Error::corrupt_file_named("BTree arrow_row key", message)
}

fn invalid_utf8(position: usize, key: &BTreeKey, error: arrow_schema::ArrowError) -> Error {
    corrupt_key(format!(
        "key {position} '{}' ({}) holds invalid UTF-8: {error}",
        key.name, key.data_type
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use arrow_array::{
        BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array, Int8Array, Int16Array,
        Int32Array, Int64Array, LargeBinaryArray, TimestampNanosecondArray, UInt8Array,
        UInt16Array, UInt32Array, UInt64Array,
    };
    use arrow_ord::sort::{SortColumn, lexsort_to_indices};
    use arrow_schema::{Field, Fields, TimeUnit};
    use arrow_select::take::take;
    use lance_datagen::{ArrayGeneratorExt, RowCount, Seed, array, gen_array};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    use super::*;

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

    fn codec(keys: &[(DataType, SortOptions)]) -> ArrowRowV1 {
        ArrowRowV1::try_new(
            keys.iter()
                .enumerate()
                .map(|(position, (data_type, options))| BTreeKey {
                    name: format!("k{position}"),
                    data_type: data_type.clone(),
                    options: *options,
                })
                .collect(),
        )
        .unwrap()
    }

    struct RowBytes {
        columns: Vec<(SortOptions, ArrayRef)>,
        labelled_hex: Vec<(&'static str, &'static str)>,
    }

    fn column<T>(
        opts: SortOptions,
        to_array: fn(Vec<Option<T>>) -> ArrayRef,
        rows: Vec<(&'static str, Option<T>, &'static str)>,
    ) -> RowBytes {
        let (labelled_hex, values) = rows
            .into_iter()
            .map(|(label, value, hex)| ((label, hex), value))
            .unzip();
        RowBytes {
            columns: vec![(opts, to_array(values))],
            labelled_hex,
        }
    }

    fn golden_mismatches(case: &RowBytes) -> Vec<String> {
        let keys = case
            .columns
            .iter()
            .map(|(options, array)| (array.data_type().clone(), *options))
            .collect::<Vec<_>>();
        let codec = codec(&keys);
        let arrays = case
            .columns
            .iter()
            .map(|(_, array)| Arc::clone(array))
            .collect::<Vec<_>>();
        let types = keys
            .iter()
            .map(|(data_type, options)| format!("{data_type} {options:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        let frozen = "docs/src/format/index/scalar/btree.md, Encoding Stability, freezes these \
             bytes. If an Arrow upgrade changed them, vendor the version 1 encoder for";
        let mut mismatches = Vec::new();
        let batch = codec.encode(&arrays).unwrap();
        for (row, (label, expected)) in case.labelled_hex.iter().enumerate() {
            let slices = arrays
                .iter()
                .map(|array| array.slice(row, 1))
                .collect::<Vec<_>>();
            let alone = codec.encode(&slices).unwrap();
            for (context, actual) in [
                ("in a batch", hex::encode(batch.value(row))),
                ("alone", hex::encode(alone.value(0))),
            ] {
                if actual != *expected {
                    mismatches.push(format!(
                        "arrow_row version 1 bytes of {label} {context} as [{types}]:\n  \
                         expected {expected}\n  actual   {actual}\n{frozen} [{types}] and keep \
                         writing the expected bytes."
                    ));
                }
            }
        }
        let stored = BinaryArray::from_iter_values(
            case.labelled_hex
                .iter()
                .map(|(_, hex)| hex::decode(hex).unwrap()),
        );
        match codec.decode(&stored) {
            Ok(decoded) if decoded == arrays => {}
            decoded => mismatches.push(format!(
                "decoding the arrow_row version 1 bytes of [{types}] gave {decoded:?}, not \
                 {arrays:?}.\n{frozen} [{types}] and keep reading the expected bytes."
            )),
        }
        mismatches
    }

    #[rstest]
    #[case::boolean(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(BooleanArray::from(v)),
        vec![
            ("null", None, "0000"),
            ("false", Some(false), "0100"),
            ("true", Some(true), "0101"),
        ],
    ))]
    #[case::int8(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(Int8Array::from(v)),
        vec![
            ("null", None, "0000"),
            ("MIN", Some(i8::MIN), "0100"),
            ("-2", Some(-2), "017e"),
            ("MAX", Some(i8::MAX), "01ff"),
        ],
    ))]
    #[case::int16(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(Int16Array::from(v)),
        vec![
            ("null", None, "000000"),
            ("MIN", Some(i16::MIN), "010000"),
            ("-2", Some(-2), "017ffe"),
            ("MAX", Some(i16::MAX), "01ffff"),
        ],
    ))]
    #[case::int32(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(Int32Array::from(v)),
        vec![
            ("null", None, "0000000000"),
            ("MIN", Some(i32::MIN), "0100000000"),
            ("-2", Some(-2), "017ffffffe"),
            ("MAX", Some(i32::MAX), "01ffffffff"),
        ],
    ))]
    #[case::int64(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(Int64Array::from(v)),
        vec![
            ("null", None, "000000000000000000"),
            ("MIN", Some(i64::MIN), "010000000000000000"),
            ("-2", Some(-2), "017ffffffffffffffe"),
            ("MAX", Some(i64::MAX), "01ffffffffffffffff"),
        ],
    ))]
    #[case::uint8(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(UInt8Array::from(v)),
        vec![
            ("null", None, "0000"),
            ("MIN", Some(u8::MIN), "0100"),
            ("2^7", Some(1 << 7), "0180"),
            ("MAX", Some(u8::MAX), "01ff"),
        ],
    ))]
    #[case::uint16(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(UInt16Array::from(v)),
        vec![
            ("null", None, "000000"),
            ("MIN", Some(u16::MIN), "010000"),
            ("2^15", Some(1 << 15), "018000"),
            ("MAX", Some(u16::MAX), "01ffff"),
        ],
    ))]
    #[case::uint32(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(UInt32Array::from(v)),
        vec![
            ("null", None, "0000000000"),
            ("MIN", Some(u32::MIN), "0100000000"),
            ("2^31", Some(1 << 31), "0180000000"),
            ("MAX", Some(u32::MAX), "01ffffffff"),
        ],
    ))]
    #[case::uint64(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(UInt64Array::from(v)),
        vec![
            ("null", None, "000000000000000000"),
            ("MIN", Some(u64::MIN), "010000000000000000"),
            ("2^63", Some(1 << 63), "018000000000000000"),
            ("MAX", Some(u64::MAX), "01ffffffffffffffff"),
        ],
    ))]
    #[case::float32(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(Float32Array::from(v)),
        vec![
            ("null", None, "0000000000"),
            ("-MAX", Some(f32::from_bits(0xFF7F_FFFF)), "0100800000"),
            ("1.5", Some(f32::from_bits(0x3FC0_0000)), "01bfc00000"),
            ("+MAX", Some(f32::from_bits(0x7F7F_FFFF)), "01ff7fffff"),
        ],
    ))]
    #[case::float32_special_values(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(Float32Array::from(v)),
        vec![
            ("null", None, "0000000000"),
            ("-NaN(quiet)", Some(f32::from_bits(0xFFC0_0000)), "01003fffff"),
            ("-inf", Some(f32::from_bits(0xFF80_0000)), "01007fffff"),
            ("-0.0", Some(f32::from_bits(0x8000_0000)), "017fffffff"),
            ("+0.0", Some(f32::from_bits(0x0000_0000)), "0180000000"),
            ("+inf", Some(f32::from_bits(0x7F80_0000)), "01ff800000"),
            ("+NaN(quiet)", Some(f32::from_bits(0x7FC0_0000)), "01ffc00000"),
        ],
    ))]
    #[case::float64(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(Float64Array::from(v)),
        vec![
            ("null", None, "000000000000000000"),
            ("-NaN(quiet)", Some(f64::from_bits(0xFFF8_0000_0000_0000)), "010007ffffffffffff"),
            ("-inf", Some(f64::from_bits(0xFFF0_0000_0000_0000)), "01000fffffffffffff"),
            ("-MAX", Some(f64::from_bits(0xFFEF_FFFF_FFFF_FFFF)), "010010000000000000"),
            ("-0.0", Some(f64::from_bits(0x8000_0000_0000_0000)), "017fffffffffffffff"),
            ("+0.0", Some(f64::from_bits(0x0000_0000_0000_0000)), "018000000000000000"),
            ("1.5", Some(f64::from_bits(0x3FF8_0000_0000_0000)), "01bff8000000000000"),
            ("+MAX", Some(f64::from_bits(0x7FEF_FFFF_FFFF_FFFF)), "01ffefffffffffffff"),
            ("+inf", Some(f64::from_bits(0x7FF0_0000_0000_0000)), "01fff0000000000000"),
            ("+NaN(quiet)", Some(f64::from_bits(0x7FF8_0000_0000_0000)), "01fff8000000000000"),
            ("+NaN(payload 1)", Some(f64::from_bits(0x7FF8_0000_0000_0001)), "01fff8000000000001"),
        ],
    ))]
    #[case::date32(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(Date32Array::from(v)),
        vec![
            ("null", None, "0000000000"),
            ("MIN", Some(i32::MIN), "0100000000"),
            ("0 (epoch)", Some(0), "0180000000"),
            ("MAX", Some(i32::MAX), "01ffffffff"),
        ],
    ))]
    #[case::date64(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(Date64Array::from(v)),
        vec![
            ("null", None, "000000000000000000"),
            ("MIN", Some(i64::MIN), "010000000000000000"),
            ("0 (epoch)", Some(0), "018000000000000000"),
            ("MAX", Some(i64::MAX), "01ffffffffffffffff"),
        ],
    ))]
    #[case::timestamp_ns(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(TimestampNanosecondArray::from(v)),
        vec![
            ("null", None, "000000000000000000"),
            ("MIN", Some(i64::MIN), "010000000000000000"),
            ("0 (epoch)", Some(0), "018000000000000000"),
            ("MAX", Some(i64::MAX), "01ffffffffffffffff"),
        ],
    ))]
    #[case::timestamp_ns_utc(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(TimestampNanosecondArray::from(v).with_timezone("UTC")),
        vec![
            ("null", None, "000000000000000000"),
            ("MIN", Some(i64::MIN), "010000000000000000"),
            ("0 (epoch)", Some(0), "018000000000000000"),
            ("MAX", Some(i64::MAX), "01ffffffffffffffff"),
        ],
    ))]
    #[case::utf8(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(StringArray::from(v)),
        vec![
            ("null", None, "00"),
            ("\"\"", Some(""), "01"),
            ("\"ab\"", Some("ab"), "02616200000000000002"),
            ("\"\\u{10FFFF}\"", Some("\u{10FFFF}"), "02f48fbfbf0000000004"),
        ],
    ))]
    #[case::utf8_block_boundaries(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(StringArray::from(v)),
        vec![
            (
                "\"0-9a-v\" (32 bytes)",
                Some("0123456789abcdefghijklmnopqrstuv"),
                concat!(
                    "02",
                    "3031323334353637ff",
                    "3839616263646566ff",
                    "6768696a6b6c6d6eff",
                    "6f7071727374757608",
                ),
            ),
            (
                "\"0-9a-w\" (33 bytes)",
                Some("0123456789abcdefghijklmnopqrstuvw"),
                concat!(
                    "02",
                    "3031323334353637ff",
                    "3839616263646566ff",
                    "6768696a6b6c6d6eff",
                    "6f70717273747576ff",
                    "77000000000000000000000000000000",
                    "00000000000000000000000000000000",
                    "01",
                ),
            ),
        ],
    ))]
    #[case::large_utf8(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(LargeStringArray::from(v)),
        vec![
            ("null", None, "00"),
            ("\"\"", Some(""), "01"),
            ("\"ab\"", Some("ab"), "02616200000000000002"),
            ("\"\\u{10FFFF}\"", Some("\u{10FFFF}"), "02f48fbfbf0000000004"),
        ],
    ))]
    #[case::binary(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(BinaryArray::from(v)),
        vec![
            ("null", None, "00"),
            ("[]", Some(b"".as_slice()), "01"),
            ("[00 FF]", Some(b"\x00\xff".as_slice()), "0200ff00000000000002"),
            ("[FF]", Some(b"\xff".as_slice()), "02ff0000000000000001"),
        ],
    ))]
    #[case::large_binary(column(
        ASC_NULLS_FIRST,
        |v| Arc::new(LargeBinaryArray::from(v)),
        vec![
            ("null", None, "00"),
            ("[]", Some(b"".as_slice()), "01"),
            ("[00 FF]", Some(b"\x00\xff".as_slice()), "0200ff00000000000002"),
            ("[FF]", Some(b"\xff".as_slice()), "02ff0000000000000001"),
        ],
    ))]
    #[case::int32_asc_nulls_last(column(
        ASC_NULLS_LAST,
        |v| Arc::new(Int32Array::from(v)),
        vec![("null", None, "ff00000000"), ("-2", Some(-2), "017ffffffe")],
    ))]
    #[case::int32_desc_nulls_first(column(
        DESC_NULLS_FIRST,
        |v| Arc::new(Int32Array::from(v)),
        vec![("null", None, "0000000000"), ("-2", Some(-2), "0180000001")],
    ))]
    #[case::int32_desc_nulls_last(column(
        DESC_NULLS_LAST,
        |v| Arc::new(Int32Array::from(v)),
        vec![("null", None, "ff00000000"), ("-2", Some(-2), "0180000001")],
    ))]
    #[case::utf8_asc_nulls_last(column(
        ASC_NULLS_LAST,
        |v| Arc::new(StringArray::from(v)),
        vec![
            ("null", None, "ff"),
            ("\"\"", Some(""), "01"),
            ("\"ab\"", Some("ab"), "02616200000000000002"),
        ],
    ))]
    #[case::utf8_desc_nulls_first(column(
        DESC_NULLS_FIRST,
        |v| Arc::new(StringArray::from(v)),
        vec![
            ("null", None, "00"),
            ("\"\"", Some(""), "fe"),
            ("\"ab\"", Some("ab"), "fd9e9dfffffffffffffd"),
            (
                "\"012345678\" (9 bytes)",
                Some("012345678"),
                concat!("fd", "cfcecdcccbcac9c800", "c7fffffffffffffffe"),
            ),
        ],
    ))]
    #[case::utf8_desc_nulls_last(column(
        DESC_NULLS_LAST,
        |v| Arc::new(StringArray::from(v)),
        vec![
            ("null", None, "ff"),
            ("\"\"", Some(""), "fe"),
            ("\"ab\"", Some("ab"), "fd9e9dfffffffffffffd"),
            (
                "\"012345678\" (9 bytes)",
                Some("012345678"),
                concat!("fd", "cfcecdcccbcac9c800", "c7fffffffffffffffe"),
            ),
        ],
    ))]
    #[case::boolean_desc_nulls_last(column(
        DESC_NULLS_LAST,
        |v| Arc::new(BooleanArray::from(v)),
        vec![
            ("null", None, "ff00"),
            ("false", Some(false), "01ff"),
            ("true", Some(true), "01fe"),
        ],
    ))]
    #[case::composite_int32_asc_utf8_desc(RowBytes {
        columns: vec![
            (ASC_NULLS_FIRST, Arc::new(Int32Array::from(vec![Some(-1), None])) as ArrayRef),
            (DESC_NULLS_LAST, Arc::new(StringArray::from(vec![Some("ab"), None])) as ArrayRef),
        ],
        labelled_hex: vec![
            ("(-1, \"ab\")", concat!("017fffffff", "fd9e9dfffffffffffffd")),
            ("(null, null)", concat!("0000000000", "ff")),
        ],
    })]
    fn golden_bytes_match_arrow_row_v1(#[case] case: RowBytes) {
        let mismatches = golden_mismatches(&case);
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    }

    #[test]
    fn golden_mismatch_cites_encoding_stability() {
        let case = column(
            ASC_NULLS_FIRST,
            |v| Arc::new(Int32Array::from(v)),
            vec![("-2", Some(-2), "017ffffffd")],
        );
        let mismatches = golden_mismatches(&case);
        assert_eq!(mismatches.len(), 3, "{mismatches:#?}");
        for mismatch in &mismatches {
            assert!(
                mismatch.contains("btree.md, Encoding Stability")
                    && mismatch.contains("vendor the version 1 encoder for [Int32 "),
                "{mismatch}"
            );
        }
        assert!(mismatches[0].contains("expected 017ffffffd\n  actual   017ffffffe"));
    }

    /// The hex strings that each `#[case::name(..)]` of the rstest function
    /// `function` in `source` expects, in order: the string literals of
    /// lowercase hex digits that end a tuple or a `concat!`.
    fn golden_hex(source: &str, function: &str) -> BTreeMap<String, Vec<String>> {
        let end = source
            .find(&format!("fn {function}("))
            .unwrap_or_else(|| panic!("no fn {function}"));
        let start = source[..end].rfind("#[rstest]").unwrap();
        let text = source[start..end]
            .split_whitespace()
            .collect::<String>()
            .replace("\",\"", "");
        text.split("#[case::")
            .skip(1)
            .map(|case| {
                let (name, body) = case.split_once('(').unwrap();
                let hex = body
                    .match_indices("\")")
                    .filter_map(|(close, _)| {
                        let open = body[..close].rfind('"')?;
                        let literal = &body[open + 1..close];
                        let ends_row =
                            body[..open].ends_with(',') || body[..open].ends_with("concat!(");
                        let is_hex = !literal.is_empty()
                            && literal
                                .bytes()
                                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
                        (ends_row && is_hex).then(|| literal.to_string())
                    })
                    .collect();
                (name.to_string(), hex)
            })
            .collect()
    }

    #[test]
    fn goldens_match_the_arrow_ordering_fixture() {
        let fixture = golden_hex(
            include_str!("../../../../lance/tests/arrow_ordering.rs"),
            "row_converter_bytes_match_baseline",
        );
        let here = golden_hex(
            include_str!("arrow_row.rs"),
            "golden_bytes_match_arrow_row_v1",
        );
        assert_eq!(here["utf8"].len(), 4, "{here:#?}");
        assert_eq!(
            here["utf8_desc_nulls_first"][3],
            "fdcfcecdcccbcac9c800c7fffffffffffffffe"
        );
        assert_eq!(
            here["composite_int32_asc_utf8_desc"],
            ["017ffffffffd9e9dfffffffffffffd", "0000000000ff"]
        );
        assert_eq!(
            here, fixture,
            "the goldens here and in row_converter_bytes_match_baseline of \
             rust/lance/tests/arrow_ordering.rs must name the same cases and hold the same \
             bytes, because both pin arrow_row version 1 (btree.md, Encoding Stability)"
        );
    }

    const OPTIONS: [SortOptions; 4] = [
        ASC_NULLS_FIRST,
        ASC_NULLS_LAST,
        DESC_NULLS_FIRST,
        DESC_NULLS_LAST,
    ];

    const DISTINCT_VALUES: usize = 16;
    const LENGTHS_AROUND_BLOCK_BOUNDARIES: [usize; 10] = [0, 1, 7, 8, 9, 31, 32, 33, 40, 70];
    const ONE_TO_FOUR_BYTE_CHARS: [char; 4] = ['a', 'z', 'é', '\u{10FFFF}'];

    fn random_column(data_type: &DataType, seed: u64, rows: usize) -> ArrayRef {
        let mut rng = SmallRng::seed_from_u64(seed);
        let pool = (0..DISTINCT_VALUES)
            .map(|_| {
                let length = LENGTHS_AROUND_BLOCK_BOUNDARIES
                    [rng.random_range(0..LENGTHS_AROUND_BLOCK_BOUNDARIES.len())];
                (0..length)
                    .map(|_| ONE_TO_FOUR_BYTE_CHARS[rng.random_range(0..4)])
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        let strings = (0..rows)
            .map(|_| {
                (!rng.random_bool(0.2)).then(|| pool[rng.random_range(0..DISTINCT_VALUES)].as_str())
            })
            .collect::<Vec<_>>();
        let bytes = || strings.iter().map(|value| value.map(str::as_bytes));
        match data_type {
            DataType::Utf8 => Arc::new(StringArray::from(strings)),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(strings)),
            DataType::Binary => Arc::new(BinaryArray::from_iter(bytes())),
            DataType::LargeBinary => Arc::new(LargeBinaryArray::from_iter(bytes())),
            _ => gen_array(
                array::low_cardinality(array::rand_type(data_type), DISTINCT_VALUES)
                    .with_random_nulls(0.2),
            )
            .with_seed(Seed::from(seed))
            .into_array_rows(RowCount::from(rows as u64))
            .unwrap(),
        }
    }

    fn assert_round_trip_and_order(codec: &ArrowRowV1, columns: &[ArrayRef]) {
        let encoded = codec.encode(columns).unwrap();
        assert_eq!(codec.decode(&encoded).unwrap(), columns);

        let mut by_bytes = (0..encoded.len() as u32).collect::<Vec<_>>();
        by_bytes.sort_by(|a, b| encoded.value(*a as usize).cmp(encoded.value(*b as usize)));
        let by_bytes = UInt32Array::from(by_bytes);
        let sort_columns = columns
            .iter()
            .zip(codec.keys())
            .map(|(values, key)| SortColumn {
                values: Arc::clone(values),
                options: Some(key.options),
            })
            .collect::<Vec<_>>();
        let by_lexsort = lexsort_to_indices(&sort_columns, None).unwrap();
        for (position, values) in columns.iter().enumerate() {
            assert_eq!(
                take(values.as_ref(), &by_bytes, None).unwrap().to_data(),
                take(values.as_ref(), &by_lexsort, None).unwrap().to_data(),
                "key {position} of {:?}",
                codec.keys()
            );
        }
    }

    #[rstest]
    fn single_keys_round_trip_and_sort_as_lexsort(
        #[values(
            DataType::Boolean,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
            DataType::Date32,
            DataType::Date64,
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("+05:00".into())),
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary
        )]
        data_type: DataType,
        #[values(ASC_NULLS_FIRST, ASC_NULLS_LAST, DESC_NULLS_FIRST, DESC_NULLS_LAST)]
        options: SortOptions,
    ) {
        let codec = codec(&[(data_type.clone(), options)]);
        assert_round_trip_and_order(&codec, &[random_column(&data_type, 7, 300)]);
    }

    fn composite_columns(seed: u64) -> Vec<ArrayRef> {
        [
            DataType::Int64,
            DataType::Utf8,
            DataType::Timestamp(TimeUnit::Microsecond, None),
        ]
        .iter()
        .enumerate()
        .map(|(position, data_type)| random_column(data_type, seed + position as u64, 300))
        .collect()
    }

    #[rstest]
    fn composite_keys_round_trip_and_sort_as_lexsort(
        #[values(ASC_NULLS_FIRST, ASC_NULLS_LAST, DESC_NULLS_FIRST, DESC_NULLS_LAST)]
        first: SortOptions,
        #[values(ASC_NULLS_FIRST, ASC_NULLS_LAST, DESC_NULLS_FIRST, DESC_NULLS_LAST)]
        second: SortOptions,
    ) {
        let columns = composite_columns(11);
        let types = columns.iter().map(|c| c.data_type().clone());
        let options = [first, second, DESC_NULLS_LAST];
        let codec = codec(&types.zip(options).collect::<Vec<_>>());
        assert_round_trip_and_order(&codec, &columns);
    }

    #[test]
    fn decode_handles_empty_sliced_and_megabyte_keys() {
        let codec = codec(&[
            (DataType::Utf8, DESC_NULLS_FIRST),
            (DataType::LargeBinary, ASC_NULLS_LAST),
        ]);
        let megabyte = "é".repeat(1 << 19);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some(megabyte.as_str()),
                None,
            ])),
            Arc::new(LargeBinaryArray::from(vec![
                Some(megabyte.as_bytes()),
                None,
                Some(b"".as_slice()),
            ])),
        ];
        let encoded = codec.encode(&columns).unwrap();
        assert_eq!(codec.decode(&encoded).unwrap(), columns);

        let sliced = codec.decode(&encoded.slice(1, 2)).unwrap();
        let expected = columns.iter().map(|c| c.slice(1, 2)).collect::<Vec<_>>();
        assert_eq!(sliced, expected);

        let empty = codec.decode(&encoded.slice(0, 0)).unwrap();
        assert_eq!(empty.len(), 2);
        assert!(empty.iter().all(|column| column.is_empty()));
        assert_eq!(empty[0].data_type(), &DataType::Utf8);
    }

    #[test]
    fn prefix_encodings_are_byte_prefixes_of_the_tuple() {
        let columns = composite_columns(5);
        for options in OPTIONS {
            let keys = columns
                .iter()
                .map(|c| (c.data_type().clone(), options))
                .collect::<Vec<_>>();
            let codec = codec(&keys);
            let encoded = codec.encode(&columns).unwrap();
            for row in 0..encoded.len() {
                let tuple = columns
                    .iter()
                    .map(|c| ScalarValue::try_from_array(c, row).unwrap())
                    .collect::<Vec<_>>();
                for k in 0..=tuple.len() {
                    let prefix = codec.encode_prefix(&tuple[..k]).unwrap();
                    assert!(
                        encoded.value(row).starts_with(&prefix),
                        "row {row} prefix {k}: {} is not a prefix of {}",
                        hex::encode(&prefix),
                        hex::encode(encoded.value(row))
                    );
                }
                assert_eq!(codec.encode_prefix(&tuple).unwrap(), encoded.value(row));
            }
        }
    }

    #[rstest]
    #[case::empty(&[], None)]
    #[case::all_ff(&[0xFF, 0xFF], None)]
    #[case::increments_the_last_byte(&[0x01, 0x02], Some(&[0x01, 0x03][..]))]
    #[case::carries_past_ff(&[0x01, 0xFF], Some(&[0x02][..]))]
    #[case::carries_past_several_ff(&[0x00, 0xFF, 0xFF], Some(&[0x01][..]))]
    #[case::zero(&[0x00], Some(&[0x01][..]))]
    fn next_prefix_bounds_the_prefix_group(#[case] prefix: &[u8], #[case] next: Option<&[u8]>) {
        assert_eq!(next_prefix(prefix).as_deref(), next);
    }

    #[test]
    fn next_prefix_range_holds_exactly_the_strings_with_the_prefix() {
        let alphabet = [0x00, 0x01, 0xFE, 0xFF];
        let strings = (0..=3)
            .flat_map(|len| {
                (0..alphabet.len().pow(len)).map(move |n| {
                    (0..len)
                        .map(|i| alphabet[n / alphabet.len().pow(i) % alphabet.len()])
                        .collect::<Vec<u8>>()
                })
            })
            .collect::<Vec<_>>();
        for prefix in strings.iter().filter(|s| s.len() <= 2) {
            let end = next_prefix(prefix);
            for string in &strings {
                let in_range = string >= prefix && end.as_ref().is_none_or(|end| string < end);
                assert_eq!(
                    in_range,
                    string.starts_with(prefix),
                    "{prefix:02x?} {string:02x?} {end:02x?}"
                );
            }
        }
    }

    #[rstest]
    #[case::truncated_fixed(DataType::Int32, ASC_NULLS_FIRST, "017fff", "key 0 'k0'")]
    #[case::fixed_marker(DataType::Int32, ASC_NULLS_FIRST, "0200000000", "key 0 'k0'")]
    #[case::fixed_null_padding(DataType::Int32, ASC_NULLS_LAST, "ff00000001", "key 0 'k0'")]
    #[case::boolean_value(DataType::Boolean, DESC_NULLS_FIRST, "01fd", "key 0 'k0'")]
    #[case::variable_marker(DataType::Utf8, ASC_NULLS_FIRST, "03", "key 0 'k0'")]
    #[case::truncated_block(DataType::Binary, ASC_NULLS_FIRST, "026162", "key 0 'k0'")]
    #[case::block_length_zero(
        DataType::Binary,
        ASC_NULLS_FIRST,
        "02616200000000000000",
        "key 0 'k0'"
    )]
    #[case::block_length_over_width(
        DataType::Binary,
        ASC_NULLS_FIRST,
        "02616200000000000009",
        "key 0 'k0'"
    )]
    #[case::block_padding(
        DataType::Binary,
        DESC_NULLS_FIRST,
        "fd9e9dfffffffffffefd",
        "key 0 'k0'"
    )]
    #[case::invalid_utf8(
        DataType::Utf8,
        ASC_NULLS_FIRST,
        "02ff0000000000000001",
        "invalid UTF-8"
    )]
    #[case::trailing_bytes(DataType::Int32, ASC_NULLS_FIRST, "017ffffffe00", "1 bytes after")]
    fn decode_rejects_noncanonical_keys(
        #[case] data_type: DataType,
        #[case] options: SortOptions,
        #[case] hex: &str,
        #[case] message: &str,
    ) {
        let codec = codec(&[(data_type, options)]);
        let keys = BinaryArray::from_iter_values([hex::decode(hex).unwrap()]);
        let error = codec.decode(&keys).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
    }

    #[test]
    fn decode_rejects_a_null_row() {
        let codec = codec(&[(DataType::Int32, ASC_NULLS_FIRST)]);
        let keys = BinaryArray::from(vec![Some(&[0x01, 0, 0, 0, 0][..]), None]);
        let error = codec.decode(&keys).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("row 1 is null"), "{error}");
    }

    #[test]
    fn encode_rejects_columns_that_do_not_match_the_keys() {
        let codec = codec(&[
            (DataType::Int32, ASC_NULLS_FIRST),
            (DataType::Utf8, ASC_NULLS_FIRST),
        ]);
        let ints: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        let error = codec.encode(std::slice::from_ref(&ints)).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("2 keys, but got 1 columns"),
            "{error}"
        );

        let error = codec.encode(&[Arc::clone(&ints), ints]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("key 1 'k1' has type Utf8, but its column has type Int32"),
            "{error}"
        );

        let error = codec
            .encode_prefix(&[ScalarValue::Int64(Some(1))])
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("key 0 'k0' has type Int32, but its column has type Int64"),
            "{error}"
        );
        let values = [
            ScalarValue::Int32(None),
            ScalarValue::Utf8(None),
            ScalarValue::Utf8(None),
        ];
        let error = codec.encode_prefix(&values).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("3 values is longer than the 2 keys"),
            "{error}"
        );
    }

    #[rstest]
    #[case::decimal(DataType::Decimal128(10, 2))]
    #[case::float16(DataType::Float16)]
    #[case::utf8_view(DataType::Utf8View)]
    #[case::fixed_size_binary(DataType::FixedSizeBinary(4))]
    #[case::list(DataType::List(Arc::new(Field::new("item", DataType::Int32, true))))]
    #[case::structure(DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int32, true)])))]
    #[case::dictionary(DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)))]
    fn try_new_rejects_types_the_encoding_does_not_admit(#[case] data_type: DataType) {
        let keys = vec![
            BTreeKey {
                name: "a".to_string(),
                data_type: DataType::Int32,
                options: ASC_NULLS_FIRST,
            },
            BTreeKey {
                name: "b".to_string(),
                data_type: data_type.clone(),
                options: ASC_NULLS_FIRST,
            },
        ];
        let error = ArrowRowV1::try_new(keys).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(
            error.to_string().contains(&format!(
                "key 1 'b' has type {data_type}, which the arrow_row"
            )),
            "{error}"
        );

        let error = ArrowRowV1::try_new(vec![]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }
}
