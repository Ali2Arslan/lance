// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{
    RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
    cast::AsArray,
    types::{UInt32Type, UInt64Type},
};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion::{physical_plan::SendableRecordBatchStream, scalar::ScalarValue};
use futures::{FutureExt, TryStreamExt};
use lance::dataset::WriteParams;
use lance::dataset::scanner::{ColumnOrdering, OrderedIndexMode, Scanner};
use lance::index::DatasetIndexExt;
use lance::{Dataset, io::ObjectStore};
use lance_core::ROW_ADDR;
use lance_core::cache::LanceCache;
use lance_core::utils::tempfile::TempStrDir;
use lance_datafusion::utils::reader_to_stream;
use lance_datagen::{BatchCount, DEFAULT_SEED, RowCount, Seed, array, gen_batch};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::{
    BuiltinIndexType, IndexStore, SargableQuery, ScalarIndex, ScalarIndexParams, SearchResult,
    btree::{DEFAULT_BTREE_BATCH_SIZE, train_btree_index},
    lance_format::LanceIndexStore,
    registry::{ScalarIndexPlugin, VALUE_COLUMN_NAME},
};
use lance_index::{metrics::NoOpMetricsCollector, scalar::btree::BTreeIndexPlugin};
use lance_select::RowSetOps;
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

struct BenchmarkFixture {
    _datadir: TempStrDir,
    index_store: Arc<dyn IndexStore>,
    baseline_dataset: Arc<Dataset>,
}

fn test_data(value_column: &str, row_column: &str) -> impl RecordBatchReader {
    gen_batch()
        .col(value_column, array::step::<UInt32Type>())
        .col(row_column, array::step::<UInt64Type>())
        .into_reader_rows(RowCount::from(1024), BatchCount::from(100 * 1024))
}

fn test_data_stream() -> SendableRecordBatchStream {
    reader_to_stream(Box::new(test_data(VALUE_COLUMN_NAME, ROW_ADDR)))
}

impl BenchmarkFixture {
    fn test_store(tempdir: &TempStrDir) -> Arc<dyn IndexStore> {
        let test_path = tempdir.as_str();
        let (object_store, test_path) = ObjectStore::from_uri(test_path)
            .now_or_never()
            .unwrap()
            .unwrap();
        Arc::new(LanceIndexStore::new(
            object_store,
            test_path,
            Arc::new(LanceCache::no_cache()),
        ))
    }

    async fn write_baseline_data(tempdir: &TempStrDir) -> Arc<Dataset> {
        let test_path = tempdir.as_str();
        Arc::new(
            Dataset::write(test_data("values", "row_ids"), test_path, None)
                .await
                .unwrap(),
        )
    }

    async fn train_scalar_index(index_store: &Arc<dyn IndexStore>) {
        train_btree_index(
            test_data_stream(),
            index_store.as_ref(),
            DEFAULT_BTREE_BATCH_SIZE,
            None,
            None,
        )
        .await
        .unwrap();
    }

    async fn open() -> Self {
        let tempdir = TempStrDir::default();
        let index_store = Self::test_store(&tempdir);
        let baseline_dataset = Self::write_baseline_data(&tempdir).await;
        Self::train_scalar_index(&index_store).await;

        Self {
            _datadir: tempdir,
            index_store,
            baseline_dataset,
        }
    }
}

async fn baseline_equality_search(fixture: &BenchmarkFixture) {
    let mut stream = fixture
        .baseline_dataset
        .scan()
        .filter("values == 10000")
        .unwrap()
        .with_row_id()
        .try_into_stream()
        .await
        .unwrap();
    let mut num_rows = 0;
    while let Some(batch) = stream.try_next().await.unwrap() {
        num_rows += batch.num_rows();
    }
    assert_eq!(num_rows, 1);
}

async fn warm_indexed_equality_search(index: &dyn ScalarIndex) {
    let result = index
        .search(
            &SargableQuery::Equals(ScalarValue::UInt32(Some(10000))),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
    let SearchResult::Exact(row_ids) = result else {
        panic!("Expected exact results")
    };
    assert_eq!(row_ids.true_rows().len(), Some(1));
}

async fn baseline_inequality_search(fixture: &BenchmarkFixture) {
    let mut stream = fixture
        .baseline_dataset
        .scan()
        .filter("values >= 50000000")
        .unwrap()
        .with_row_id()
        .try_into_stream()
        .await
        .unwrap();
    let mut num_rows = 0;
    while let Some(batch) = stream.try_next().await.unwrap() {
        num_rows += batch.num_rows();
    }
    // 100Mi - 50M = 54,857,600
    assert_eq!(num_rows, 54857600);
}

async fn warm_indexed_inequality_search(index: &dyn ScalarIndex) {
    let result = index
        .search(
            &SargableQuery::Range(
                std::ops::Bound::Included(ScalarValue::UInt32(Some(50_000_000))),
                std::ops::Bound::Unbounded,
            ),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
    let SearchResult::Exact(row_ids) = result else {
        panic!("Expected exact results")
    };

    // 100Mi - 50M = 54,857,600
    assert_eq!(row_ids.true_rows().len(), Some(54857600));
}

async fn warm_indexed_isin_search(index: &dyn ScalarIndex) {
    let result = index
        .search(
            &SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(10000)),
                ScalarValue::UInt32(Some(50000000)),
                ScalarValue::UInt32(Some(150000000)), // Not found
                ScalarValue::UInt32(Some(287123)),
            ]),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
    let SearchResult::Exact(row_ids) = result else {
        panic!("Expected exact results")
    };

    // Only 3 because 150M is not in dataset
    assert_eq!(row_ids.true_rows().len(), Some(3));
}

fn status(id: u64) -> &'static str {
    match id {
        id if id % 2 == 0 => "50%",
        id if id % 100 == 1 => "1%",
        id if id % 10_000 == 3 => "0.01%",
        _ => "other",
    }
}

fn age(ts: u64) -> &'static str {
    match ts {
        ts if ts < u64::MAX / 10_000 => "0-0.01%",
        ts if ts < u64::MAX / 10_000 * 3 => "0.01-0.03%",
        ts if ts < u64::MAX / 10_000 * 6 => "0.03-0.06%",
        ts if ts < u64::MAX / 100 => "0.06-1%",
        ts if ts < u64::MAX / 2 => "1-50%",
        _ => "50-100%",
    }
}

// The age of an appended row starts with this, so that a mask of the oldest indexed rows selects
// none of the appended ones unless it names their ages too.
const APPENDED_AGE: &str = "appended ";

fn with_mask_columns(
    rows: impl RecordBatchReader,
    age_prefix: &'static str,
) -> impl RecordBatchReader {
    let mut fields = rows.schema().fields().to_vec();
    fields.push(Arc::new(Field::new("status", DataType::Utf8, false)));
    fields.push(Arc::new(Field::new("age", DataType::Utf8, false)));
    let schema = Arc::new(Schema::new(fields));
    let batches = rows.map({
        let schema = schema.clone();
        move |batch| {
            let batch = batch?;
            let ids = batch["id"].as_primitive::<UInt64Type>().values();
            let ts = batch["ts"].as_primitive::<UInt64Type>().values();
            let mut columns = batch.columns().to_vec();
            columns.push(Arc::new(StringArray::from_iter_values(
                ids.iter().copied().map(status),
            )));
            columns.push(Arc::new(StringArray::from_iter_values(
                ts.iter().map(|&ts| format!("{age_prefix}{}", age(ts))),
            )));
            RecordBatch::try_new(schema.clone(), columns)
        }
    });
    RecordBatchIterator::new(batches, schema)
}

async fn ordered_limit_dataset(tempdir: &TempStrDir, segments: u32) -> Dataset {
    const BATCHES: u32 = 100;
    assert_eq!(
        BATCHES % segments,
        0,
        "{segments} segments must split {BATCHES} batches evenly"
    );
    let batches_per_segment = BATCHES / segments;
    let data = |segment: u32| {
        with_mask_columns(
            gen_batch()
                .with_seed(Seed::from(DEFAULT_SEED.0 + u64::from(segment)))
                .col("ts", array::rand::<UInt64Type>())
                .col("id", array::step::<UInt64Type>())
                .into_reader_rows(
                    RowCount::from(100_000),
                    BatchCount::from(batches_per_segment),
                ),
            "",
        )
    };
    let params = WriteParams {
        max_rows_per_file: 1_000_000,
        ..Default::default()
    };
    let mut dataset = Dataset::write(data(0), tempdir.as_str(), Some(params.clone()))
        .await
        .unwrap();
    dataset
        .create_index(
            &["ts"],
            IndexType::BTree,
            Some("ts_idx".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();
    for segment in 1..segments {
        dataset
            .append(data(segment), Some(params.clone()))
            .await
            .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
    }
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
    assert_eq!(
        dataset.load_indices_by_name("ts_idx").await.unwrap().len(),
        segments as usize
    );
    dataset
}

/// Appends rows that every index misses.
async fn append_unindexed(dataset: &mut Dataset, batches: u32, seed: u64) {
    let data = with_mask_columns(
        gen_batch()
            .with_seed(Seed::from(seed))
            .col("ts", array::rand::<UInt64Type>())
            .col("id", array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(100_000), BatchCount::from(batches)),
        APPENDED_AGE,
    );
    let params = WriteParams {
        max_rows_per_file: 1_000_000,
        ..Default::default()
    };
    dataset.append(data, Some(params)).await.unwrap();
}

/// Indexes the appended rows' ages, so that a mask on `age` is exact over
/// every fragment.
async fn index_ages(dataset: &mut Dataset) {
    dataset
        .optimize_indices(&OptimizeOptions::append().index_names(vec!["age_idx".into()]))
        .await
        .unwrap();
}

/// The `age` filters of the anti-correlated masks at an unindexed share: the
/// oldest 0.01% and 1% of the indexed rows, which `ORDER BY ts DESC` reaches
/// last, alone and with the appended rows of the same ages.
fn oldest_masks() -> Vec<(String, String)> {
    let ages = ["0-0.01%", "0.01-0.03%", "0.03-0.06%", "0.06-1%"];
    [("0.01%", &ages[..1]), ("1%", &ages[..])]
        .into_iter()
        .flat_map(|(share, ages)| {
            let indexed = ages.iter().map(|age| format!("'{age}'"));
            let appended = ages.iter().map(|age| format!("'{APPENDED_AGE}{age}'"));
            let filter = |ages: Vec<String>| format!("age IN ({})", ages.join(", "));
            [
                (format!("oldest_{share}"), filter(indexed.clone().collect())),
                (
                    format!("oldest_{share}_and_appended"),
                    filter(indexed.chain(appended).collect()),
                ),
            ]
        })
        .collect()
}

async fn ordered_limit(dataset: &Dataset, configure: impl Fn(&mut Scanner)) {
    let mut scanner = dataset.scan();
    scanner
        .project(&["id"])
        .unwrap()
        .order_by(Some(vec![ColumnOrdering::desc_nulls_last("ts".into())]))
        .unwrap()
        .limit(Some(100), None)
        .unwrap();
    configure(&mut scanner);
    assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 100);
}

fn bench_baseline(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let fixture = rt.block_on(BenchmarkFixture::open());

    c.bench_function("baseline_equality", |b| {
        b.iter(|| rt.block_on(baseline_equality_search(&fixture)));
    });

    c.bench_function("baseline_inequality", |b| {
        b.iter(|| rt.block_on(baseline_inequality_search(&fixture)));
    });
}

fn bench_warm_indexed(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(BenchmarkFixture::open());
    let details =
        prost_types::Any::from_msg(&lance_index::pbold::BTreeIndexDetails::default()).unwrap();

    let index = rt
        .block_on(BTreeIndexPlugin.load_index(
            fixture.index_store.clone(),
            &details,
            lance_index::scalar::btree::BTREE_INDEX_VERSION,
            None,
            &LanceCache::no_cache(),
        ))
        .unwrap();

    c.bench_function("windexed_equality", |b| {
        b.iter(|| rt.block_on(warm_indexed_equality_search(index.as_ref())))
    });

    c.bench_function("windexed_inequality", |b| {
        b.iter(|| rt.block_on(warm_indexed_inequality_search(index.as_ref())))
    });

    c.bench_function("windexed_is_in", |b| {
        b.iter(|| rt.block_on(warm_indexed_isin_search(index.as_ref())))
    });
}

fn bench_ordered_limit(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("ordered_limit");
    for segments in [1, 4] {
        let tempdir = TempStrDir::default();
        let dataset = rt.block_on(ordered_limit_dataset(&tempdir, segments));
        group.bench_with_input(
            BenchmarkId::new("ordered_index_auto", segments),
            &dataset,
            |b, dataset| {
                b.iter(|| rt.block_on(ordered_limit(dataset, |_| {})));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ordered_index_off", segments),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    rt.block_on(ordered_limit(dataset, |scanner| {
                        scanner.ordered_index(OrderedIndexMode::Off);
                    }))
                });
            },
        );
        for (selectivity, residual) in [("50%", "id % 2 = 0"), ("0.01%", "id % 10000 = 0")] {
            group.bench_with_input(
                BenchmarkId::new(format!("residual_{selectivity}"), segments),
                &dataset,
                |b, dataset| {
                    b.iter(|| {
                        rt.block_on(ordered_limit(dataset, |scanner| {
                            scanner.filter(residual).unwrap();
                        }))
                    });
                },
            );
        }
        for selectivity in ["50%", "1%", "0.01%"] {
            let filter = format!("status = '{selectivity}'");
            group.bench_with_input(
                BenchmarkId::new(format!("mask_{selectivity}"), segments),
                &dataset,
                |b, dataset| {
                    b.iter(|| {
                        rt.block_on(ordered_limit(dataset, |scanner| {
                            scanner.filter(&filter).unwrap();
                        }))
                    });
                },
            );
        }
        for (share, filter) in [
            ("0.01%", "age = '0-0.01%'"),
            ("0.03%", "age IN ('0-0.01%', '0.01-0.03%')"),
            ("0.06%", "age IN ('0-0.01%', '0.01-0.03%', '0.03-0.06%')"),
            (
                "1%",
                "age IN ('0-0.01%', '0.01-0.03%', '0.03-0.06%', '0.06-1%')",
            ),
            (
                "50%",
                "age IN ('0-0.01%', '0.01-0.03%', '0.03-0.06%', '0.06-1%', '1-50%')",
            ),
        ] {
            group.bench_with_input(
                BenchmarkId::new(format!("oldest_{share}"), segments),
                &dataset,
                |b, dataset| {
                    b.iter(|| {
                        rt.block_on(ordered_limit(dataset, |scanner| {
                            scanner.filter(filter).unwrap();
                        }))
                    });
                },
            );
        }
        if segments > 1 {
            continue;
        }
        let mut appended = dataset.clone();
        let mut appended_batches = 0;
        for (share, total_batches) in [
            ("1%", 1),
            ("9%", 10),
            ("25%", 33),
            ("49%", 96),
            ("51%", 104),
        ] {
            let seed = DEFAULT_SEED.0 + 1_000 + u64::from(total_batches);
            rt.block_on(append_unindexed(
                &mut appended,
                total_batches - appended_batches,
                seed,
            ));
            appended_batches = total_batches;
            // While `age_idx` misses the appended rows, the read of them checks each one
            // against an oldest mask and yields no row until it has read them all.
            let stale_mask = appended.clone();
            rt.block_on(index_ages(&mut appended));
            let mut cases = vec![(String::new(), None, &appended)];
            if share == "9%" {
                cases.push((
                    "_mask_50%".to_string(),
                    Some("status = '50%'".to_string()),
                    &appended,
                ));
            }
            if matches!(share, "9%" | "49%") {
                cases.extend(
                    oldest_masks()
                        .into_iter()
                        .map(|(name, filter)| (format!("_{name}"), Some(filter), &appended)),
                );
            }
            if share == "49%" {
                cases.extend(
                    oldest_masks()
                        .into_iter()
                        .filter(|(name, _)| !name.ends_with("_and_appended"))
                        .map(|(name, filter)| {
                            (format!("_{name}_stale_mask"), Some(filter), &stale_mask)
                        }),
                );
            }
            for (case, filter, dataset) in &cases {
                for (mode, suffix) in [
                    (OrderedIndexMode::Auto, ""),
                    (OrderedIndexMode::Off, "_off"),
                ] {
                    group.bench_with_input(
                        BenchmarkId::new(format!("unindexed_{share}{case}{suffix}"), segments),
                        *dataset,
                        |b, dataset| {
                            b.iter(|| {
                                rt.block_on(ordered_limit(dataset, |scanner| {
                                    scanner.ordered_index(mode);
                                    if let Some(filter) = filter {
                                        scanner.filter(filter).unwrap();
                                    }
                                }))
                            });
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_baseline, bench_warm_indexed, bench_ordered_limit);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_baseline, bench_warm_indexed, bench_ordered_limit);

criterion_main!(benches);
