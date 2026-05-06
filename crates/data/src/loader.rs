use aionfm_utils::{
    AionError, AionResult, CovariateSeries, EntityMetadata, EntitySeries, Frequency,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

/// Supported corpus storage formats from the spec.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFormat {
    Parquet,
    ArrowIpc,
    Csv,
    ZippedCsv,
    MetadataCatalog,
}

/// Data-loading request used by corpus readers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoadRequest {
    pub path: PathBuf,
    pub format: DataFormat,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub frequency: Option<Frequency>,
    #[serde(default)]
    pub selected_columns: Vec<String>,
}

/// Abstract loader so parquet, arrow, CSV, and streaming backends are swappable.
#[async_trait]
pub trait DataLoader: Send + Sync {
    async fn load_series(&self, request: LoadRequest) -> AionResult<Vec<EntitySeries>>;
}

/// Placeholder loader that preserves the API surface for unsupported storage backends.
#[derive(Clone, Debug, Default)]
pub struct UnsupportedDataLoader;

#[async_trait]
impl DataLoader for UnsupportedDataLoader {
    async fn load_series(&self, request: LoadRequest) -> AionResult<Vec<EntitySeries>> {
        Err(AionError::Unsupported(format!(
            "{:?} loading is not implemented yet for {}",
            request.format,
            request.path.display()
        )))
    }
}

/// Local corpus loader with a concrete CSV implementation.
#[derive(Clone, Debug, Default)]
pub struct LocalDataLoader;

#[async_trait]
impl DataLoader for LocalDataLoader {
    async fn load_series(&self, request: LoadRequest) -> AionResult<Vec<EntitySeries>> {
        match request.format {
            DataFormat::Csv => load_csv(&request),
            other => Err(AionError::Unsupported(format!(
                "{other:?} loading is not implemented yet for {}",
                request.path.display()
            ))),
        }
    }
}

#[derive(Debug, Default)]
struct SeriesBuilder {
    entity_id: String,
    target: String,
    values: Vec<f32>,
    timestamps: Vec<String>,
    covariates: BTreeMap<String, Vec<f32>>,
}

impl SeriesBuilder {
    fn push_value(&mut self, timestamp: Option<&str>, value: f32) {
        self.values.push(value);
        self.timestamps
            .push(timestamp.unwrap_or_default().to_string());
        for covariate in self.covariates.values_mut() {
            if covariate.len() + 1 < self.values.len() {
                covariate.resize(self.values.len() - 1, f32::NAN);
            }
        }
    }

    fn push_covariate(&mut self, name: &str, value: f32) {
        let covariate = self.covariates.entry(name.to_string()).or_default();
        if covariate.len() + 1 < self.values.len() {
            covariate.resize(self.values.len() - 1, f32::NAN);
        }
        covariate.push(value);
    }

    fn finish(mut self, request: &LoadRequest) -> EntitySeries {
        for covariate in self.covariates.values_mut() {
            covariate.resize(self.values.len(), f32::NAN);
        }
        let frequency = request.frequency.clone().unwrap_or_default();
        EntitySeries {
            entity_id: self.entity_id.clone(),
            target: self.target,
            values: self.values,
            timestamps: self.timestamps,
            covariates: self
                .covariates
                .into_iter()
                .map(|(name, values)| CovariateSeries {
                    name,
                    values,
                    known_future: false,
                })
                .collect(),
            metadata: EntityMetadata {
                entity_id: self.entity_id,
                domain: request.domain.clone(),
                frequency,
                attributes: Default::default(),
            },
        }
    }
}

fn load_csv(request: &LoadRequest) -> AionResult<Vec<EntitySeries>> {
    let mut reader = csv::Reader::from_path(&request.path).map_err(csv_error)?;
    let headers = reader.headers().map_err(csv_error)?.clone();
    let schema = CsvSchema::from_headers(&headers, &request.selected_columns)?;
    let mut builders: BTreeMap<(String, String), SeriesBuilder> = BTreeMap::new();

    for record in reader.records() {
        let record = record.map_err(csv_error)?;
        let entity_id = field(&record, schema.entity_id).unwrap_or("default");
        let target = schema
            .target
            .and_then(|index| field(&record, Some(index)))
            .or_else(|| {
                schema
                    .variable_name
                    .and_then(|index| field(&record, Some(index)))
            })
            .unwrap_or("value");
        let key = (entity_id.to_string(), target.to_string());
        let builder = builders.entry(key).or_insert_with(|| SeriesBuilder {
            entity_id: entity_id.to_string(),
            target: target.to_string(),
            ..SeriesBuilder::default()
        });
        let value = schema
            .value
            .and_then(|index| parse_f32(field(&record, Some(index))))
            .unwrap_or(f32::NAN);
        let timestamp = schema
            .timestamp
            .and_then(|index| field(&record, Some(index)));
        builder.push_value(timestamp, value);
        for (name, index) in &schema.covariates {
            let value = parse_f32(field(&record, Some(*index))).unwrap_or(f32::NAN);
            builder.push_covariate(name, value);
        }
    }

    let mut series: Vec<EntitySeries> = builders
        .into_values()
        .map(|builder| builder.finish(request))
        .collect();
    sort_series(&mut series);
    Ok(series)
}

#[derive(Debug)]
struct CsvSchema {
    timestamp: Option<usize>,
    entity_id: Option<usize>,
    variable_name: Option<usize>,
    target: Option<usize>,
    value: Option<usize>,
    covariates: Vec<(String, usize)>,
}

impl CsvSchema {
    fn from_headers(headers: &csv::StringRecord, selected_columns: &[String]) -> AionResult<Self> {
        let header_index: BTreeMap<String, usize> = headers
            .iter()
            .enumerate()
            .map(|(index, header)| (header.trim().to_ascii_lowercase(), index))
            .collect();
        let selected: BTreeSet<String> = selected_columns
            .iter()
            .map(|name| name.trim().to_ascii_lowercase())
            .collect();
        let timestamp = first_index(&header_index, &["timestamp", "time", "date", "datetime"]);
        let entity_id = first_index(&header_index, &["entity_id", "entity", "series_id", "id"]);
        let variable_name = first_index(&header_index, &["variable_name", "variable", "metric"]);
        let target = first_index(&header_index, &["target", "target_name"]);
        let value = first_index(&header_index, &["value", "target_value", "y"])
            .or_else(|| selected.iter().find_map(|name| header_index.get(name).copied()))
            .ok_or_else(|| {
                AionError::Validation(
                    "CSV requires a value column named value, target_value, y, or selected_columns entry".into(),
                )
            })?;
        let mut covariates = Vec::new();
        for (index, header) in headers.iter().enumerate() {
            let normalized = header.trim().to_ascii_lowercase();
            if Some(index) == timestamp
                || Some(index) == entity_id
                || Some(index) == variable_name
                || Some(index) == target
                || index == value
            {
                continue;
            }
            if normalized.starts_with("covariate_") || selected.contains(&normalized) {
                covariates.push((
                    header.trim().trim_start_matches("covariate_").to_string(),
                    index,
                ));
            }
        }
        Ok(Self {
            timestamp,
            entity_id,
            variable_name,
            target,
            value: Some(value),
            covariates,
        })
    }
}

fn first_index(index: &BTreeMap<String, usize>, names: &[&str]) -> Option<usize> {
    names.iter().find_map(|name| index.get(*name).copied())
}

fn field(record: &csv::StringRecord, index: Option<usize>) -> Option<&str> {
    index
        .and_then(|index| record.get(index))
        .filter(|value| !value.trim().is_empty())
}

fn parse_f32(value: Option<&str>) -> Option<f32> {
    value.and_then(|value| value.trim().parse::<f32>().ok())
}

fn sort_series(series: &mut [EntitySeries]) {
    series.sort_by(|left, right| {
        left.entity_id
            .cmp(&right.entity_id)
            .then_with(|| left.target.cmp(&right.target))
    });
}

fn csv_error(error: csv::Error) -> AionError {
    AionError::Validation(error.to_string())
}

/// Convenience helper for one-off CSV loads in examples and tests.
pub async fn load_csv_file(path: impl AsRef<Path>) -> AionResult<Vec<EntitySeries>> {
    LocalDataLoader
        .load_series(LoadRequest {
            path: path.as_ref().to_path_buf(),
            format: DataFormat::Csv,
            domain: None,
            frequency: None,
            selected_columns: vec![],
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn loads_grouped_csv_series_with_covariates() {
        let path = std::env::temp_dir().join(format!("aionfm-data-{}.csv", std::process::id()));
        {
            let mut file = std::fs::File::create(&path).unwrap();
            writeln!(
                file,
                "timestamp,entity_id,target,value,covariate_temperature\n2026-01-01,store_1,demand,10,70\n2026-01-02,store_1,demand,11,71"
            )
            .unwrap();
        }
        let series = load_csv_file(&path).await.unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(series.len(), 1);
        assert_eq!(series[0].entity_id, "store_1");
        assert_eq!(series[0].covariates[0].name, "temperature");
        assert_eq!(series[0].values, vec![10.0, 11.0]);
    }
}
