use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub model_name: String,
    pub input_width: usize,
    pub input_height: usize,
    pub labels: Vec<String>,
}
