mod rust;
mod ts;

use std::path::Path;

use anyhow::{Result, bail};

#[derive(Debug, Clone)]
pub struct Definition {
    pub kind: String,
    pub name: String,
    pub parent_name: Option<String>,
    pub span_start: usize,
    pub span_end: usize,
    pub start_line: usize,
    pub start_column: usize,
    pub end_line: usize,
    pub end_column: usize,
    pub code: String,
    pub normalized_code: String,
}

pub fn extract_definitions(path: &Path, source: &str) -> Result<Vec<Definition>> {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default();
    match ext {
        "rs" => rust::extract(path, source),
        "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "mts" | "cts" => ts::extract(path, source),
        _ => bail!("unsupported file extension for {}", path.display()),
    }
}
