use std::collections::VecDeque;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use serde::Serialize;

use crate::db::{self, FileRecord, PendingEnrichment, SearchResult, SimilarPair, SymbolRecord};
use crate::extract;
use crate::gateway;
use crate::normalize::{embedding_text, hash_hex};
use crate::{FuseMethod, ScoreSpace, SearchUsing};

#[derive(Debug)]
struct EnrichmentResult {
    rowid: i64,
    kind: String,
    name: String,
    code_embedding: Option<std::result::Result<Vec<f32>, String>>,
    description: Option<std::result::Result<String, String>>,
    description_embedding: Option<std::result::Result<Vec<f32>, String>>,
}

pub fn run_index(
    db_path: PathBuf,
    project_path: PathBuf,
    reindex_all: bool,
    threads: usize,
) -> Result<()> {
    let summary = index_data(db_path, project_path, reindex_all, threads)?;
    println!(
        "indexed {} source files, refreshed {}, code embeddings {}, descriptions {}, description embeddings {} in {}",
        summary.seen_files,
        summary.refreshed_files,
        summary.code_embeddings,
        summary.descriptions,
        summary.description_embeddings,
        summary.db_path
    );
    Ok(())
}

pub fn index_data(
    db_path: PathBuf,
    project_path: PathBuf,
    reindex_all: bool,
    threads: usize,
) -> Result<IndexSummary> {
    let mut conn = db::open(&db_path)?;
    let project_path = project_path
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", project_path.display()))?;

    let mut seen = 0usize;
    let mut changed = 0usize;

    for path in source_files(&project_path)? {
        seen += 1;
        let source = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let metadata =
            fs::metadata(&path).with_context(|| format!("failed to stat {}", path.display()))?;
        let relative_path = path
            .strip_prefix(&project_path)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let file_hash = hash_hex(source.as_bytes());
        let file_record = FileRecord {
            path: relative_path.clone(),
            file_hash: file_hash.clone(),
            mtime_ns: metadata.mtime_nsec() + metadata.mtime() * 1_000_000_000,
            ctime_ns: metadata.ctime_nsec() + metadata.ctime() * 1_000_000_000,
            size: metadata.size() as i64,
        };

        if !reindex_all && file_unchanged(&conn, &file_record)? {
            continue;
        }

        changed += 1;
        let definitions = extract::extract_definitions(&path, &source)
            .with_context(|| format!("failed to extract definitions from {}", path.display()))?;

        let symbols = definitions
            .into_iter()
            .map(|definition| {
                let item_hash = hash_hex(definition.normalized_code.as_bytes());
                let stable_id = hash_hex(format!(
                    "{}\0{}\0{}\0{}\0{}\0{}",
                    relative_path,
                    definition.kind,
                    definition.name,
                    definition.span_start,
                    definition.span_end,
                    item_hash,
                ));
                SymbolRecord {
                    stable_id,
                    file_path: relative_path.clone(),
                    file_hash: file_hash.clone(),
                    kind: definition.kind,
                    name: definition.name,
                    parent_name: definition.parent_name,
                    span_start: definition.span_start as i64,
                    span_end: definition.span_end as i64,
                    start_line: definition.start_line as i64,
                    start_column: definition.start_column as i64,
                    end_line: definition.end_line as i64,
                    end_column: definition.end_column as i64,
                    code: definition.code,
                    normalized_code: definition.normalized_code,
                    item_hash,
                }
            })
            .collect::<Vec<_>>();

        db::replace_file(&mut conn, &file_record, &symbols)?;
    }

    let stats = process_pending_enrichment(&conn, threads)?;

    Ok(IndexSummary {
        seen_files: seen,
        refreshed_files: changed,
        code_embeddings: stats.code_embeddings,
        descriptions: stats.descriptions,
        description_embeddings: stats.description_embeddings,
        db_path: db_path.display().to_string(),
    })
}

pub fn run_query(
    db_path: PathBuf,
    query: String,
    limit: usize,
    using: SearchUsing,
    score_space: ScoreSpace,
    fuse_method: FuseMethod,
    code_weight: f64,
    description_weight: f64,
    kinds: Vec<String>,
) -> Result<()> {
    let results = query_data(
        db_path,
        query,
        limit,
        using,
        score_space,
        fuse_method,
        code_weight,
        description_weight,
        kinds,
    )?;
    print_results(&results, using, score_space);
    Ok(())
}

pub fn query_data(
    db_path: PathBuf,
    query: String,
    limit: usize,
    using: SearchUsing,
    score_space: ScoreSpace,
    fuse_method: FuseMethod,
    code_weight: f64,
    description_weight: f64,
    kinds: Vec<String>,
) -> Result<Vec<SearchResult>> {
    let conn = db::open(&db_path)?;
    let embedding = gateway::embed_text(&query)?;
    db::search(
        &conn,
        &embedding,
        limit,
        using,
        score_space,
        fuse_method,
        code_weight,
        description_weight,
        &kinds,
    )
}

pub fn run_similar(
    db_path: PathBuf,
    limit: usize,
    _neighbors: usize,
    using: SearchUsing,
    score_space: ScoreSpace,
    fuse_method: FuseMethod,
    code_weight: f64,
    description_weight: f64,
    kinds: Vec<String>,
) -> Result<()> {
    let pairs = similar_data(
        db_path,
        limit,
        using,
        score_space,
        fuse_method,
        code_weight,
        description_weight,
        kinds,
    )?;
    print_similar_pairs(&pairs, using, score_space);
    Ok(())
}

pub fn run_graph(
    db_path: PathBuf,
    out_path: PathBuf,
    k: usize,
    min_score: f64,
    cross_file_only: bool,
    kinds: Vec<String>,
) -> Result<()> {
    let graph = graph_data(db_path.clone(), k, min_score, cross_file_only, kinds)?;
    let json = serde_json::to_string_pretty(&graph)?;
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, json)?;
    println!(
        "wrote graph with {} nodes and {} edges to {}",
        graph.nodes.len(),
        graph.edges.len(),
        out_path.display()
    );
    println!("source db: {}", db_path.display());
    Ok(())
}

pub fn graph_data(
    db_path: PathBuf,
    k: usize,
    min_score: f64,
    cross_file_only: bool,
    kinds: Vec<String>,
) -> Result<GraphFile> {
    let conn = db::open(&db_path)?;
    let items = db::load_graph_items(&conn, SearchUsing::Description, &kinds)?;

    let nodes = items
        .iter()
        .map(|item| GraphNode {
            id: item.rowid,
            file_path: item.file_path.clone(),
            line: item.start_line,
            kind: item.kind.clone(),
            name: item.name.clone(),
            parent_name: item.parent_name.clone(),
        })
        .collect::<Vec<_>>();

    let edges = build_semantic_edges(&items, k, min_score, cross_file_only);
    let meta = GraphMeta {
        db_path: db_path.display().to_string(),
        using: "description".to_string(),
        score_space: "similarity".to_string(),
        k,
        min_score,
        cross_file_only,
        node_count: nodes.len(),
        edge_count: edges.len(),
    };

    Ok(GraphFile { meta, nodes, edges })
}

fn build_semantic_edges(
    items: &[db::SimilarityItem],
    k: usize,
    min_score: f64,
    cross_file_only: bool,
) -> Vec<GraphEdge> {
    let mut neighbor_lists = Vec::with_capacity(items.len());

    for (left_index, left) in items.iter().enumerate() {
        let mut neighbors = Vec::new();
        for (right_index, right) in items.iter().enumerate() {
            if left_index == right_index {
                continue;
            }
            if cross_file_only && left.file_path == right.file_path {
                continue;
            }
            let score = cosine_similarity(&left.embedding, &right.embedding) as f64;
            if score >= min_score && !is_generic_pair(left, right) {
                neighbors.push((right_index, score));
            }
        }
        neighbors.sort_by(|a, b| b.1.total_cmp(&a.1));
        neighbors.truncate(k);
        neighbor_lists.push(neighbors);
    }

    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (left_index, neighbors) in neighbor_lists.iter().enumerate() {
        for (rank_index, (right_index, score)) in neighbors.iter().enumerate() {
            let left = &items[left_index];
            let right = &items[*right_index];
            let key = if left.rowid < right.rowid {
                (left.rowid, right.rowid)
            } else {
                (right.rowid, left.rowid)
            };
            if seen.contains(&key) {
                continue;
            }

            let reciprocal = neighbor_lists[*right_index]
                .iter()
                .take(k)
                .any(|(candidate_index, _)| *candidate_index == left_index);
            if !reciprocal {
                continue;
            }

            seen.insert(key);
            edges.push(GraphEdge {
                source: left.rowid,
                target: right.rowid,
                edge_type: "semantic_description".to_string(),
                score: *score,
                rank: rank_index + 1,
                reciprocal,
            });
        }
    }

    edges.sort_by(|a, b| b.score.total_cmp(&a.score));
    edges
}

fn is_generic_pair(left: &db::SimilarityItem, right: &db::SimilarityItem) -> bool {
    is_generic_name(&left.name) && is_generic_name(&right.name)
}

fn is_generic_name(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase();
    ["util", "utils", "helper", "helpers", "common", "base"]
        .iter()
        .any(|needle| lowered.contains(needle))
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;

    for (left_value, right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

pub fn similar_data(
    db_path: PathBuf,
    limit: usize,
    using: SearchUsing,
    score_space: ScoreSpace,
    fuse_method: FuseMethod,
    code_weight: f64,
    description_weight: f64,
    kinds: Vec<String>,
) -> Result<Vec<SimilarPair>> {
    let conn = db::open(&db_path)?;
    db::top_similar_pairs(
        &conn,
        limit,
        using,
        score_space,
        fuse_method,
        code_weight,
        description_weight,
        &kinds,
    )
}

fn file_unchanged(conn: &rusqlite::Connection, file_record: &FileRecord) -> Result<bool> {
    Ok(matches!(
        db::get_file_record(conn, &file_record.path)?,
        Some(existing)
            if existing.file_hash == file_record.file_hash
                && existing.mtime_ns == file_record.mtime_ns
                && existing.ctime_ns == file_record.ctime_ns
                && existing.size == file_record.size
    ))
}

#[derive(Default, Serialize)]
pub struct EnrichmentStats {
    code_embeddings: usize,
    descriptions: usize,
    description_embeddings: usize,
}

#[derive(Debug, Serialize)]
pub struct IndexSummary {
    pub seen_files: usize,
    pub refreshed_files: usize,
    pub code_embeddings: usize,
    pub descriptions: usize,
    pub description_embeddings: usize,
    pub db_path: String,
}

#[derive(Debug, Serialize)]
pub struct GraphNode {
    pub id: i64,
    pub file_path: String,
    pub line: i64,
    pub kind: String,
    pub name: String,
    pub parent_name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GraphEdge {
    pub source: i64,
    pub target: i64,
    pub edge_type: String,
    pub score: f64,
    pub rank: usize,
    pub reciprocal: bool,
}

#[derive(Debug, Serialize)]
pub struct GraphFile {
    pub meta: GraphMeta,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Serialize)]
pub struct GraphMeta {
    pub db_path: String,
    pub using: String,
    pub score_space: String,
    pub k: usize,
    pub min_score: f64,
    pub cross_file_only: bool,
    pub node_count: usize,
    pub edge_count: usize,
}

fn process_pending_enrichment(
    conn: &rusqlite::Connection,
    threads: usize,
) -> Result<EnrichmentStats> {
    let jobs = db::pending_enrichments(conn)?;
    if jobs.is_empty() {
        return Ok(EnrichmentStats::default());
    }

    let worker_count = threads.max(1).min(4);
    let queue = Arc::new(Mutex::new(VecDeque::from(jobs)));
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();

    for _ in 0..worker_count {
        let queue = Arc::clone(&queue);
        let tx = tx.clone();
        handles.push(thread::spawn(move || {
            loop {
                let job = {
                    let mut guard = queue.lock().expect("queue lock poisoned");
                    guard.pop_front()
                };
                let Some(job) = job else {
                    break;
                };
                let result = enrich_job(job);
                if tx.send(result).is_err() {
                    break;
                }
            }
        }));
    }
    drop(tx);

    let mut stats = EnrichmentStats::default();
    for result in rx {
        apply_enrichment_result(conn, result, &mut stats)?;
    }

    for handle in handles {
        let _ = handle.join();
    }

    Ok(stats)
}

fn enrich_job(job: PendingEnrichment) -> EnrichmentResult {
    let mut description = if job.needs_description {
        Some(
            gateway::describe_code(&job.kind, &job.name, &job.normalized_code)
                .map_err(|err| err.to_string()),
        )
    } else {
        None
    };

    let code_embedding = if job.needs_code_embedding {
        Some(
            gateway::embed_text(&embedding_text(
                &job.file_path,
                &job.kind,
                &job.name,
                job.parent_name.as_deref(),
                &job.normalized_code,
            ))
            .map_err(|err| err.to_string()),
        )
    } else {
        None
    };

    let description_text = match &description {
        Some(Ok(text)) => Some(text.clone()),
        Some(Err(_)) => None,
        None => job.description.clone(),
    };

    let description_embedding = if job.needs_description_embedding {
        description_text
            .as_deref()
            .map(|text| gateway::embed_text(text).map_err(|err| err.to_string()))
            .or_else(|| {
                description = Some(Err("description unavailable for embedding".to_string()));
                None
            })
    } else {
        None
    };

    EnrichmentResult {
        rowid: job.rowid,
        kind: job.kind,
        name: job.name,
        code_embedding,
        description,
        description_embedding,
    }
}

fn apply_enrichment_result(
    conn: &rusqlite::Connection,
    result: EnrichmentResult,
    stats: &mut EnrichmentStats,
) -> Result<()> {
    if let Some(code_embedding) = result.code_embedding {
        match code_embedding {
            Ok(embedding) => {
                db::set_code_embedding(conn, result.rowid, &embedding)?;
                stats.code_embeddings += 1;
            }
            Err(err) => {
                db::set_code_embedding_error(conn, result.rowid, &err)?;
                eprintln!(
                    "code embedding failed for {} {}: {err}",
                    result.kind, result.name
                );
            }
        }
    }

    if let Some(description) = result.description {
        match description {
            Ok(text) => {
                db::set_description(conn, result.rowid, &text)?;
                stats.descriptions += 1;
            }
            Err(err) => {
                db::set_description_error(conn, result.rowid, &err)?;
                eprintln!(
                    "description failed for {} {}: {err}",
                    result.kind, result.name
                );
            }
        }
    }

    if let Some(description_embedding) = result.description_embedding {
        match description_embedding {
            Ok(embedding) => {
                db::set_description_embedding(conn, result.rowid, &embedding)?;
                stats.description_embeddings += 1;
            }
            Err(err) => {
                db::set_description_embedding_error(conn, result.rowid, &err)?;
                eprintln!(
                    "description embedding failed for {} {}: {err}",
                    result.kind, result.name
                );
            }
        }
    }

    Ok(())
}

fn source_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let walker = WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .build();

    for entry in walker {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        if matches_extension(path) {
            files.push(path.to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

fn matches_extension(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("rs" | "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "mts" | "cts")
    )
}

fn print_results(results: &[SearchResult], using: SearchUsing, score_space: ScoreSpace) {
    if results.is_empty() {
        println!("no matches");
        return;
    }

    for (index, result) in results.iter().enumerate() {
        let parent = result
            .parent_name
            .as_deref()
            .map(|parent| format!(" parent={parent}"))
            .unwrap_or_default();
        println!("# {}", index + 1);
        println!("kind: {}", result.kind);
        println!("name: {}{}", result.name, parent);
        println!(
            "location: {}:{}:{}-{}:{}",
            result.file_path,
            result.start_line,
            result.start_column,
            result.end_line,
            result.end_column,
        );
        println!("score_space: {:?}", score_space);
        if let Some(score) = result.code_score {
            println!("code_score: {:.6}", score);
        }
        if let Some(score) = result.description_score {
            println!("description_score: {:.6}", score);
        }
        println!("fused_score: {:.6}", result.fused_score);
        if matches!(using, SearchUsing::Description | SearchUsing::Fused) {
            if let Some(description) = &result.description {
                println!("description: {}", description);
            }
        }
        println!("{}", result.normalized_code);
        println!();
    }
}

fn print_similar_pairs(pairs: &[SimilarPair], using: SearchUsing, score_space: ScoreSpace) {
    if pairs.is_empty() {
        println!("no similar pairs");
        return;
    }

    println!("using: {:?}", using);
    println!("score_space: {:?}", score_space);
    for (index, pair) in pairs.iter().enumerate() {
        println!("# {}", index + 1);
        if let Some(score) = pair.code_score {
            println!("code_score: {:.6}", score);
        }
        if let Some(score) = pair.description_score {
            println!("description_score: {:.6}", score);
        }
        println!("fused_score: {:.6}", pair.fused_score);
        print_similarity_side("left", &pair.left);
        print_similarity_side("right", &pair.right);
        println!();
    }
}

fn print_similarity_side(label: &str, item: &db::SimilarityItem) {
    let parent = item
        .parent_name
        .as_deref()
        .map(|parent| format!(" parent={parent}"))
        .unwrap_or_default();
    println!("{}: {} {}{}", label, item.kind, item.name, parent);
    println!("{} path: {}", label, item.file_path);
    if let Some(description) = &item.description {
        println!("{} description: {}", label, description);
    }
    println!("{} code:", label);
    println!("{}", item.normalized_code);
}
