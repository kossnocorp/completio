use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use bytemuck::{cast_slice, try_cast_slice};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use sqlite_vec::sqlite3_vec_init;

use crate::gateway::{CHAT_MODEL, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL};
use crate::{FuseMethod, ScoreSpace, SearchUsing};

#[derive(Debug, Clone)]
pub struct FileRecord {
    pub path: String,
    pub file_hash: String,
    pub mtime_ns: i64,
    pub ctime_ns: i64,
    pub size: i64,
}

#[derive(Debug, Clone)]
pub struct SymbolRecord {
    pub stable_id: String,
    pub file_path: String,
    pub file_hash: String,
    pub kind: String,
    pub name: String,
    pub parent_name: Option<String>,
    pub span_start: i64,
    pub span_end: i64,
    pub start_line: i64,
    pub start_column: i64,
    pub end_line: i64,
    pub end_column: i64,
    pub code: String,
    pub normalized_code: String,
    pub item_hash: String,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub file_path: String,
    pub kind: String,
    pub name: String,
    pub parent_name: Option<String>,
    pub start_line: i64,
    pub start_column: i64,
    pub end_line: i64,
    pub end_column: i64,
    pub code_score: Option<f64>,
    pub description_score: Option<f64>,
    pub fused_score: f64,
    pub normalized_code: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PendingEnrichment {
    pub rowid: i64,
    pub file_path: String,
    pub kind: String,
    pub name: String,
    pub parent_name: Option<String>,
    pub normalized_code: String,
    pub description: Option<String>,
    pub needs_code_embedding: bool,
    pub needs_description: bool,
    pub needs_description_embedding: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimilarityItem {
    pub rowid: i64,
    pub file_path: String,
    pub start_line: i64,
    pub kind: String,
    pub name: String,
    pub parent_name: Option<String>,
    pub normalized_code: String,
    pub description: Option<String>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimilarPair {
    pub left: SimilarityItem,
    pub right: SimilarityItem,
    pub code_score: Option<f64>,
    pub description_score: Option<f64>,
    pub fused_score: f64,
}

pub fn open(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite3_vec_init as *const (),
        )));
    }

    let conn =
        Connection::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    migrate(&conn)?;
    Ok(conn)
}

fn migrate(conn: &Connection) -> Result<()> {
    conn.execute_batch(&format!(
        "
        CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY,
          file_hash TEXT NOT NULL,
          mtime_ns INTEGER NOT NULL,
          ctime_ns INTEGER NOT NULL,
          size INTEGER NOT NULL,
          indexed_at INTEGER NOT NULL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS symbols (
          id INTEGER PRIMARY KEY,
          stable_id TEXT NOT NULL UNIQUE,
          file_path TEXT NOT NULL,
          file_hash TEXT NOT NULL,
          kind TEXT NOT NULL,
          name TEXT NOT NULL,
          parent_name TEXT,
          span_start INTEGER NOT NULL,
          span_end INTEGER NOT NULL,
          start_line INTEGER NOT NULL,
          start_column INTEGER NOT NULL,
          end_line INTEGER NOT NULL,
          end_column INTEGER NOT NULL,
          code TEXT NOT NULL,
          normalized_code TEXT NOT NULL,
          item_hash TEXT NOT NULL,
          description TEXT,
          code_embedding_model TEXT,
          code_embedded_at INTEGER,
          code_embedding_error TEXT,
          description_model TEXT,
          description_generated_at INTEGER,
          description_error TEXT,
          description_embedding_model TEXT,
          description_embedded_at INTEGER,
          description_embedding_error TEXT,
          FOREIGN KEY (file_path) REFERENCES files(path) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS symbols_file_path_idx ON symbols(file_path);
        CREATE INDEX IF NOT EXISTS symbols_stable_id_idx ON symbols(stable_id);
        CREATE INDEX IF NOT EXISTS symbols_name_idx ON symbols(name);

        CREATE VIRTUAL TABLE IF NOT EXISTS symbol_code_vec USING vec0(
          embedding float[{}]
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS symbol_desc_vec USING vec0(
          embedding float[{}]
        );
        ",
        EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS
    ))?;

    for statement in [
        "ALTER TABLE symbols ADD COLUMN description TEXT",
        "ALTER TABLE symbols ADD COLUMN code_embedding_model TEXT",
        "ALTER TABLE symbols ADD COLUMN code_embedded_at INTEGER",
        "ALTER TABLE symbols ADD COLUMN code_embedding_error TEXT",
        "ALTER TABLE symbols ADD COLUMN description_model TEXT",
        "ALTER TABLE symbols ADD COLUMN description_generated_at INTEGER",
        "ALTER TABLE symbols ADD COLUMN description_error TEXT",
        "ALTER TABLE symbols ADD COLUMN description_embedding_model TEXT",
        "ALTER TABLE symbols ADD COLUMN description_embedded_at INTEGER",
        "ALTER TABLE symbols ADD COLUMN description_embedding_error TEXT",
    ] {
        let _ = conn.execute(statement, []);
    }

    Ok(())
}

pub fn get_file_record(conn: &Connection, path: &str) -> Result<Option<FileRecord>> {
    conn.query_row(
        "SELECT path, file_hash, mtime_ns, ctime_ns, size FROM files WHERE path = ?1",
        [path],
        |row| {
            Ok(FileRecord {
                path: row.get(0)?,
                file_hash: row.get(1)?,
                mtime_ns: row.get(2)?,
                ctime_ns: row.get(3)?,
                size: row.get(4)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

pub fn replace_file(
    conn: &mut Connection,
    file: &FileRecord,
    symbols: &[SymbolRecord],
) -> Result<Vec<(i64, SymbolRecord)>> {
    let tx = conn.transaction()?;
    tx.execute(
        "DELETE FROM symbol_code_vec WHERE rowid IN (SELECT id FROM symbols WHERE file_path = ?1)",
        [&file.path],
    )?;
    tx.execute(
        "DELETE FROM symbol_desc_vec WHERE rowid IN (SELECT id FROM symbols WHERE file_path = ?1)",
        [&file.path],
    )?;
    let _ = tx.execute(
        "DELETE FROM symbol_vec WHERE rowid IN (SELECT id FROM symbols WHERE file_path = ?1)",
        [&file.path],
    );
    tx.execute("DELETE FROM symbols WHERE file_path = ?1", [&file.path])?;
    tx.execute(
        "INSERT INTO files(path, file_hash, mtime_ns, ctime_ns, size, indexed_at)
         VALUES(?1, ?2, ?3, ?4, ?5, unixepoch())
         ON CONFLICT(path) DO UPDATE SET
           file_hash = excluded.file_hash,
           mtime_ns = excluded.mtime_ns,
           ctime_ns = excluded.ctime_ns,
           size = excluded.size,
           indexed_at = unixepoch()",
        params![
            file.path,
            file.file_hash,
            file.mtime_ns,
            file.ctime_ns,
            file.size
        ],
    )?;

    let mut inserted = Vec::with_capacity(symbols.len());
    {
        let mut stmt = tx.prepare(
            "INSERT INTO symbols(
               stable_id, file_path, file_hash, kind, name, parent_name,
               span_start, span_end, start_line, start_column, end_line, end_column,
               code, normalized_code, item_hash
             ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        )?;
        for symbol in symbols {
            stmt.execute(params![
                symbol.stable_id,
                symbol.file_path,
                symbol.file_hash,
                symbol.kind,
                symbol.name,
                symbol.parent_name,
                symbol.span_start,
                symbol.span_end,
                symbol.start_line,
                symbol.start_column,
                symbol.end_line,
                symbol.end_column,
                symbol.code,
                symbol.normalized_code,
                symbol.item_hash,
            ])?;
            inserted.push((tx.last_insert_rowid(), symbol.clone()));
        }
    }
    tx.commit()?;
    Ok(inserted)
}

pub fn set_code_embedding(conn: &Connection, symbol_rowid: i64, embedding: &[f32]) -> Result<()> {
    conn.execute(
        "DELETE FROM symbol_code_vec WHERE rowid = ?1",
        [symbol_rowid],
    )?;
    conn.execute(
        "INSERT INTO symbol_code_vec(rowid, embedding) VALUES(?1, ?2)",
        params![symbol_rowid, cast_slice(embedding)],
    )?;
    conn.execute(
        "UPDATE symbols
         SET code_embedding_model = ?2,
             code_embedded_at = unixepoch(),
             code_embedding_error = NULL
         WHERE id = ?1",
        params![symbol_rowid, EMBEDDING_MODEL],
    )?;
    Ok(())
}

pub fn set_code_embedding_error(conn: &Connection, symbol_rowid: i64, message: &str) -> Result<()> {
    conn.execute(
        "UPDATE symbols SET code_embedding_model = ?2, code_embedding_error = ?3 WHERE id = ?1",
        params![symbol_rowid, EMBEDDING_MODEL, message],
    )?;
    Ok(())
}

pub fn set_description(conn: &Connection, symbol_rowid: i64, description: &str) -> Result<()> {
    conn.execute(
        "UPDATE symbols
         SET description = ?2,
             description_model = ?3,
             description_generated_at = unixepoch(),
             description_error = NULL
         WHERE id = ?1",
        params![symbol_rowid, description, CHAT_MODEL],
    )?;
    Ok(())
}

pub fn set_description_error(conn: &Connection, symbol_rowid: i64, message: &str) -> Result<()> {
    conn.execute(
        "UPDATE symbols SET description_model = ?2, description_error = ?3 WHERE id = ?1",
        params![symbol_rowid, CHAT_MODEL, message],
    )?;
    Ok(())
}

pub fn set_description_embedding(
    conn: &Connection,
    symbol_rowid: i64,
    embedding: &[f32],
) -> Result<()> {
    conn.execute(
        "DELETE FROM symbol_desc_vec WHERE rowid = ?1",
        [symbol_rowid],
    )?;
    conn.execute(
        "INSERT INTO symbol_desc_vec(rowid, embedding) VALUES(?1, ?2)",
        params![symbol_rowid, cast_slice(embedding)],
    )?;
    conn.execute(
        "UPDATE symbols
         SET description_embedding_model = ?2,
             description_embedded_at = unixepoch(),
             description_embedding_error = NULL
         WHERE id = ?1",
        params![symbol_rowid, EMBEDDING_MODEL],
    )?;
    Ok(())
}

pub fn set_description_embedding_error(
    conn: &Connection,
    symbol_rowid: i64,
    message: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE symbols
         SET description_embedding_model = ?2,
             description_embedding_error = ?3
         WHERE id = ?1",
        params![symbol_rowid, EMBEDDING_MODEL, message],
    )?;
    Ok(())
}

pub fn pending_enrichments(conn: &Connection) -> Result<Vec<PendingEnrichment>> {
    let mut stmt = conn.prepare(
        "SELECT
           id,
           file_path,
           kind,
           name,
           parent_name,
           normalized_code,
           description,
           code_embedded_at IS NULL,
           description_generated_at IS NULL,
           description_embedded_at IS NULL
         FROM symbols
         WHERE code_embedded_at IS NULL
            OR description_generated_at IS NULL
            OR description_embedded_at IS NULL
         ORDER BY file_path, span_start, id",
    )?;

    let rows = stmt.query_map([], |row| {
        Ok(PendingEnrichment {
            rowid: row.get(0)?,
            file_path: row.get(1)?,
            kind: row.get(2)?,
            name: row.get(3)?,
            parent_name: row.get(4)?,
            normalized_code: row.get(5)?,
            description: row.get(6)?,
            needs_code_embedding: row.get(7)?,
            needs_description: row.get(8)?,
            needs_description_embedding: row.get(9)?,
        })
    })?;

    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

pub fn search(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
    using: SearchUsing,
    score_space: ScoreSpace,
    fuse_method: FuseMethod,
    code_weight: f64,
    description_weight: f64,
    kinds: &[String],
) -> Result<Vec<SearchResult>> {
    match using {
        SearchUsing::Code => search_single(
            conn,
            query_embedding,
            limit,
            "symbol_code_vec",
            SearchUsing::Code,
            score_space,
            kinds,
        ),
        SearchUsing::Description => search_single(
            conn,
            query_embedding,
            limit,
            "symbol_desc_vec",
            SearchUsing::Description,
            score_space,
            kinds,
        ),
        SearchUsing::Fused => search_fused(
            conn,
            query_embedding,
            limit,
            score_space,
            fuse_method,
            code_weight,
            description_weight,
            kinds,
        ),
    }
}

pub fn top_similar_pairs(
    conn: &Connection,
    limit: usize,
    using: SearchUsing,
    score_space: ScoreSpace,
    fuse_method: FuseMethod,
    code_weight: f64,
    description_weight: f64,
    kinds: &[String],
) -> Result<Vec<SimilarPair>> {
    let code_items = load_similarity_items(conn, "symbol_code_vec", kinds)?;
    let code_by_id = code_items
        .iter()
        .cloned()
        .map(|item| (item.rowid, item))
        .collect::<HashMap<_, _>>();
    let desc_by_id = load_similarity_items(conn, "symbol_desc_vec", kinds)?
        .into_iter()
        .map(|item| (item.rowid, item))
        .collect::<HashMap<_, _>>();

    let mut ranked = Vec::new();
    for left_index in 0..code_items.len() {
        for right_index in left_index + 1..code_items.len() {
            let left = &code_items[left_index];
            let right = &code_items[right_index];
            let Some(score) = pair_score(
                left,
                right,
                desc_by_id.get(&left.rowid),
                desc_by_id.get(&right.rowid),
                using,
                score_space,
                fuse_method,
                code_weight,
                description_weight,
            ) else {
                continue;
            };

            ranked.push(SimilarPair {
                left: code_by_id
                    .get(&left.rowid)
                    .cloned()
                    .unwrap_or_else(|| left.clone()),
                right: code_by_id
                    .get(&right.rowid)
                    .cloned()
                    .unwrap_or_else(|| right.clone()),
                code_score: score.code_score,
                description_score: score.description_score,
                fused_score: score.fused_score,
            });
        }
    }

    sort_best_first_pairs(&mut ranked, score_space);
    ranked.truncate(limit);
    Ok(ranked)
}

pub fn load_graph_items(
    conn: &Connection,
    using: SearchUsing,
    kinds: &[String],
) -> Result<Vec<SimilarityItem>> {
    let table = match using {
        SearchUsing::Code => "symbol_code_vec",
        SearchUsing::Description | SearchUsing::Fused => "symbol_desc_vec",
    };
    load_similarity_items(conn, table, kinds)
}

#[allow(dead_code)]
pub fn list_kinds(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT DISTINCT kind FROM symbols ORDER BY kind")?;
    let rows = stmt.query_map([], |row| row.get(0))?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn search_single(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
    table: &str,
    using: SearchUsing,
    score_space: ScoreSpace,
    kinds: &[String],
) -> Result<Vec<SearchResult>> {
    let kind_filter = kind_filter_sql(kinds, "symbols.kind");
    let sql = format!(
        "SELECT
           symbols.file_path,
           symbols.kind,
           symbols.name,
           symbols.parent_name,
           symbols.start_line,
           symbols.start_column,
           symbols.end_line,
           symbols.end_column,
           vec.distance,
           symbols.normalized_code,
           symbols.description
         FROM {table} AS vec
         JOIN symbols ON symbols.id = vec.rowid
         WHERE vec.embedding MATCH ?1
           AND k = ?2
           {kind_filter}
         LIMIT ?2"
    );
    let mut stmt = conn.prepare(&sql)?;

    let rows = stmt.query_map(params![cast_slice(query_embedding), limit as i64], |row| {
        let score = score_from_distance(row.get(8)?, score_space);
        Ok(SearchResult {
            file_path: row.get(0)?,
            kind: row.get(1)?,
            name: row.get(2)?,
            parent_name: row.get(3)?,
            start_line: row.get(4)?,
            start_column: row.get(5)?,
            end_line: row.get(6)?,
            end_column: row.get(7)?,
            code_score: matches!(using, SearchUsing::Code).then_some(score),
            description_score: matches!(using, SearchUsing::Description).then_some(score),
            fused_score: score,
            normalized_code: row.get(9)?,
            description: row.get(10)?,
        })
    })?;

    let mut results = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    sort_best_first_results(&mut results, score_space);
    Ok(results)
}

fn search_fused(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
    score_space: ScoreSpace,
    fuse_method: FuseMethod,
    code_weight: f64,
    description_weight: f64,
    kinds: &[String],
) -> Result<Vec<SearchResult>> {
    let k = ((limit.max(1)) * 8) as i64;
    let kind_filter = kind_filter_sql(kinds, "symbols.kind");
    let sql = "WITH code_hits AS (
         SELECT rowid, distance
         FROM symbol_code_vec
         WHERE embedding MATCH ?1 AND k = ?2
       ),
       desc_hits AS (
         SELECT rowid, distance
         FROM symbol_desc_vec
         WHERE embedding MATCH ?1 AND k = ?2
       )
       SELECT
         symbols.file_path,
         symbols.kind,
         symbols.name,
         symbols.parent_name,
         symbols.start_line,
         symbols.start_column,
         symbols.end_line,
         symbols.end_column,
         code_hits.distance,
         desc_hits.distance,
         symbols.normalized_code,
         symbols.description
       FROM code_hits
       JOIN desc_hits ON desc_hits.rowid = code_hits.rowid
       JOIN symbols ON symbols.id = code_hits.rowid
       {kind_filter}
       LIMIT ?3";
    let sql = sql.replace("{kind_filter}", &kind_filter);
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(
        params![cast_slice(query_embedding), k, limit as i64],
        |row| {
            let code_score = score_from_distance(row.get(8)?, score_space);
            let description_score = score_from_distance(row.get(9)?, score_space);
            Ok(SearchResult {
                file_path: row.get(0)?,
                kind: row.get(1)?,
                name: row.get(2)?,
                parent_name: row.get(3)?,
                start_line: row.get(4)?,
                start_column: row.get(5)?,
                end_line: row.get(6)?,
                end_column: row.get(7)?,
                code_score: Some(code_score),
                description_score: Some(description_score),
                fused_score: fuse_scores(
                    code_score,
                    description_score,
                    score_space,
                    fuse_method,
                    code_weight,
                    description_weight,
                ),
                normalized_code: row.get(10)?,
                description: row.get(11)?,
            })
        },
    )?;

    let mut results = rows.collect::<std::result::Result<Vec<_>, _>>()?;
    sort_best_first_results(&mut results, score_space);
    results.truncate(limit);
    Ok(results)
}

fn load_similarity_items(
    conn: &Connection,
    table: &str,
    kinds: &[String],
) -> Result<Vec<SimilarityItem>> {
    let kind_filter = kind_filter_sql(kinds, "symbols.kind");
    let sql = format!(
        "SELECT
           symbols.id,
           symbols.file_path,
           symbols.start_line,
           symbols.kind,
           symbols.name,
           symbols.parent_name,
           symbols.normalized_code,
           symbols.description,
           vec.embedding
         FROM {table} AS vec
         JOIN symbols ON symbols.id = vec.rowid
         WHERE 1 = 1
         {kind_filter}"
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let bytes: Vec<u8> = row.get(8)?;
        let embedding = try_cast_slice::<u8, f32>(&bytes)
            .map(|slice| slice.to_vec())
            .map_err(|err| {
                rusqlite::Error::FromSqlConversionFailure(
                    bytes.len(),
                    rusqlite::types::Type::Blob,
                    Box::new(err),
                )
            })?;

        Ok(SimilarityItem {
            rowid: row.get(0)?,
            file_path: row.get(1)?,
            start_line: row.get(2)?,
            kind: row.get(3)?,
            name: row.get(4)?,
            parent_name: row.get(5)?,
            normalized_code: row.get(6)?,
            description: row.get(7)?,
            embedding,
        })
    })?;

    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn kind_filter_sql(kinds: &[String], column: &str) -> String {
    if kinds.is_empty() {
        String::new()
    } else {
        let quoted = kinds
            .iter()
            .map(|kind| format!("'{}'", kind.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(", ");
        format!("AND {column} IN ({quoted})")
    }
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

struct PairScore {
    code_score: Option<f64>,
    description_score: Option<f64>,
    fused_score: f64,
}

fn pair_score(
    left_code: &SimilarityItem,
    right_code: &SimilarityItem,
    left_desc: Option<&SimilarityItem>,
    right_desc: Option<&SimilarityItem>,
    using: SearchUsing,
    score_space: ScoreSpace,
    fuse_method: FuseMethod,
    code_weight: f64,
    description_weight: f64,
) -> Option<PairScore> {
    let code_similarity = cosine_similarity(&left_code.embedding, &right_code.embedding) as f64;
    let code_score = score_from_similarity(code_similarity, score_space);
    match using {
        SearchUsing::Code => Some(PairScore {
            code_score: Some(code_score),
            description_score: None,
            fused_score: code_score,
        }),
        SearchUsing::Description => {
            let left_desc = left_desc?;
            let right_desc = right_desc?;
            let desc_similarity =
                cosine_similarity(&left_desc.embedding, &right_desc.embedding) as f64;
            let desc_score = score_from_similarity(desc_similarity, score_space);
            Some(PairScore {
                code_score: None,
                description_score: Some(desc_score),
                fused_score: desc_score,
            })
        }
        SearchUsing::Fused => {
            let left_desc = left_desc?;
            let right_desc = right_desc?;
            let desc_similarity =
                cosine_similarity(&left_desc.embedding, &right_desc.embedding) as f64;
            let desc_score = score_from_similarity(desc_similarity, score_space);
            Some(PairScore {
                code_score: Some(code_score),
                description_score: Some(desc_score),
                fused_score: fuse_scores(
                    code_score,
                    desc_score,
                    score_space,
                    fuse_method,
                    code_weight,
                    description_weight,
                ),
            })
        }
    }
}

fn score_from_distance(distance: f64, score_space: ScoreSpace) -> f64 {
    match score_space {
        ScoreSpace::Distance => distance,
        ScoreSpace::Similarity => 1.0 / (1.0 + distance),
    }
}

fn score_from_similarity(similarity: f64, score_space: ScoreSpace) -> f64 {
    match score_space {
        ScoreSpace::Distance => 1.0 - similarity,
        ScoreSpace::Similarity => similarity,
    }
}

fn fuse_scores(
    code_score: f64,
    description_score: f64,
    score_space: ScoreSpace,
    fuse_method: FuseMethod,
    code_weight: f64,
    description_weight: f64,
) -> f64 {
    match fuse_method {
        FuseMethod::Sum => code_score + description_score,
        FuseMethod::Max => match score_space {
            ScoreSpace::Distance => code_score.min(description_score),
            ScoreSpace::Similarity => code_score.max(description_score),
        },
        FuseMethod::Min => match score_space {
            ScoreSpace::Distance => code_score.max(description_score),
            ScoreSpace::Similarity => code_score.min(description_score),
        },
        FuseMethod::Weighted => {
            let total_weight = code_weight + description_weight;
            if total_weight == 0.0 {
                code_score + description_score
            } else {
                (code_score * code_weight + description_score * description_weight) / total_weight
            }
        }
    }
}

fn sort_best_first_results(results: &mut [SearchResult], score_space: ScoreSpace) {
    match score_space {
        ScoreSpace::Distance => results.sort_by(|a, b| a.fused_score.total_cmp(&b.fused_score)),
        ScoreSpace::Similarity => results.sort_by(|a, b| b.fused_score.total_cmp(&a.fused_score)),
    }
}

fn sort_best_first_pairs(results: &mut [SimilarPair], score_space: ScoreSpace) {
    match score_space {
        ScoreSpace::Distance => results.sort_by(|a, b| a.fused_score.total_cmp(&b.fused_score)),
        ScoreSpace::Similarity => results.sort_by(|a, b| b.fused_score.total_cmp(&a.fused_score)),
    }
}
