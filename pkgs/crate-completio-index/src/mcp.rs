use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::db::SearchResult;
use crate::indexer;
use crate::{FuseMethod, ScoreSpace, SearchUsing};

const SERVER_NAME: &str = "completio-index";
const SERVER_TITLE: &str = "Search the current codebase for relevant code snippets";
const MCP_DB_PATH: &str = ".completio/index.sqlite";

#[derive(Debug, Deserialize)]
struct SearchCodeArgs {
    query: String,
    #[serde(default = "default_db_path")]
    db_path: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

#[derive(Serialize)]
struct ToolDef<'a> {
    name: &'a str,
    description: &'a str,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

#[derive(Debug, Serialize)]
struct SearchCodeHit {
    file_path: String,
    line: i64,
    kind: String,
    name: String,
    parent_name: Option<String>,
}

pub fn run_stdio() -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(err) => {
                write_json(
                    &mut stdout,
                    &json_rpc_error(Value::Null, -32700, &err.to_string()),
                )?;
                continue;
            }
        };

        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let method = request
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();

        if method.starts_with("notifications/") && request.get("id").is_none() {
            continue;
        }

        let response = match handle_request(
            method,
            request.get("params").cloned().unwrap_or(Value::Null),
        ) {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err(err) => json_rpc_error(id, -32000, &err.to_string()),
        };
        write_json(&mut stdout, &response)?;
    }

    Ok(())
}

fn handle_request(method: &str, params: Value) -> Result<Value> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2025-06-18",
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": {
                "name": SERVER_NAME,
                "title": SERVER_TITLE,
                "version": env!("CARGO_PKG_VERSION")
            }
        })),
        "notifications/initialized" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tools() })),
        "tools/call" => call_tool(params),
        _ => Err(anyhow!("unsupported method: {method}")),
    }
}

fn call_tool(params: Value) -> Result<Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .context("missing tool name")?;
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    match name {
        "search_code" => {
            let args: SearchCodeArgs = serde_json::from_value(arguments)?;
            let result = indexer::query_data(
                PathBuf::from(args.db_path),
                args.query,
                args.limit,
                SearchUsing::Description,
                ScoreSpace::Distance,
                FuseMethod::Sum,
                1.0,
                1.0,
                Vec::new(),
            )?;
            let compact = result.into_iter().map(compact_hit).collect::<Vec<_>>();
            ok_tool_result("results", &compact)
        }
        _ => Err(anyhow!("unknown tool: {name}")),
    }
}

fn compact_hit(result: SearchResult) -> SearchCodeHit {
    SearchCodeHit {
        file_path: result.file_path,
        line: result.start_line,
        kind: result.kind,
        name: result.name,
        parent_name: result.parent_name,
    }
}

fn ok_tool_result<T: Serialize>(field_name: &str, value: &T) -> Result<Value> {
    let structured = json!({ field_name: value });
    Ok(json!({
        "content": [{"type": "text", "text": serde_json::to_string_pretty(&structured)?}],
        "structuredContent": structured,
        "isError": false
    }))
}

fn tools() -> Vec<ToolDef<'static>> {
    vec![ToolDef {
        name: "search_code",
        description: "Search the current codebase by meaning using vector search over code descriptions, and return the most relevant matching code snippets.",
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the code you want to find in the current codebase."
                },
                "db_path": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["query"]
        }),
    }]
}

fn json_rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })
}

fn write_json(stdout: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *stdout, value)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn default_db_path() -> String {
    MCP_DB_PATH.to_string()
}

fn default_limit() -> usize {
    5
}
