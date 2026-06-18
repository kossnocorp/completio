use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

pub const EMBEDDING_MODEL: &str = "openai/text-embedding-3-large";
pub const EMBEDDING_DIMENSIONS: usize = 3072;
pub const CHAT_MODEL: &str = "gpt-5-nano";

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: String,
}

pub fn embed_text(input: &str) -> Result<Vec<f32>> {
    let payload = serde_json::json!({
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "input": input,
    });
    let body = post_json("https://ai-gateway.vercel.sh/v1/embeddings", &payload)?;
    let response: EmbeddingsResponse =
        serde_json::from_str(&body).context("failed to decode embedding response")?;
    let embedding = response
        .data
        .into_iter()
        .next()
        .map(|datum| datum.embedding)
        .context("embedding response contained no vectors")?;

    if embedding.len() != EMBEDDING_DIMENSIONS {
        bail!(
            "expected embedding with {} dims, got {}",
            EMBEDDING_DIMENSIONS,
            embedding.len()
        );
    }

    Ok(embedding)
}

pub fn describe_code(kind: &str, name: &str, code: &str) -> Result<String> {
    let payload = serde_json::json!({
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You explain source code in one concise paragraph. Focus on behavior and intent, not line-by-line narration."
            },
            {
                "role": "user",
                "content": format!(
                    "Explain what this code does in a paragraph.\n\nKind: {kind}\nName: {name}\n\n```\n{code}\n```"
                )
            }
        ]
    });
    let body = post_json("https://ai-gateway.vercel.sh/v1/chat/completions", &payload)?;
    let response: ChatResponse =
        serde_json::from_str(&body).context("failed to decode chat response")?;
    let description = response
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.message.content.trim().to_string())
        .filter(|content| !content.is_empty())
        .context("chat response contained no text")?;
    Ok(description)
}

fn post_json(url: &str, payload: &serde_json::Value) -> Result<String> {
    let command = format!(
        "curl -sS {url} -H \"Authorization: Bearer $AI_GATEWAY_API_KEY\" -H \"Content-Type: application/json\" --data-binary @-"
    );

    let mut child = Command::new("mise")
        .arg("exec")
        .arg("fnox")
        .arg("--")
        .arg("fnox")
        .arg("exec")
        .arg("--")
        .arg("sh")
        .arg("-lc")
        .arg(command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn fnox via mise for gateway request")?;

    child
        .stdin
        .as_mut()
        .context("missing stdin for gateway child")?
        .write_all(payload.to_string().as_bytes())
        .context("failed to write gateway payload")?;

    let output = child
        .wait_with_output()
        .context("failed to wait for gateway request")?;
    if !output.status.success() {
        bail!(
            "gateway request failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let body = String::from_utf8(output.stdout).context("gateway response was not utf-8")?;
    if body.trim_start().starts_with('{') {
        if let Ok(error_value) = serde_json::from_str::<serde_json::Value>(&body) {
            if let Some(message) = error_value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(|message| message.as_str())
            {
                return Err(anyhow!("gateway API error: {message}"));
            }
        }
    }

    Ok(body)
}
