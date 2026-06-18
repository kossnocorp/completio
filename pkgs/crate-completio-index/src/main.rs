mod db;
mod extract;
mod gateway;
mod indexer;
mod mcp;
mod normalize;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Index and search definitions in the current codebase, with CLI and MCP access."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Index(IndexArgs),
    Query(QueryArgs),
    Similar(SimilarArgs),
    Graph(GraphArgs),
    Mcp,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchUsing {
    Code,
    Description,
    Fused,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScoreSpace {
    Distance,
    Similarity,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FuseMethod {
    Sum,
    Max,
    Min,
    Weighted,
}

#[derive(Args, Clone)]
struct CommonArgs {
    #[arg(long, default_value = ".completio/index.sqlite")]
    db_path: PathBuf,
}

#[derive(Args)]
struct IndexArgs {
    #[command(flatten)]
    common: CommonArgs,
    project_path: PathBuf,
    #[arg(long, default_value_t = false)]
    reindex_all: bool,
    #[arg(long, default_value_t = 4)]
    threads: usize,
}

#[derive(Args)]
struct QueryArgs {
    #[command(flatten)]
    common: CommonArgs,
    query: String,
    #[arg(long, default_value_t = 10)]
    limit: usize,
    #[arg(long, value_enum, default_value_t = SearchUsing::Description)]
    using: SearchUsing,
    #[arg(long, value_enum, default_value_t = ScoreSpace::Distance)]
    score_space: ScoreSpace,
    #[arg(long, value_enum, default_value_t = FuseMethod::Sum)]
    fuse_method: FuseMethod,
    #[arg(long, default_value_t = 1.0)]
    code_weight: f64,
    #[arg(long, default_value_t = 1.0)]
    description_weight: f64,
    #[arg(long, value_delimiter = ',')]
    kinds: Vec<String>,
}

#[derive(Args)]
struct SimilarArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value_t = 10)]
    limit: usize,
    #[arg(long, default_value_t = 20)]
    neighbors: usize,
    #[arg(long, value_enum, default_value_t = SearchUsing::Description)]
    using: SearchUsing,
    #[arg(long, value_enum, default_value_t = ScoreSpace::Similarity)]
    score_space: ScoreSpace,
    #[arg(long, value_enum, default_value_t = FuseMethod::Sum)]
    fuse_method: FuseMethod,
    #[arg(long, default_value_t = 1.0)]
    code_weight: f64,
    #[arg(long, default_value_t = 1.0)]
    description_weight: f64,
    #[arg(long, value_delimiter = ',')]
    kinds: Vec<String>,
}

#[derive(Args)]
struct GraphArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value = "graph.json")]
    out: PathBuf,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 0.72)]
    min_score: f64,
    #[arg(long, default_value_t = true)]
    cross_file_only: bool,
    #[arg(long, value_delimiter = ',')]
    kinds: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Index(args) => indexer::run_index(
            args.common.db_path,
            args.project_path,
            args.reindex_all,
            args.threads,
        ),
        Command::Query(args) => indexer::run_query(
            args.common.db_path,
            args.query,
            args.limit,
            args.using,
            args.score_space,
            args.fuse_method,
            args.code_weight,
            args.description_weight,
            args.kinds,
        ),
        Command::Similar(args) => indexer::run_similar(
            args.common.db_path,
            args.limit,
            args.neighbors,
            args.using,
            args.score_space,
            args.fuse_method,
            args.code_weight,
            args.description_weight,
            args.kinds,
        ),
        Command::Graph(args) => indexer::run_graph(
            args.common.db_path,
            args.out,
            args.k,
            args.min_score,
            args.cross_file_only,
            args.kinds,
        ),
        Command::Mcp => mcp::run_stdio(),
    }
}
