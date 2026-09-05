use anyhow::{Context, Result};
use std::path::Path;
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<_> = std::env::args().collect();
    let source = args.get(1).context("Usage: prepare_city SOURCE_HYPC_DIR CACHE_DIR [WORKERS=4]")?;
    let cache = args.get(2).context("Missing cache directory")?;
    let workers = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(4);
    holographic_viewer::data::dataset::prepare_dataset(Path::new(source), Path::new(cache), workers)?;
    Ok(())
}
