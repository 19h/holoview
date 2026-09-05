fn main() -> anyhow::Result<()> {
    let root = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("Usage: prepare_terrain CACHE_DIR"))?;
    let start = std::time::Instant::now();
    holographic_viewer::data::terrain::upgrade_terrain(std::path::Path::new(&root))?;
    println!("Navigation terrain prepared in {:.2} s", start.elapsed().as_secs_f64());
    Ok(())
}
