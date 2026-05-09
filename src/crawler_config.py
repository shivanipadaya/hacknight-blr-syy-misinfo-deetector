import argparse
from pathlib import Path

import yaml
from rich.console import Console

from src.settings import ROOT_DIR


console = Console()


def load_sources(path: Path | None = None) -> list[dict]:
    source_path = path or ROOT_DIR / "config" / "sources.yaml"
    with source_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["sources"]


def write_seed_urls(output_path: Path) -> None:
    sources = load_sources()
    lines = [source["url"] for source in sources]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"Wrote {len(lines)} seed URLs to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate crawler helper files.")
    parser.add_argument(
        "--seed-output",
        default=str(ROOT_DIR / "config" / "crawler_seed_urls.txt"),
        help="Path to write one seed URL per line.",
    )
    args = parser.parse_args()
    write_seed_urls(Path(args.seed_output))


if __name__ == "__main__":
    main()
