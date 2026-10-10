"""Release artifacts: export a trained checkpoint, or fetch the released one.

python -m src.artifacts export outputs/runs/baseline/best.pt checkpoints/model.pt --url <release-url>
python -m src.artifacts fetch      # downloads checkpoints/model.pt using checkpoints/RELEASE.json
"""
from __future__ import annotations

import argparse
import json

from src.generate import DEFAULT_CHECKPOINT, RELEASE_FILE, resolve_checkpoint
from src.utils import export_model, sha256_file


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="strip optimizer state and write RELEASE.json")
    e.add_argument("src")
    e.add_argument("dst", nargs="?", default=DEFAULT_CHECKPOINT)
    e.add_argument("--url", default=None, help="where the file will be published")
    sub.add_parser("fetch", help="download and verify the released model")
    args = p.parse_args()
    if args.cmd == "export":
        digest = export_model(args.src, args.dst)
        print(f"{args.dst}  sha256={digest}")
        if args.url:
            with open(RELEASE_FILE, "w", encoding="utf-8") as f:
                json.dump({"url": args.url, "sha256": digest}, f, indent=2)
            print(f"wrote {RELEASE_FILE}")
    else:
        path = resolve_checkpoint(None)
        print(f"{path}  sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
