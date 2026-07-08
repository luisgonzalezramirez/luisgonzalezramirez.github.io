#!/usr/bin/env python3
"""
Update only the bright-star layer of the 400 pc web map.

This wrapper intentionally preserves the cloud/surface/map layers. It runs the
normal exporter in a temporary copy of the project, extracts only the newly
generated bright-star layer, and then writes only:

  - assets/data/bright_stars.json
  - the `stars` field inside assets/js/solar-data.js

Because the full exporter runs in a temporary copy, your real `assets/data/`
folder is never deleted or regenerated. This avoids OneDrive/Windows
PermissionError issues and prevents molecular-cloud/dust layers from being
changed.

Examples:
  python scripts/export_stars_only.py --star-vmag-max 4
  python scripts/export_stars_only.py --star-vmag-max 4 --refresh-all-sky-star-cache
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PREFIX = 'window.SOLAR_MAP_DATA = '


def load_solar_data(path: Path) -> dict:
    text = path.read_text(encoding='utf-8').strip()
    if text.startswith(PREFIX):
        text = text[len(PREFIX):]
    if text.endswith(';'):
        text = text[:-1]
    return json.loads(text)


def write_solar_data(path: Path, data: dict) -> None:
    path.write_text(PREFIX + json.dumps(data, indent=2, ensure_ascii=False) + ';\n', encoding='utf-8')


def copytree_ignore_junk(src: Path, dst: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        junk = {
            '__pycache__', '.git', '.star_only_backup_tmp',
            '.DS_Store', 'Thumbs.db'
        }
        return {name for name in names if name in junk or name.endswith('.tmp')}
    shutil.copytree(src, dst, ignore=ignore)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--star-vmag-max', type=float, default=4.0)
    ap.add_argument('--radius-pc', type=float, default=400.0)
    ap.add_argument('--refresh-all-sky-star-cache', action='store_true')
    ap.add_argument('--no-resolve-simbad-stars', action='store_true')
    ap.add_argument('--no-all-sky-simbad-stars', action='store_true')
    ap.add_argument('--respect-yaml-star-exclusions', action='store_true')
    args = ap.parse_args()

    root = args.root.resolve()
    data_dir = root / 'assets' / 'data'
    js_path = root / 'assets' / 'js' / 'solar-data.js'
    exporter_rel = Path('scripts') / 'export_400pc_web_json.py'
    exporter = root / exporter_rel

    if not exporter.exists():
        raise SystemExit(f'Exporter not found: {exporter}')
    if not js_path.exists():
        raise SystemExit(f'SOLAR data JS not found: {js_path}')
    if not data_dir.exists():
        raise SystemExit(f'Data directory not found: {data_dir}')

    # Preserve the real/current non-star SOLAR_MAP_DATA payload.
    current_solar = load_solar_data(js_path)

    # Run the destructive/full exporter only inside a disposable temporary copy.
    with tempfile.TemporaryDirectory(prefix='stars_only_export_') as tmp:
        tmp_root = Path(tmp) / 'web_copy'
        copytree_ignore_junk(root, tmp_root)
        tmp_exporter = tmp_root / exporter_rel
        tmp_data_dir = tmp_root / 'assets' / 'data'

        cmd = [
            sys.executable, str(tmp_exporter),
            '--root', str(tmp_root),
            '--radius-pc', str(args.radius_pc),
            '--star-vmag-max', str(args.star_vmag_max),
        ]
        if args.refresh_all_sky_star_cache:
            cmd.append('--refresh-all-sky-star-cache')
        if args.no_resolve_simbad_stars:
            cmd.append('--no-resolve-simbad-stars')
        if args.no_all_sky_simbad_stars:
            cmd.append('--no-all-sky-simbad-stars')
        if args.respect_yaml_star_exclusions:
            cmd.append('--respect-yaml-star-exclusions')

        subprocess.run(cmd, check=True)
        new_stars = json.loads((tmp_data_dir / 'bright_stars.json').read_text(encoding='utf-8'))

    # Write only the bright-star layer in the real project.
    (data_dir / 'bright_stars.json').write_text(json.dumps(new_stars, indent=2, ensure_ascii=False), encoding='utf-8')
    current_solar['stars'] = new_stars
    write_solar_data(js_path, current_solar)

    print(f'Updated stars only: {len(new_stars)} bright stars')
    print('Preserved cloud/surface layers and all non-star map data in the real project.')


if __name__ == '__main__':
    main()
