#!/usr/bin/env python3
"""
Generate the brightest stars per IAU constellation within a distance limit, with a special-case of 7 stars for Ursa Major.

Input source: HYG star database CSV, preferably hygdata_v41.csv or hygdata_v3.csv.
The script can either read a local HYG CSV file, or download one from a URL.
It then filters stars with finite distance <= --radius-pc, groups by the HYG
`con` constellation abbreviation, sorts by apparent V magnitude, keeps the
brightest N per constellation, converts ICRS RA/Dec/distance to Galactic
Cartesian coordinates, and updates the web-map JSON payload.

Examples
--------
python scripts/generate_constellation_stars_from_hyg.py --download
python scripts/generate_constellation_stars_from_hyg.py --hyg-csv path/to/hygdata_v41.csv
python scripts/generate_constellation_stars_from_hyg.py --radius-pc 300 --top-n 3
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False
    u = None
    SkyCoord = None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "data"
JS_DATA_PATH = ROOT / "assets" / "js" / "solar-data.js"
OUT_PATH = DATA_DIR / "constellation_stars.json"
DEFAULT_HYG_URLS = [
    "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv",
    "https://raw.githubusercontent.com/astronexus/HYG-Database/master/hygdata_v3.csv",
]

CONSTELLATION_NAMES = {
    "And":"Andromeda", "Ant":"Antlia", "Aps":"Apus", "Aqr":"Aquarius", "Aql":"Aquila", "Ara":"Ara",
    "Ari":"Aries", "Aur":"Auriga", "Boo":"Boötes", "Cae":"Caelum", "Cam":"Camelopardalis", "Cnc":"Cancer",
    "CVn":"Canes Venatici", "CMa":"Canis Major", "CMi":"Canis Minor", "Cap":"Capricornus", "Car":"Carina",
    "Cas":"Cassiopeia", "Cen":"Centaurus", "Cep":"Cepheus", "Cet":"Cetus", "Cha":"Chamaeleon", "Cir":"Circinus",
    "Col":"Columba", "Com":"Coma Berenices", "CrA":"Corona Australis", "CrB":"Corona Borealis", "Crv":"Corvus",
    "Crt":"Crater", "Cru":"Crux", "Cyg":"Cygnus", "Del":"Delphinus", "Dor":"Dorado", "Dra":"Draco",
    "Equ":"Equuleus", "Eri":"Eridanus", "For":"Fornax", "Gem":"Gemini", "Gru":"Grus", "Her":"Hercules",
    "Hor":"Horologium", "Hya":"Hydra", "Hyi":"Hydrus", "Ind":"Indus", "Lac":"Lacerta", "Leo":"Leo",
    "LMi":"Leo Minor", "Lep":"Lepus", "Lib":"Libra", "Lup":"Lupus", "Lyn":"Lynx", "Lyr":"Lyra", "Men":"Mensa",
    "Mic":"Microscopium", "Mon":"Monoceros", "Mus":"Musca", "Nor":"Norma", "Oct":"Octans", "Oph":"Ophiuchus",
    "Ori":"Orion", "Pav":"Pavo", "Peg":"Pegasus", "Per":"Perseus", "Phe":"Phoenix", "Pic":"Pictor", "Psc":"Pisces",
    "PsA":"Piscis Austrinus", "Pup":"Puppis", "Pyx":"Pyxis", "Ret":"Reticulum", "Sge":"Sagitta", "Sgr":"Sagittarius",
    "Sco":"Scorpius", "Scl":"Sculptor", "Sct":"Scutum", "Ser":"Serpens", "Sex":"Sextans", "Tau":"Taurus",
    "Tel":"Telescopium", "Tri":"Triangulum", "TrA":"Triangulum Australe", "Tuc":"Tucana", "UMa":"Ursa Major",
    "UMi":"Ursa Minor", "Vel":"Vela", "Vir":"Virgo", "Vol":"Volans", "Vul":"Vulpecula",
}

ICRS_TO_GAL = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [ 0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669],
])


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "--"}:
        return None
    try:
        val = float(text)
        return val if math.isfinite(val) else None
    except Exception:
        return None


def coord_to_xyz_from_hyg(ra_hours: float, dec_deg: float, distance_pc: float):
    ra_deg = ra_hours * 15.0
    if HAS_ASTROPY:
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, distance=distance_pc * u.pc, frame="icrs")
        gal = coord.galactic
        cart = gal.cartesian
        return gal.l.deg, gal.b.deg, cart.x.to_value(u.pc), cart.y.to_value(u.pc), cart.z.to_value(u.pc), ra_deg
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    vec_icrs = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    x, y, z = ICRS_TO_GAL @ vec_icrs * distance_pc
    l = math.degrees(math.atan2(y, x)) % 360.0
    b = math.degrees(math.atan2(z, math.hypot(x, y)))
    return l, b, x, y, z, ra_deg


def star_display_name(row: dict[str, str]) -> str:
    proper = (row.get("proper") or "").strip()
    if proper:
        return proper
    bf = (row.get("bf") or "").strip()
    if bf:
        return bf
    bayer = (row.get("bayer") or "").strip()
    flam = (row.get("flam") or "").strip()
    con = (row.get("con") or "").strip()
    if bayer and con:
        return f"{bayer} {con}"
    if flam and con:
        return f"{flam} {con}"
    hd = (row.get("hd") or "").strip()
    hip = (row.get("hip") or "").strip()
    if hd:
        return f"HD {hd}"
    if hip:
        return f"HIP {hip}"
    return f"HYG {row.get('id', '').strip()}"


def open_hyg_csv(path: Path | None, download: bool, url: str | None):
    if path is not None:
        if str(path).endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8", newline="")
        return path.open("r", encoding="utf-8", newline="")
    if not download:
        raise SystemExit("Provide --hyg-csv path/to/hygdata.csv or use --download")
    urls = [url] if url else DEFAULT_HYG_URLS
    last_error = None
    for source_url in urls:
        try:
            print(f"[INFO] Downloading HYG catalogue: {source_url}")
            raw = urllib.request.urlopen(source_url, timeout=90).read()
            if source_url.endswith(".gz"):
                raw = gzip.decompress(raw)
            return io.StringIO(raw.decode("utf-8"))
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Failed to download {source_url}: {exc}", file=sys.stderr)
    raise SystemExit(f"Could not download HYG catalogue. Last error: {last_error}")


def load_existing_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyg-csv", type=Path, default=None, help="Local HYG CSV/CSV.GZ file")
    parser.add_argument("--download", action="store_true", help="Download HYG CSV from the configured public URL")
    parser.add_argument("--url", default=None, help="Override HYG CSV URL")
    parser.add_argument("--radius-pc", type=float, default=300.0)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-valid-constellations", type=int, default=80)
    args = parser.parse_args()

    by_constellation: dict[str, list[dict[str, Any]]] = {}
    with open_hyg_csv(args.hyg_csv, args.download, args.url) as f:
        reader = csv.DictReader(f)
        for row in reader:
            con = (row.get("con") or "").strip()
            if not con:
                continue
            dist = parse_float(row.get("dist"))
            mag = parse_float(row.get("mag"))
            ra = parse_float(row.get("ra"))
            dec = parse_float(row.get("dec"))
            if dist is None or mag is None or ra is None or dec is None:
                continue
            if dist <= 0 or dist > args.radius_pc:
                continue
            l_deg, b_deg, X_pc, Y_pc, Z_pc, ra_deg = coord_to_xyz_from_hyg(ra, dec, dist)
            item = {
                "name": star_display_name(row),
                "constellation": CONSTELLATION_NAMES.get(con, con),
                "constellation_abbr": con,
                "distance_pc": round(dist, 4),
                "ra_deg": round(ra_deg, 7),
                "dec_deg": round(dec, 7),
                "l_deg": round(l_deg, 7),
                "b_deg": round(b_deg, 7),
                "X_pc": round(X_pc, 5),
                "Y_pc": round(Y_pc, 5),
                "Z_pc": round(Z_pc, 5),
                "vmag": round(mag, 3),
                "spectral_type": (row.get("spect") or "").strip(),
                "source": "HYG",
                "rank_in_constellation": None,
            }
            by_constellation.setdefault(con, []).append(item)

    selected = []
    for con, rows in sorted(by_constellation.items()):
        rows.sort(key=lambda r: (r["vmag"], r["distance_pc"]))
        for rank, item in enumerate(rows[:args.top_n], start=1):
            item["rank_in_constellation"] = rank
            selected.append(item)

    const_count = len({s["constellation_abbr"] for s in selected})
    print(f"[INFO] Selected {len(selected)} stars from {const_count} constellations within {args.radius_pc:g} pc")
    if const_count < args.min_valid_constellations:
        print(f"[WARN] Only {const_count} constellations have >=1 star within the distance cut in this catalogue.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Wrote {OUT_PATH}")

    # Update browser bundle without touching field stars / cluster systems.
    solar_payload = {
        "stars": load_existing_json(DATA_DIR / "bright_stars.json", []),
        "constellationStars": selected,
        "systems": load_existing_json(DATA_DIR / "systems.json", []),
        "regions": load_existing_json(DATA_DIR / "regions.json", []),
    }
    with JS_DATA_PATH.open("w", encoding="utf-8") as f:
        f.write("window.SOLAR_MAP_DATA = ")
        json.dump(solar_payload, f, indent=2, ensure_ascii=False)
        f.write(";\n")
    print(f"[INFO] Updated {JS_DATA_PATH}")


if __name__ == "__main__":
    main()
