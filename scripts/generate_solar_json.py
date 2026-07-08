#!/usr/bin/env python3
"""
Generate JSON catalogues for the interactive solar-neighbourhood web map.

This script is the bridge between the curated Python/SIMBAD workflow used for
figures and the browser-side Three.js atlas. It resolves field stars, open
cluster centres and named cluster members, converts them to Galactic Cartesian
coordinates, and writes JSON files consumed by assets/js/solar-map.js.

Outputs:
  assets/data/bright_stars.json
  assets/data/systems.json

Default behaviour:
  - Resolve object coordinates with SIMBAD when internet access is available.
  - Use SIMBAD parallax when available; otherwise fall back to the seed distance.
  - For named cluster members, compute offset_pc = member_xyz - cluster_xyz.
  - Unresolved named cluster members are skipped by default, so the web map does
    not show fake catalogue stars. Use --allow-visual-fallback only for demos.

Optional Gaia mode:
  --with-gaia-members adds a Gaia DR3 cone-query supplement for bright candidate
  cluster stars. This is a visual candidate selection, not a rigorous membership
  analysis. For a publication-grade cluster model, replace this with a proper
  membership table and feed it through --members.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.simbad import Simbad
    HAS_ASTROQUERY = True
except Exception as exc:  # pragma: no cover - user-facing runtime guard
    u = None
    SkyCoord = None
    Simbad = None
    HAS_ASTROQUERY = False
    print(
        "[WARN] astropy/astroquery are not available. Manual RA/Dec/l,b seeds will still be converted, "
        "but online SIMBAD resolution will be skipped. Install with:\n"
        "  python -m pip install -r requirements-data.txt\n"
        f"Original import error: {exc}",
        file=sys.stderr,
    )

# ICRS(J2000) -> Galactic rotation matrix. Used as a dependency-light fallback
# and also keeps manual seed conversion reproducible.
ICRS_TO_GAL = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [ 0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669],
])


@dataclass
class ResolvedObject:
    name: str
    ra_deg: float
    dec_deg: float
    distance_pc: float | None
    l_deg: float
    b_deg: float
    X_pc: float
    Y_pc: float
    Z_pc: float
    vmag: float | None = None
    spectral_type: str | None = None
    source: str = "SIMBAD"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "--"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def coord_to_xyz(ra_deg: float, dec_deg: float, distance_pc: float) -> tuple[float, float, float, float, float]:
    if HAS_ASTROQUERY and SkyCoord is not None:
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, distance=distance_pc * u.pc, frame="icrs")
        gal = coord.galactic
        cart = gal.cartesian
        return (
            float(gal.l.deg),
            float(gal.b.deg),
            float(cart.x.to_value(u.pc)),
            float(cart.y.to_value(u.pc)),
            float(cart.z.to_value(u.pc)),
        )

    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    vec_icrs = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    x, y, z = ICRS_TO_GAL @ vec_icrs * float(distance_pc)
    l = math.degrees(math.atan2(y, x)) % 360.0
    b = math.degrees(math.atan2(z, math.hypot(x, y)))
    return float(l), float(b), float(x), float(y), float(z)


def manual_object_from_row(row: dict[str, str], *, display_name: str | None = None) -> ResolvedObject | None:
    ra = parse_float(row.get("ra_deg"))
    dec = parse_float(row.get("dec_deg"))
    d = parse_float(row.get("distance_pc"))
    if ra is None or dec is None or d is None:
        return None
    l_deg, b_deg, X_pc, Y_pc, Z_pc = coord_to_xyz(ra, dec, d)
    return ResolvedObject(
        name=display_name or row.get("name", "object"),
        ra_deg=ra, dec_deg=dec, distance_pc=d,
        l_deg=l_deg, b_deg=b_deg, X_pc=X_pc, Y_pc=Y_pc, Z_pc=Z_pc,
        vmag=parse_float(row.get("vmag")), spectral_type=(row.get("spectral_type") or None),
        source="manual_seed",
    )


def finite_or_none(value: Any) -> float | None:
    val = parse_float(value)
    return val if val is not None and math.isfinite(val) else None


def configure_simbad() -> Simbad | None:
    if not HAS_ASTROQUERY or Simbad is None:
        return None
    simbad = Simbad()
    # The astroquery SIMBAD interface changed names over time; try the modern
    # fields first and keep going if one is unavailable.
    for field in ("ra(d)", "dec(d)", "plx", "flux(V)", "sp_type"):
        try:
            simbad.add_votable_fields(field)
        except Exception:
            pass
    return simbad


def row_get_case_insensitive(row: Any, *names: str) -> Any:
    if row is None:
        return None
    keys = {str(k).lower().replace("_", ""): k for k in getattr(row, "colnames", [])}
    if not keys and hasattr(row, "columns"):
        keys = {str(k).lower().replace("_", ""): k for k in row.columns}
    for name in names:
        key = name.lower().replace("_", "")
        if key in keys:
            return row[keys[key]]
    # Astropy Row has keys but not always colnames
    try:
        row_keys = {str(k).lower().replace("_", ""): k for k in row.keys()}
        for name in names:
            key = name.lower().replace("_", "")
            if key in row_keys:
                return row[row_keys[key]]
    except Exception:
        pass
    return None


def resolve_simbad_object(
    simbad: Simbad,
    query_name: str,
    *,
    display_name: str | None = None,
    fallback_distance_pc: float | None = None,
    fallback_vmag: float | None = None,
    fallback_spectral_type: str | None = None,
) -> ResolvedObject | None:
    if simbad is None:
        return None
    try:
        table = simbad.query_object(query_name)
    except Exception as exc:
        print(f"[WARN] SIMBAD failed for {query_name}: {exc}")
        return None
    if table is None or len(table) == 0:
        print(f"[WARN] SIMBAD could not resolve {query_name}")
        return None

    row = table[0]
    ra = finite_or_none(row_get_case_insensitive(row, "RA_d", "RA"))
    dec = finite_or_none(row_get_case_insensitive(row, "DEC_d", "DEC"))

    # If RA/DEC arrive as strings, parse them as hourangle/degree.
    if ra is None or dec is None:
        ra_raw = row_get_case_insensitive(row, "RA")
        dec_raw = row_get_case_insensitive(row, "DEC")
        try:
            c = SkyCoord(str(ra_raw), str(dec_raw), unit=(u.hourangle, u.deg), frame="icrs")
            ra, dec = float(c.ra.deg), float(c.dec.deg)
        except Exception:
            print(f"[WARN] Could not parse coordinates for {query_name}")
            return None

    parallax_mas = finite_or_none(row_get_case_insensitive(row, "PLX_VALUE", "PLX"))
    distance_pc = fallback_distance_pc
    if parallax_mas is not None and parallax_mas > 0:
        distance_pc = 1000.0 / parallax_mas
    if distance_pc is None or not math.isfinite(distance_pc) or distance_pc <= 0:
        print(f"[WARN] No usable distance for {query_name}")
        return None

    vmag = finite_or_none(row_get_case_insensitive(row, "FLUX_V", "FLUX_V_VALUE"))
    if vmag is None:
        vmag = fallback_vmag
    spectral = row_get_case_insensitive(row, "SP_TYPE", "SP_TYPE_VALUE")
    spectral = str(spectral).strip() if spectral is not None and str(spectral).strip() not in {"--", ""} else fallback_spectral_type

    l_deg, b_deg, X_pc, Y_pc, Z_pc = coord_to_xyz(float(ra), float(dec), float(distance_pc))
    return ResolvedObject(
        name=display_name or query_name,
        ra_deg=float(ra),
        dec_deg=float(dec),
        distance_pc=float(distance_pc),
        l_deg=l_deg,
        b_deg=b_deg,
        X_pc=X_pc,
        Y_pc=Y_pc,
        Z_pc=Z_pc,
        vmag=vmag,
        spectral_type=spectral,
    )


def object_to_json(obj: ResolvedObject, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": obj.name,
        "distance_pc": round(obj.distance_pc or 0.0, 4),
        "ra_deg": round(obj.ra_deg, 7),
        "dec_deg": round(obj.dec_deg, 7),
        "l_deg": round(obj.l_deg, 7),
        "b_deg": round(obj.b_deg, 7),
        "X_pc": round(obj.X_pc, 5),
        "Y_pc": round(obj.Y_pc, 5),
        "Z_pc": round(obj.Z_pc, 5),
        "source": obj.source,
    }
    if obj.vmag is not None:
        out["vmag"] = round(float(obj.vmag), 4)
    if obj.spectral_type:
        out["spectral_type"] = str(obj.spectral_type)
    if extra:
        out.update(extra)
    return out


def generate_bright_stars(simbad: Simbad, seeds: list[dict[str, str]]) -> list[dict[str, Any]]:
    stars: list[dict[str, Any]] = []
    for row in seeds:
        name = row["name"].strip()
        query_name = row.get("query_name", name).strip() or name
        obj = manual_object_from_row(row, display_name=row.get("label") or name)
        if obj is None:
            obj = resolve_simbad_object(
                simbad,
                query_name,
                display_name=row.get("label") or name,
                fallback_distance_pc=parse_float(row.get("distance_pc")),
                fallback_vmag=parse_float(row.get("vmag")),
                fallback_spectral_type=row.get("spectral_type") or None,
            )
        if obj is None:
            continue
        stars.append(object_to_json(obj, {"constellation": row.get("constellation", "")}))
    stars.sort(key=lambda s: s.get("vmag", 99))
    return stars


def deterministic_offset(seed: str, scale: Iterable[float]) -> list[float]:
    # Fallback used only when a named member cannot be resolved. The output is
    # stable, small and visibly inside the cluster.
    h = 2166136261
    for ch in seed:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    vals = []
    for i, radius in enumerate(scale):
        h = (h * 1664525 + 1013904223 + i) & 0xFFFFFFFF
        vals.append(((h / 0xFFFFFFFF) * 2 - 1) * float(radius) * 0.28)
    return [round(v, 4) for v in vals]


def generate_systems(
    simbad: Simbad,
    cluster_seeds: list[dict[str, str]],
    member_seeds: list[dict[str, str]],
    *,
    allow_visual_fallback: bool = False,
) -> list[dict[str, Any]]:
    members_by_cluster: dict[str, list[dict[str, str]]] = {}
    for m in member_seeds:
        members_by_cluster.setdefault(m["cluster"].strip(), []).append(m)

    systems: list[dict[str, Any]] = []
    for cseed in cluster_seeds:
        name = cseed["name"].strip()
        query_name = cseed.get("query_name", name).strip() or name
        fallback_distance = parse_float(cseed.get("distance_pc"))
        center = manual_object_from_row(cseed, display_name=name)
        if center is None:
            center = resolve_simbad_object(
                simbad,
                query_name,
            display_name=name,
                fallback_distance_pc=fallback_distance,
            )

        # Cluster centres often do not have useful SIMBAD parallaxes. If SIMBAD
        # resolves the sky position but not a distance, use the curated distance.
        if center is None and parse_float(cseed.get("l_deg")) is not None and parse_float(cseed.get("b_deg")) is not None and fallback_distance:
            l = float(parse_float(cseed["l_deg"]))
            b = float(parse_float(cseed["b_deg"]))
            d = float(fallback_distance)
            lr = math.radians(l); br = math.radians(b)
            X = d * math.cos(br) * math.cos(lr)
            Y = d * math.cos(br) * math.sin(lr)
            Z = d * math.sin(br)
            # RA/Dec are metadata only in this branch. Without astropy we leave
            # them as 0.0; the map uses the Galactic Cartesian coordinates.
            if HAS_ASTROQUERY and SkyCoord is not None:
                coord = SkyCoord(l=l * u.deg, b=b * u.deg, distance=d * u.pc, frame="galactic").icrs
                ra_meta, dec_meta = float(coord.ra.deg), float(coord.dec.deg)
            else:
                ra_meta, dec_meta = 0.0, 0.0
            center = ResolvedObject(name, ra_meta, dec_meta, d, l, b, X, Y, Z, source="seed_lbd")

        if center is None:
            continue

        radii = [
            parse_float(cseed.get("radius_x_pc"), 8.0) or 8.0,
            parse_float(cseed.get("radius_y_pc"), 6.0) or 6.0,
            parse_float(cseed.get("radius_z_pc"), 5.0) or 5.0,
        ]

        resolved_members: list[dict[str, Any]] = []
        pending_member_names: list[str] = []
        for mseed in members_by_cluster.get(name, []):
            mname = mseed["name"].strip()
            query = mseed.get("query_name", mname).strip() or mname
            member = manual_object_from_row(mseed, display_name=mname)
            if member is None:
                member = resolve_simbad_object(
                    simbad,
                    query,
                    display_name=mname,
                    fallback_distance_pc=center.distance_pc,
                    fallback_vmag=parse_float(mseed.get("vmag")),
                    fallback_spectral_type=mseed.get("spectral_type") or None,
                )
            if member is not None:
                offset = [
                    member.X_pc - center.X_pc,
                    member.Y_pc - center.Y_pc,
                    member.Z_pc - center.Z_pc,
                ]
                resolved_members.append({
                    "name": member.name,
                    "offset_pc": [round(v, 5) for v in offset],
                    "ra_deg": round(member.ra_deg, 7),
                    "dec_deg": round(member.dec_deg, 7),
                    "l_deg": round(member.l_deg, 7),
                    "b_deg": round(member.b_deg, 7),
                    "distance_pc": round(member.distance_pc or center.distance_pc or 0.0, 4),
                    "vmag": round(float(member.vmag), 4) if member.vmag is not None else parse_float(mseed.get("vmag")),
                    "spectral_type": member.spectral_type or mseed.get("spectral_type", ""),
                    "source": member.source,
                    "query_name": query,
                    "position_note": mseed.get("member_note") or "Resolved sky position; distance from SIMBAD parallax when available, otherwise cluster distance fallback.",
                })
            elif allow_visual_fallback:
                resolved_members.append({
                    "name": mname,
                    "offset_pc": deterministic_offset(name + mname, radii),
                    "vmag": parse_float(mseed.get("vmag"), 5.0),
                    "spectral_type": mseed.get("spectral_type", ""),
                    "source": "visual_fallback",
                    "position_note": "Fallback visual offset; not a resolved catalogue coordinate.",
                })
            else:
                pending_member_names.append(mname)
                print(f"[WARN] Skipping unresolved member {mname} in {name}; no fake named marker written.")

        total_seed_members = len(members_by_cluster.get(name, []))
        system_status = (
            "catalogue_named_members" if resolved_members and not pending_member_names
            else "partial_catalogue_named_members" if resolved_members
            else "catalogue_members_pending"
        )
        systems.append({
            "name": name,
            "kind": cseed.get("kind", "open_cluster") or "open_cluster",
            "age_myr": parse_float(cseed.get("age_myr")),
            "distance_pc": round(center.distance_pc or 0.0, 4),
            "ra_deg": round(center.ra_deg, 7),
            "dec_deg": round(center.dec_deg, 7),
            "l_deg": round(center.l_deg, 7),
            "b_deg": round(center.b_deg, 7),
            "X_pc": round(center.X_pc, 5),
            "Y_pc": round(center.Y_pc, 5),
            "Z_pc": round(center.Z_pc, 5),
            "radius_pc": [round(float(v), 3) for v in radii],
            "star_count": int(parse_float(cseed.get("star_count"), 120) or 120),
            "description": cseed.get("description", ""),
            "source": center.source,
            "member_model_status": system_status,
            "member_seed_count": total_seed_members,
            "pending_member_names": pending_member_names,
            "members": resolved_members,
        })

    return systems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root directory")
    parser.add_argument("--stars", type=Path, default=None, help="CSV seed list for bright field stars")
    parser.add_argument("--clusters", type=Path, default=None, help="CSV seed list for open clusters")
    parser.add_argument("--members", type=Path, default=None, help="CSV seed list for named cluster members")
    parser.add_argument("--dry-run", action="store_true", help="Resolve objects but do not overwrite JSON files")
    parser.add_argument(
        "--allow-visual-fallback",
        action="store_true",
        help="Write deterministic fake offsets for unresolved named cluster members. Off by default so named members are never silently fake.",
    )
    args = parser.parse_args()

    root = args.root
    source_dir = root / "assets" / "data_sources"
    data_dir = root / "assets" / "data"
    stars_csv = args.stars or source_dir / "bright_stars_seed.csv"
    clusters_csv = args.clusters or source_dir / "open_clusters_seed.csv"
    members_csv = args.members or source_dir / "cluster_members_seed.csv"

    simbad = configure_simbad()
    bright = generate_bright_stars(simbad, read_csv(stars_csv))
    systems = generate_systems(
        simbad,
        read_csv(clusters_csv),
        read_csv(members_csv),
        allow_visual_fallback=args.allow_visual_fallback,
    )

    existing_bright = root / "assets" / "data" / "bright_stars.json"
    if not bright and existing_bright.exists():
        print("[WARN] No bright field stars were resolved; preserving the existing bright_stars.json catalogue.")
        bright = json.loads(existing_bright.read_text(encoding="utf-8"))

    print(f"Resolved {len(bright)} bright field stars")
    print(f"Resolved {len(systems)} open clusters")
    print(f"Resolved {sum(len(s.get('members', [])) for s in systems)} named cluster members")

    if args.dry_run:
        print(json.dumps({"stars": bright[:3], "systems": systems[:1]}, indent=2, ensure_ascii=False))
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    regions_path = data_dir / "regions.json"
    regions = json.loads(regions_path.read_text(encoding="utf-8")) if regions_path.exists() else []
    (data_dir / "bright_stars.json").write_text(json.dumps(bright, indent=2, ensure_ascii=False), encoding="utf-8")
    (data_dir / "systems.json").write_text(json.dumps(systems, indent=2, ensure_ascii=False), encoding="utf-8")

    # The current prototype loads a browser bundle rather than fetching JSON,
    # which keeps it compatible with very simple static hosting. Keep it in
    # sync whenever the generated data are refreshed.
    bundle = {"stars": bright, "systems": systems, "regions": regions}
    bundle_text = "window.SOLAR_MAP_DATA = " + json.dumps(bundle, indent=2, ensure_ascii=False) + ";\n"
    (root / "assets" / "js" / "solar-data.js").write_text(bundle_text, encoding="utf-8")

    print(f"Wrote {data_dir / 'bright_stars.json'}")
    print(f"Wrote {data_dir / 'systems.json'}")
    print(f"Wrote {root / 'assets/js/solar-data.js'}")


if __name__ == "__main__":
    main()
