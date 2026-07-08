#!/usr/bin/env python3
"""
Export the 400 pc thesis-map data into the Three.js web-map bundle.

This version is deliberately browser-safe: the browser never reads HDF5/HTML
source files directly. Instead this script preprocesses the scientific inputs
used by solar_map_400pc.py:

  - solar_map_400pc_data.yaml
  - handbook_distances.html
  - map3D_GAIAdr2_feb2019.h5  (optional; large, not bundled by default)

and writes normal JSON files consumed by assets/js/solar-map.js.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
except Exception:  # pragma: no cover
    SkyCoord = None
    u = None
    Table = None

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

try:
    from astroquery.simbad import Simbad
except Exception:  # pragma: no cover
    Simbad = None

# SciPy is optional and is imported lazily inside the H5 exporter.  This avoids
# noisy SciPy/Numpy version warnings at script startup in base conda environments.
gaussian_filter = None
ndi_label = None


def get_ndimage_tools():
    global gaussian_filter, ndi_label
    if gaussian_filter is not None and ndi_label is not None:
        return gaussian_filter, ndi_label
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"A NumPy version >=.* is required for this version of SciPy.*",
                category=UserWarning,
            )
            from scipy.ndimage import gaussian_filter as _gaussian_filter
            from scipy.ndimage import label as _ndi_label
        gaussian_filter = _gaussian_filter
        ndi_label = _ndi_label
    except Exception:
        gaussian_filter = None
        ndi_label = None
    return gaussian_filter, ndi_label

# Fixed ICRS(J2000) -> Galactic rotation matrix.  We use this by default instead
# of astropy for scalar RA/Dec conversions because old astropy builds can fail
# with NumPy >= 1.26 (TypeError: concatenate() got an unexpected keyword
# argument 'dtype').  This keeps the exporter robust in base conda envs.
ICRS_TO_GAL = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [ 0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669],
], dtype=float)


def clean_label(text: Any) -> str:
    out = str(text)
    out = out.replace('$\x07lpha$', r'$\alpha$').replace('$\x08eta$', r'$\beta$')
    return (out.replace(r'$\alpha$', 'α')
               .replace(r'$\beta$', 'β')
               .replace(r'$\eta$', 'η')
               .replace(r'$\epsilon$', 'ε')
               .replace(r'$\rho$', 'ρ')
               .replace('$', ''))


def fval(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        val = float(value)
        return val if math.isfinite(val) else default
    except Exception:
        return default


GREEK_TOKEN_MAP = {
    'α': 'alpha', 'alf': 'alpha', 'alp': 'alpha', 'alpha': 'alpha',
    'β': 'beta', 'bet': 'beta', 'beta': 'beta',
    'γ': 'gamma', 'gam': 'gamma', 'gamma': 'gamma',
    'δ': 'delta', 'del': 'delta', 'delta': 'delta',
    'ε': 'epsilon', 'eps': 'epsilon', 'epsilon': 'epsilon',
    'ζ': 'zeta', 'zet': 'zeta', 'zeta': 'zeta',
    'η': 'eta', 'eta': 'eta',
    'θ': 'theta', 'tet': 'theta', 'theta': 'theta',
    'ι': 'iota', 'iot': 'iota', 'iota': 'iota',
    'κ': 'kappa', 'kap': 'kappa', 'kappa': 'kappa',
    'λ': 'lambda', 'lam': 'lambda', 'lambda': 'lambda',
    'μ': 'mu', 'mu': 'mu',
    'ν': 'nu', 'nu': 'nu',
    'ξ': 'xi', 'ksi': 'xi', 'xi': 'xi',
    'ο': 'omicron', 'omi': 'omicron', 'omicron': 'omicron',
    'π': 'pi', 'pi': 'pi',
    'ρ': 'rho', 'rho': 'rho',
    'σ': 'sigma', 'sig': 'sigma', 'sigma': 'sigma',
    'τ': 'tau', 'tau': 'tau',
    'υ': 'upsilon', 'ups': 'upsilon', 'upsilon': 'upsilon',
    'φ': 'phi', 'phi': 'phi',
    'χ': 'chi', 'chi': 'chi',
    'ψ': 'psi', 'psi': 'psi',
    'ω': 'omega', 'ome': 'omega', 'omega': 'omega',
}

# Common-name aliases needed to prevent all-sky SIMBAD objects from duplicating
# the curated YAML/web objects with slightly different coordinates or IDs.
COMMON_STAR_CANONICAL = {
    'sirius': 'alpha_cma', 'canopus': 'alpha_car', 'arcturus': 'alpha_boo',
    'vega': 'alpha_lyr', 'capella': 'alpha_aur', 'rigel': 'beta_ori',
    'procyon': 'alpha_cmi', 'betelgeuse': 'alpha_ori', 'altair': 'alpha_aql',
    'aldebaran': 'alpha_tau', 'spica': 'alpha_vir', 'antares': 'alpha_sco',
    'pollux': 'beta_gem', 'fomalhaut': 'alpha_psa', 'regulus': 'alpha_leo',
    'castor': 'alpha_gem', 'bellatrix': 'gamma_ori', 'alioth': 'epsilon_uma',
    'dubhe': 'alpha_uma', 'mirfak': 'alpha_per', 'deneb': 'alpha_cyg',
    'mizar': 'zeta_uma', 'alkaid': 'eta_uma', 'merak': 'beta_uma',
    'phecda': 'gamma_uma', 'megrez': 'delta_uma', 'achernar': 'alpha_eri',
    'acrux': 'alpha_cru', 'gacrux': 'gamma_cru', 'shaula': 'lambda_sco',
    'denebola': 'beta_leo', 'alphecca': 'alpha_crb', 'alfeca': 'alpha_crb',
    'alhena': 'gamma_gem', 'peacock': 'alpha_pav', 'alfirk': 'beta_cep',
    'alderamin': 'alpha_cep', 'errai': 'gamma_cep', 'algol': 'beta_per',
    'sargas': 'theta_sco', 'kaus australis': 'epsilon_sgr', 'nunki': 'sigma_sgr',
    'alnair': 'alpha_gru', 'alphard': 'alpha_hya', 'diphdha': 'beta_cet',
    'schedar': 'alpha_cas', 'rasalhague': 'alpha_oph', 'cebalrai': 'beta_oph', 'gomeisa': 'beta_cmi', 'miaplacidus': 'beta_car', 'kausaustralis': 'epsilon_sgr', 'kaus_australis': 'epsilon_sgr', 'miram': 'eta_per', 'alpheratz': 'alpha_and', 'sirrah': 'alpha_and', 'mirach': 'beta_and', 'almach': 'gamma_and',
}


def canonical_star_key(value: Any) -> str:
    s = clean_label(value).lower()
    s = s.replace('name ', '').replace('ids ', '')
    s = s.replace('*', ' ').replace('$', ' ')
    s = s.replace('01', ' 1 ').replace('02', ' 2 ').replace('1 ', ' ').replace('2 ', ' ')
    for ch in ',;:/()[]{}+-':
        s = s.replace(ch, ' ')
    parts = []
    for tok in s.split():
        tok = tok.strip().strip('.')
        if not tok or tok in {'the', 'star'}:
            continue
        parts.append(GREEK_TOKEN_MAP.get(tok, tok))
    # Convert SIMBAD greek-style names such as "* alf Aur" -> alpha_aur.
    if len(parts) >= 2 and parts[0] in set(GREEK_TOKEN_MAP.values()):
        key = f'{parts[0]}_{parts[1]}'
    else:
        key = '_'.join(parts)
    return COMMON_STAR_CANONICAL.get(key, key)


def star_position_tuple(row: dict) -> tuple[float, float, float] | None:
    try:
        return (float(row['X_pc']), float(row['Y_pc']), float(row['Z_pc']))
    except Exception:
        return None


def is_same_physical_star(candidate: dict, accepted: list[dict], accepted_keys: set[str]) -> bool:
    ckeys = star_alias_keys(candidate)
    if ckeys & accepted_keys:
        return True
    cpos = star_position_tuple(candidate)
    if cpos is None:
        return False
    cv = fval(candidate.get('vmag'), 99.0) or 99.0
    for row in accepted:
        rpos = star_position_tuple(row)
        if rpos is None:
            continue
        sep = math.sqrt(sum((cpos[i] - rpos[i])**2 for i in range(3)))
        # Same source with a different SIMBAD/common-name identifier.  Keep real
        # close binaries when magnitudes differ substantially, but remove exact
        # duplicate objects such as Sirius / * alf CMa or Alioth / * eps UMa.
        rv = fval(row.get('vmag'), 99.0) or 99.0
        if sep < 0.35 and abs(cv - rv) < 0.85:
            return True
    return False


def star_alias_keys(row: dict) -> set[str]:
    keys = set()
    for field in ('name', 'label', 'catalog_name', 'simbad_name', 'main_id'):
        value = row.get(field) if isinstance(row, dict) else None
        if value:
            keys.add(canonical_star_key(value))
    return {k for k in keys if k}


def add_preferred_star(out: list[dict], keys: set[str], row: dict, *, replace_allsky_duplicate: bool = False) -> bool:
    if is_same_physical_star(row, out, keys):
        if replace_allsky_duplicate:
            # Prefer curated YAML/web metadata/label over an all-sky SIMBAD alias.
            rpos = star_position_tuple(row)
            rv = fval(row.get('vmag'), 99.0) or 99.0
            for i, old in enumerate(list(out)):
                opos = star_position_tuple(old)
                if opos is None or rpos is None:
                    continue
                sep = math.sqrt(sum((rpos[j] - opos[j])**2 for j in range(3)))
                ov = fval(old.get('vmag'), 99.0) or 99.0
                if sep < 0.35 and abs(rv - ov) < 0.85:
                    out[i] = row
                    keys.update(star_alias_keys(row))
                    return True
        return False
    out.append(row)
    keys.update(star_alias_keys(row))
    return True


def dedupe_by_label(rows: list[dict], key: str = 'label') -> list[dict]:
    out, seen = [], set()
    for row in rows:
        k = clean_label(row.get(key, row.get('name', '')))
        if k in seen:
            continue
        seen.add(k)
        out.append(row)
    return out


def collect_cluster_star_exclusions(systems: list[dict]) -> tuple[set[str], list[tuple[tuple[float,float,float], float]]]:
    member_aliases: set[str] = set()
    cluster_zones: list[tuple[tuple[float,float,float], float]] = []
    for sysobj in systems:
        if sysobj.get('kind') not in {'open_cluster', 'association', 'sco_cen'}:
            continue
        try:
            center = (float(sysobj['X_pc']), float(sysobj['Y_pc']), float(sysobj['Z_pc']))
            rr = sysobj.get('radius_pc', [0,0,0])
            if isinstance(rr, (list, tuple)) and rr:
                rad = float(max(rr))
            else:
                rad = float(rr or 0)
            cluster_zones.append((center, max(4.0, 1.35 * rad + 2.0)))
        except Exception:
            pass
        for mem in sysobj.get('members', []) or []:
            member_aliases.update(star_alias_keys(mem))
            nm = mem.get('name')
            if nm:
                member_aliases.add(canonical_star_key(nm))
    return member_aliases, cluster_zones


def is_in_cluster_zone(row: dict, member_aliases: set[str], cluster_zones: list[tuple[tuple[float,float,float], float]]) -> bool:
    if star_alias_keys(row) & member_aliases:
        return True
    pos = star_position_tuple(row)
    if pos is None:
        return False
    for center, radius in cluster_zones:
        sep = math.sqrt(sum((pos[i] - center[i])**2 for i in range(3)))
        if sep <= radius:
            return True
    return False


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f'Missing YAML file: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise SystemExit(f'Unexpected YAML content: {path}')
    return data


def gal_from_radec(ra_deg: float, dec_deg: float, distance_pc: float) -> dict[str, float]:
    # Browser/export-safe scalar conversion, independent of astropy runtime.
    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    d = float(distance_pc)
    icrs = np.array([
        math.cos(dec) * math.cos(ra),
        math.cos(dec) * math.sin(ra),
        math.sin(dec),
    ], dtype=float)
    gal = ICRS_TO_GAL @ icrs
    x_unit, y_unit, z_unit = [float(v) for v in gal]
    l = math.degrees(math.atan2(y_unit, x_unit)) % 360.0
    b = math.degrees(math.asin(max(-1.0, min(1.0, z_unit))))
    return {
        'l_deg': l,
        'b_deg': b,
        'X_pc': d * x_unit,
        'Y_pc': d * y_unit,
        'Z_pc': d * z_unit,
    }


def gal_from_lbd(l_deg: float, b_deg: float, distance_pc: float) -> dict[str, float]:
    l = math.radians(float(l_deg)); b = math.radians(float(b_deg)); d = float(distance_pc)
    return {
        'l_deg': float(l_deg), 'b_deg': float(b_deg),
        'X_pc': d * math.cos(b) * math.cos(l),
        'Y_pc': d * math.cos(b) * math.sin(l),
        'Z_pc': d * math.sin(b),
    }


def coords_for_row(row: dict, existing: dict | None = None) -> dict[str, float] | None:
    d = fval(row.get('distance_pc'), None)
    if d is None:
        return None
    # Existing curated web coordinates win for objects already fixed by hand.
    if existing and all(k in existing for k in ('X_pc', 'Y_pc', 'Z_pc')):
        out = {k: fval(existing.get(k), 0.0) or 0.0 for k in ('X_pc','Y_pc','Z_pc')}
        out['l_deg'] = fval(existing.get('l_deg'), fval(row.get('l_deg'), 0.0)) or 0.0
        out['b_deg'] = fval(existing.get('b_deg'), fval(row.get('b_deg'), 0.0)) or 0.0
        return out
    if 'ra_deg' in row and 'dec_deg' in row:
        return gal_from_radec(row['ra_deg'], row['dec_deg'], d)
    if 'l_deg' in row and 'b_deg' in row:
        return gal_from_lbd(row['l_deg'], row['b_deg'], d)
    return None


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return default


def system_radius(label: str, row: dict, yaml_data: dict) -> list[float]:
    spec = yaml_data.get('SYSTEM_SKY_REGION_SPECS', {}).get(label)
    d = fval(row.get('distance_pc'), 100.0) or 100.0
    kind = str(row.get('kind', 'open_cluster'))
    if spec:
        a = max(2.2, d * math.radians(float(spec.get('a_deg', 4.0))))
        b = max(1.8, d * math.radians(float(spec.get('b_deg', 2.0))))
        return [round(a, 3), round(b, 3), round(max(1.5, min(a, b) * 0.65), 3)]
    if label == 'Pleiades': return [5.2, 3.6, 2.8]
    if label == 'Hyades': return [16.0, 10.0, 8.0]
    if kind == 'sco_cen': return [42.0, 24.0, 18.0]
    if kind == 'association': return [34.0, 20.0, 14.0]
    return [10.0, 6.5, 5.0]


def keep_existing_members(label: str, existing_systems: list[dict], n: int = 4) -> list[dict]:
    # Preserve the web-tuned members for every cluster that already has them.
    for sysobj in existing_systems:
        if clean_label(sysobj.get('name')) == label:
            members = list(sysobj.get('members', []))
            members.sort(key=lambda m: fval(m.get('vmag'), 99.0) or 99.0)
            return members[:n]
    return []

def table_row_get(row, *names, default=None):
    try:
        colnames = list(getattr(row, 'colnames', [])) or list(getattr(getattr(row, 'table', None), 'colnames', []))
    except Exception:
        colnames = []
    lower = {str(c).lower(): c for c in colnames}
    for name in names:
        key = lower.get(str(name).lower())
        if key is None:
            continue
        try:
            val = row[key]
            if val is not None and str(val) not in {'--', 'nan', 'None'}:
                return val
        except Exception:
            pass
    return default


def parse_simbad_angle_pair(row):
    ra_val = table_row_get(row, 'RA', 'ra', 'ra_deg', 'RA_d')
    dec_val = table_row_get(row, 'DEC', 'dec', 'dec_deg', 'DEC_d')
    if ra_val is None or dec_val is None:
        return None

    def _scalar(x):
        try:
            if hasattr(x, 'value'):
                x = x.value
            return float(x)
        except Exception:
            try:
                s = str(x).strip()
                # Newer astroquery/SIMBAD configurations can return decimal degrees
                # as strings.  The old exporter only understood sexagesimal text,
                # so all those successful queries were silently discarded.
                if len(s.replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).split()) == 1:
                    return float(s)
            except Exception:
                pass
        return None

    ra_num = _scalar(ra_val)
    dec_num = _scalar(dec_val)
    if ra_num is not None and dec_num is not None:
        # If RA is a plain decimal value in [0, 360], interpret as degrees.
        # If someone explicitly gives hours in [0, 24] but decimal, the ambiguity
        # is rare for SIMBAD table output; SIMBAD decimal RA is degrees.
        if 0.0 <= ra_num <= 360.0 and -90.0 <= dec_num <= 90.0:
            return float(ra_num), float(dec_num)

    ra_txt = str(ra_val).strip()
    dec_txt = str(dec_val).strip()
    try:
        # Classic SIMBAD output: RA in h m s and DEC in d m s.
        ra_parts = [float(x) for x in ra_txt.replace(':', ' ').split()[:3]]
        dec_parts = [float(x) for x in dec_txt.replace('+','').replace('-','').replace(':',' ').split()[:3]]
        if len(ra_parts) == 1 and len(dec_parts) == 1:
            return float(ra_parts[0]), float(dec_parts[0])
        if len(ra_parts) < 3 or len(dec_parts) < 3:
            return None
        hh, mm, ss = ra_parts
        sign = -1.0 if dec_txt.startswith('-') else 1.0
        dd, dm, ds = dec_parts
        ra_deg = 15.0 * (hh + mm/60.0 + ss/3600.0)
        dec_deg = sign * (dd + dm/60.0 + ds/3600.0)
        return ra_deg, dec_deg
    except Exception:
        return None


SIMBAD_NAME_ALIASES = {
    # Common star names in the YAML that SIMBAD does not always resolve as-is.
    # Keep labels in the exported JSON, but query SIMBAD with stable Bayer/SIMBAD ids.
    'Cih': ['* gam Cas', 'gam Cas', 'Gamma Cas', 'NAME Cih'],
    'Girtab': ['* kap Sco', 'kap Sco', 'Kappa Sco', 'NAME Girtab'],
    'Gienah Cyg': ['* eps Cyg', 'eps Cyg', 'Epsilon Cyg', 'NAME Gienah Cyg'],
    'Nash': ['* gam02 Sgr', '* gam Sgr', 'gam02 Sgr', 'Gamma2 Sgr', 'NAME Alnasl', 'NAME Nash'],
}

GREEK_TO_SIMBAD = {
    'α': 'alf', 'β': 'bet', 'γ': 'gam', 'δ': 'del', 'ε': 'eps', 'ζ': 'zet',
    'η': 'eta', 'θ': 'tet', 'ι': 'iot', 'κ': 'kap', 'λ': 'lam', 'μ': 'mu',
    'ν': 'nu', 'ξ': 'ksi', 'ο': 'omi', 'π': 'pi', 'ρ': 'rho', 'σ': 'sig',
    'τ': 'tau', 'υ': 'ups', 'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'ome',
}


def candidate_simbad_names(row: dict) -> list[str]:
    label = clean_label(row.get('label', row.get('name', ''))).strip()
    raw = [row.get('simbad_name'), row.get('name'), row.get('label'), label]
    out = []

    def add(value):
        if value is None:
            return
        cand = str(value).strip()
        if cand and cand not in out:
            out.append(cand)

    # Hard aliases first: these fix the four currently failing stars.
    for alias in SIMBAD_NAME_ALIASES.get(label, []):
        add(alias)

    for value in raw:
        if not value:
            continue
        add(value)
        cleaned = clean_label(value).strip()
        add(cleaned)
        # SIMBAD often likes Bayer identifiers in its abbreviated notation.
        bayer = cleaned
        for greek, simbad_abbr in GREEK_TO_SIMBAD.items():
            bayer = bayer.replace(greek, simbad_abbr)
        add(bayer)
        add(f'NAME {cleaned}')
        add(f'* {bayer}')
    return out


def load_bright_star_cache(root: Path, radius_pc: float, vmag_max: float, excluded: set[str]) -> list[dict]:
    if Table is None:
        return []
    paths = [
        root / 'cache' / 'bright_named_stars_400pc_v3.ecsv',
        root / 'cache' / 'bright_named_stars_400pc.ecsv',
        root / 'bright_named_stars_400pc_v3.ecsv',
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            tab = Table.read(path, format='ascii.ecsv')
        except Exception:
            continue
        out = []
        for row in tab:
            label = clean_label(row['label'] if 'label' in tab.colnames else row['name'])
            if label in excluded:
                continue
            d = fval(row['distance_pc'] if 'distance_pc' in tab.colnames else None, None)
            vmag = fval(row['vmag'] if 'vmag' in tab.colnames else None, 99.0)
            if d is None or d > radius_pc or vmag > vmag_max:
                continue
            if all(k in tab.colnames for k in ['X_pc','Y_pc','Z_pc']):
                coords = {'X_pc': float(row['X_pc']), 'Y_pc': float(row['Y_pc']), 'Z_pc': float(row['Z_pc'])}
                coords['l_deg'] = fval(row['l_deg'] if 'l_deg' in tab.colnames else None, 0.0) or 0.0
                coords['b_deg'] = fval(row['b_deg'] if 'b_deg' in tab.colnames else None, 0.0) or 0.0
            elif all(k in tab.colnames for k in ['ra_deg','dec_deg']):
                coords = gal_from_radec(float(row['ra_deg']), float(row['dec_deg']), d)
            else:
                continue
            out.append({
                'name': label,
                'catalog_name': str(row['name'] if 'name' in tab.colnames else label),
                'distance_pc': round(float(d), 4),
                'vmag': vmag,
                'spectral_type': str(row['spectral_type']) if 'spectral_type' in tab.colnames else '',
                'source': f'400 pc ECSV cache: {path.name}',
                **{k: round(float(v), 5 if k.endswith('_pc') else 7) for k, v in coords.items()},
            })
        return out
    return []


def resolve_simbad_positions(rows: list[dict], *, vmag_max: float = 4.2, radius_pc: float = 400.0) -> list[dict]:
    if Simbad is None:
        return []
    simbad = Simbad()
    try:
        simbad.add_votable_fields('V', 'sp')
    except Exception:
        pass
    out = []
    seen: set[str] = set()
    failed_names = []
    for row in rows:
        d = fval(row.get('distance_pc'), None)
        vmag = fval(row.get('vmag'), 99.0)
        if d is None or d > radius_pc or (vmag is not None and vmag > vmag_max):
            continue
        label = clean_label(row.get('label', row.get('name', '')))
        if not label or label in seen:
            continue
        seen.add(label)
        rec = None
        for qname in candidate_simbad_names(row):
            try:
                q = simbad.query_object(qname)
            except Exception:
                q = None
            if q is not None and len(q) > 0:
                rec = q[0]
                break
        if rec is None:
            failed_names.append(label)
            continue
        parsed = parse_simbad_angle_pair(rec)
        if parsed is None:
            failed_names.append(label)
            continue
        ra_deg, dec_deg = parsed
        coords = gal_from_radec(ra_deg, dec_deg, d)
        spec = str(table_row_get(rec, 'SP_TYPE', 'sp_type', default='') or '').strip()
        out.append({
            'name': label,
            'catalog_name': str(row.get('name', label)),
            'distance_pc': round(float(d), 4),
            'vmag': vmag,
            'spectral_type': spec,
            'source': 'SIMBAD via export_400pc_web_json.py',
            'constellation': row.get('constellation', ''),
            'ra_deg': round(ra_deg, 6),
            'dec_deg': round(dec_deg, 6),
            **{k: round(float(v), 5 if k.endswith('_pc') else 7) for k, v in coords.items()},
        })
    if rows:
        print(f"Bright-star SIMBAD: resolved {len(out)} / {len(rows)} pending stars", file=sys.stderr)
        if failed_names:
            print("Bright-star SIMBAD unresolved/parse-failed: " + ", ".join(failed_names[:20]) + ("…" if len(failed_names) > 20 else ""), file=sys.stderr)
    return out



def _sanitize_star_name(value: Any) -> str:
    name = clean_label(value)
    name = name.replace('NAME ', '').replace('  ', ' ').strip()
    name = name.replace('  ', ' ').strip()
    # SIMBAD main_id values can be bytes-like strings in older astropy tables.
    name = name.strip("b'").strip('"').strip()
    return name


def _row_to_float(row, *names, default=None):
    val = table_row_get(row, *names, default=default)
    return fval(val, default)


def query_simbad_all_sky_bright_stars(*, radius_pc: float, vmag_max: float, top: int = 10000) -> list[dict]:
    """Query SIMBAD for all stars with V <= vmag_max and parallax distance <= radius_pc.

    This is the magnitude-limited all-sky layer.  It deliberately does not rely
    on the curated YAML bright-star list.
    """
    if Simbad is None:
        print('All-sky SIMBAD bright-star query skipped: astroquery.simbad is unavailable', file=sys.stderr)
        return []
    if not hasattr(Simbad, 'query_tap'):
        print('All-sky SIMBAD bright-star query skipped: this astroquery version has no Simbad.query_tap', file=sys.stderr)
        return []

    plx_min_mas = 1000.0 / float(radius_pc)
    top = int(top)
    vlim = float(vmag_max)

    # Important ADQL detail for SIMBAD TAP:
    # ORDER BY table.column can fail in some TAP parser versions after a JOIN.
    # Use the selected alias `vmag` in ORDER BY instead.  Also try both the
    # `allfluxes` view and the measurement table `mesFlux`, because deployed
    # SIMBAD schemas differ slightly across astroquery/server versions.
    queries = [
        f"""
        SELECT TOP {top}
            b.main_id AS main_id,
            b.ra AS ra,
            b.dec AS dec,
            b.plx_value AS plx_value,
            b.sp_type AS sp_type,
            f."V" AS vmag
        FROM basic AS b
        JOIN allfluxes AS f ON b.oid = f.oidref
        WHERE f."V" <= {vlim}
          AND b.plx_value >= {plx_min_mas}
          AND b.plx_value IS NOT NULL
        ORDER BY vmag ASC
        """,
        f"""
        SELECT TOP {top}
            b.main_id AS main_id,
            b.ra AS ra,
            b.dec AS dec,
            b.plx_value AS plx_value,
            b.sp_type AS sp_type,
            f.V AS vmag
        FROM basic AS b
        JOIN allfluxes AS f ON b.oid = f.oidref
        WHERE f.V <= {vlim}
          AND b.plx_value >= {plx_min_mas}
          AND b.plx_value IS NOT NULL
        ORDER BY vmag ASC
        """,
        f"""
        SELECT TOP {top}
            basic.main_id AS main_id,
            basic.ra AS ra,
            basic.dec AS dec,
            basic.plx_value AS plx_value,
            basic.sp_type AS sp_type,
            mesFlux.flux AS vmag
        FROM basic
        JOIN mesFlux ON basic.oid = mesFlux.oidref
        WHERE mesFlux.filter = 'V'
          AND mesFlux.flux <= {vlim}
          AND basic.plx_value >= {plx_min_mas}
          AND basic.plx_value IS NOT NULL
        ORDER BY vmag ASC
        """,
    ]

    tab = None
    errors: list[str] = []
    for adql in queries:
        try:
            candidate = Simbad.query_tap(adql)
            if candidate is not None and len(candidate) > 0:
                tab = candidate
                break
        except Exception as exc:
            errors.append(str(exc).replace('\n', ' ')[:260])

    if tab is None or len(tab) == 0:
        if errors:
            print('All-sky SIMBAD bright-star query returned 0 rows. Tried 3 schema variants. Last error: ' + errors[-1], file=sys.stderr)
        else:
            print('All-sky SIMBAD bright-star query returned 0 rows.', file=sys.stderr)
        return []

    out = []
    seen = set()
    for row in tab:
        name = _sanitize_star_name(table_row_get(row, 'main_id', 'MAIN_ID', default=''))
        if not name or name in seen:
            continue
        vmag = _row_to_float(row, 'vmag', 'V', 'flux', 'FLUX', default=None)
        plx = _row_to_float(row, 'plx_value', 'PLX_VALUE', default=None)
        ra = _row_to_float(row, 'ra', 'RA', default=None)
        dec = _row_to_float(row, 'dec', 'DEC', default=None)
        if vmag is None or plx is None or plx <= 0 or ra is None or dec is None:
            continue
        d = 1000.0 / float(plx)
        if d > radius_pc or vmag > vmag_max:
            continue
        coords = gal_from_radec(float(ra), float(dec), d)
        spec = str(table_row_get(row, 'sp_type', 'SP_TYPE', default='') or '').strip()
        out.append({
            'name': name,
            'catalog_name': name,
            'distance_pc': round(float(d), 4),
            'vmag': round(float(vmag), 4),
            'spectral_type': spec,
            'source': f'SIMBAD all-sky query: V <= {vmag_max}, parallax distance <= {radius_pc} pc',
            'ra_deg': round(float(ra), 7),
            'dec_deg': round(float(dec), 7),
            **{k: round(float(v), 5 if k.endswith('_pc') else 7) for k, v in coords.items()},
        })
        seen.add(name)
    print(f'All-sky SIMBAD bright-star query: {len(out)} stars with V <= {vmag_max} and d <= {radius_pc} pc', file=sys.stderr)
    return out


def load_or_query_all_sky_bright_stars(root: Path | None, *, radius_pc: float, vmag_max: float, enabled: bool = True, force_refresh: bool = False) -> list[dict]:
    if not enabled:
        return []
    cache_path = None
    if root is not None:
        cache_dir = root / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f'simbad_allsky_bright_Vle{str(vmag_max).replace(".", "p")}_R{int(radius_pc)}pc.json'
        if cache_path.exists() and not force_refresh:
            try:
                rows = json.loads(cache_path.read_text(encoding='utf-8'))
                print(f'All-sky SIMBAD bright-star cache loaded: {len(rows)} stars from {cache_path}', file=sys.stderr)
                return rows
            except Exception:
                pass

    rows = query_simbad_all_sky_bright_stars(radius_pc=radius_pc, vmag_max=vmag_max)
    if cache_path is not None and rows:
        cache_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    return rows


def _blob_effective_radius(blob: dict) -> float:
    r = blob.get('radius_pc', [10,10,6])
    if isinstance(r, (list, tuple)) and len(r) >= 2:
        return float(max(1.0, 0.5 * (float(r[0]) + float(r[1]))))
    return float(max(1.0, fval(r, 8.0) or 8.0))


def merge_nearby_blobs(blobs: list[dict]) -> list[dict]:
    if not blobs:
        return []
    n = len(blobs)
    used = [False] * n
    merged = []
    centers = np.array([[float(b['X_pc']), float(b['Y_pc']), float(b['Z_pc'])] for b in blobs], dtype=float)
    radii = np.array([_blob_effective_radius(b) for b in blobs], dtype=float)
    for i in range(n):
        if used[i]:
            continue
        stack = [i]
        group = []
        used[i] = True
        while stack:
            a = stack.pop()
            group.append(a)
            for j in range(n):
                if used[j]:
                    continue
                d = float(np.linalg.norm(centers[a] - centers[j]))
                thr = radii[a] + radii[j] + min(radii[a], radii[j])
                if d <= thr:
                    used[j] = True
                    stack.append(j)
        pts = centers[group]
        ws = np.array([max(1.0, radii[g] ** 2) for g in group], dtype=float)
        ctr = np.average(pts, axis=0, weights=ws)
        deltas = np.abs(pts - ctr)
        base = np.array([radii[g] for g in group], dtype=float)
        rx = float(max(np.max(deltas[:,0] + base), np.mean(base) * 1.3, 6.0))
        ry = float(max(np.max(deltas[:,1] + base), np.mean(base) * 1.3, 6.0))
        rz = float(max(np.max(deltas[:,2] + 0.55*base), np.mean(base) * 0.9, 4.0))
        merged.append({
            'X_pc': round(float(ctr[0]), 3),
            'Y_pc': round(float(ctr[1]), 3),
            'Z_pc': round(float(ctr[2]), 3),
            'radius_pc': [round(rx,3), round(ry,3), round(rz,3)],
            'alpha': round(float(np.clip(0.012 + 0.0022 * len(group), 0.016, 0.048)), 4),
            'members_merged': len(group),
        })
    return merged


def sample_cloud_points_from_blobs(blobs: list[dict], *, max_points: int = 240, alpha_base: float = 0.06) -> list[dict]:
    # Create a true 3-D point-cloud representation from merged ellipsoidal blobs.
    usable = [b for b in blobs if star_position_tuple(b) is not None]
    if not usable:
        return []
    weights = []
    for b in usable:
        r = b.get('radius_pc', [10,10,6])
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            vol = max(1.0, float(r[0]) * float(r[1]) * float(r[2]))
        else:
            rr = _blob_effective_radius(b)
            vol = max(1.0, rr ** 3)
        weights.append(vol * max(0.6, float(b.get('members_merged', 1))))
    total = float(sum(weights)) or 1.0
    counts = [max(18, int(round(max_points * w / total))) for w in weights]
    while sum(counts) > max_points:
        idx = max(range(len(counts)), key=lambda i: counts[i])
        if counts[idx] <= 18:
            break
        counts[idx] -= 1
    while sum(counts) < max_points:
        idx = max(range(len(counts)), key=lambda i: weights[i])
        counts[idx] += 1
    pts: list[dict] = []
    for ib, (b, npts) in enumerate(zip(usable, counts)):
        rng = np.random.default_rng(12345 + ib * 997)
        cx, cy, cz = star_position_tuple(b)
        r = b.get('radius_pc', [10,10,6])
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            rx, ry, rz = max(2.0,float(r[0])), max(2.0,float(r[1])), max(1.5,float(r[2]))
        else:
            rr = _blob_effective_radius(b)
            rx = ry = rr; rz = max(1.5, 0.6 * rr)
        ang = math.radians(fval(b.get('angle_deg'), fval(b.get('angle'), 0.0)) or 0.0)
        ca, sa = math.cos(ang), math.sin(ang)
        made = 0
        trials = 0
        while made < npts and trials < npts * 20:
            trials += 1
            vec = rng.normal(size=3)
            norm = float(np.linalg.norm(vec)) or 1.0
            vec /= norm
            rad = rng.random() ** (1/3)
            x, y, z = vec[0] * rx * rad, vec[1] * ry * rad, vec[2] * rz * rad
            xr = x * ca - y * sa
            yr = x * sa + y * ca
            dens = max(0.25, 1.0 - 0.72 * ((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2))
            pts.append({
                'X_pc': round(cx + xr, 3),
                'Y_pc': round(cy + yr, 3),
                'Z_pc': round(cz + z, 3),
                'alpha': round(float(alpha_base * dens), 4),
            })
            made += 1
    return pts


def sample_cloud_points_from_voxel_component(pts: np.ndarray, strengths: np.ndarray, *, max_points: int = 260, alpha_scale: float = 0.085) -> list[dict]:
    if len(pts) == 0:
        return []
    n = min(int(max_points), len(pts))
    prob = np.asarray(strengths, dtype=float)
    prob = np.clip(prob, 0, None)
    if float(prob.sum()) <= 0:
        prob = None
    else:
        prob /= prob.sum()
    rng = np.random.default_rng(20260707 + len(pts))
    choose = rng.choice(len(pts), size=n, replace=len(pts) < n, p=prob)
    sel = pts[choose]
    selw = strengths[choose] if len(strengths) else np.ones(len(sel))
    swmax = float(np.max(selw)) if len(selw) else 1.0
    out = []
    for (x, y, z), w in zip(sel, selw):
        jitter = rng.normal(scale=[0.55, 0.55, 0.45], size=3)
        dens = float(w) / swmax if swmax > 0 else 0.5
        out.append({
            'X_pc': round(float(x + jitter[0]), 3),
            'Y_pc': round(float(y + jitter[1]), 3),
            'Z_pc': round(float(z + jitter[2]), 3),
            'alpha': round(float(np.clip(alpha_scale * (0.35 + 0.8 * dens), 0.025, 0.12)), 4),
        })
    return out


def build_systems(yaml_data: dict, existing_systems: list[dict], radius_pc: float, include_associations: bool, top_members: int) -> list[dict]:
    existing_by_name = {clean_label(s.get('name')): s for s in existing_systems}
    rows = dedupe_by_label(list(yaml_data.get('NEARBY_SYSTEMS', [])) + list(yaml_data.get('SYSTEMS_400PC_EXTRA', [])))
    systems: list[dict] = []
    for row in rows:
        label = clean_label(row.get('label', row.get('name', '')))
        kind = str(row.get('kind', 'open_cluster'))
        if not include_associations and kind != 'open_cluster':
            continue
        d = fval(row.get('distance_pc'), 9999.0) or 9999.0
        if d > radius_pc + 60:
            continue
        existing = existing_by_name.get(label)
        coords = coords_for_row(row, existing)
        if coords is None:
            # Keep no fake position. The user can regenerate with SIMBAD/cache later.
            continue
        members = keep_existing_members(label, existing_systems, top_members)
        sysobj = {
            'name': label,
            'catalog_name': str(row.get('name', label)),
            'kind': kind,
            'age_myr': fval(row.get('age_myr'), None),
            'distance_pc': round(d, 4),
            'ra_deg': fval(row.get('ra_deg'), fval(existing.get('ra_deg') if existing else None, 0.0)) if (row or existing) else 0.0,
            'dec_deg': fval(row.get('dec_deg'), fval(existing.get('dec_deg') if existing else None, 0.0)) if (row or existing) else 0.0,
            **{k: round(float(v), 5 if k.endswith('_pc') else 7) for k, v in coords.items()},
            'radius_pc': system_radius(label, row, yaml_data),
            'star_count': int((existing or {}).get('star_count', 120 if kind == 'open_cluster' else 70)),
            'source': 'solar_map_400pc_data.yaml + web curated members',
            'description': 'Imported from the 400 pc thesis YAML. Bright named members are preserved when already curated; the unresolved population is schematic.',
            'member_model_status': 'catalogue_named_members' if members else 'visual_population_only',
            'member_seed_count': len(members),
            'pending_member_names': [],
            'members': members,
        }
        systems.append(sysobj)
    systems.sort(key=lambda s: (s.get('kind',''), s.get('name','')))
    return systems


def merge_bright_stars(yaml_data: dict, existing_stars: list[dict], radius_pc: float, vmag_max: float = 4.2, resolve_simbad: bool = True, root: Path | None = None, all_sky_simbad: bool = True, force_all_sky_refresh: bool = False, respect_yaml_exclusions: bool = False, systems_for_exclusion: list[dict] | None = None) -> list[dict]:
    excluded = {clean_label(x) for x in yaml_data.get('EXCLUDED_BRIGHT_STAR_NAMES', [])} if respect_yaml_exclusions else set()
    out: list[dict] = []
    seen_keys: set[str] = set()
    existing_by_name = {clean_label(s.get('name')): dict(s) for s in existing_stars}
    yaml_rows = dedupe_by_label(list(yaml_data.get('BRIGHT_NAMED_STARS', [])) + list(yaml_data.get('BRIGHT_STARS_400PC_EXTRA', [])))
    member_aliases, cluster_zones = collect_cluster_star_exclusions(systems_for_exclusion or [])

    def passes(row: dict) -> bool:
        name = clean_label(row.get('name', row.get('label', '')))
        if name in excluded:
            return False
        if (fval(row.get('distance_pc'), 9999.0) or 9999.0) > radius_pc:
            return False
        if (fval(row.get('vmag'), 99.0) or 99.0) > vmag_max:
            return False
        return True

    # 1) Curated cache from the original Python map wins.
    if root is not None:
        for s in load_bright_star_cache(root, radius_pc, vmag_max, excluded):
            if passes(s):
                add_preferred_star(out, seen_keys, s)

    # 2) Existing web/YAML curated objects win over all-sky aliases. Do not
    # re-import a previously generated SIMBAD all-sky catalogue as if it were
    # curated data, otherwise old duplicates and cluster members persist forever.
    for s in existing_stars:
        if 'SIMBAD all-sky query' in str(s.get('source', '')):
            continue
        name = clean_label(s.get('name'))
        if not passes({**s, 'name': name}):
            continue
        ss = dict(s)
        ss['name'] = name
        if is_in_cluster_zone(ss, member_aliases, cluster_zones):
            continue
        ss.setdefault('source', 'web curated / previous SIMBAD seed')
        add_preferred_star(out, seen_keys, ss)

    # 3) Resolve additional curated YAML names before the all-sky catalogue.
    if resolve_simbad:
        pending = []
        for row in yaml_rows:
            label = clean_label(row.get('label', row.get('name', '')))
            if not passes({**row, 'name': label}):
                continue
            existing = existing_by_name.get(label)
            if existing and all(k in existing for k in ('X_pc','Y_pc','Z_pc')):
                add_preferred_star(out, seen_keys, existing)
            elif star_alias_keys(row) & seen_keys:
                continue
            else:
                pending.append(row)
        for s in resolve_simbad_positions(pending, vmag_max=vmag_max, radius_pc=radius_pc):
            add_preferred_star(out, seen_keys, s, replace_allsky_duplicate=True)

    # 4) Magnitude-limited all-sky layer. Add only stars that are not already
    # represented by the curated/YAML list or by a near-identical coordinate.
    allsky_count = 0
    skipped_duplicate = 0
    for s in load_or_query_all_sky_bright_stars(root, radius_pc=radius_pc, vmag_max=vmag_max, enabled=all_sky_simbad, force_refresh=force_all_sky_refresh):
        if not passes(s):
            continue
        allsky_count += 1
        if is_in_cluster_zone(s, member_aliases, cluster_zones):
            skipped_duplicate += 1
            continue
        if not add_preferred_star(out, seen_keys, s):
            skipped_duplicate += 1
    if all_sky_simbad:
        print(f'All-sky merge: accepted {allsky_count - skipped_duplicate} / {allsky_count}; skipped {skipped_duplicate} duplicates already in YAML/web.', file=sys.stderr)

    out.sort(key=lambda s: (fval(s.get('vmag'), 99.0) or 99.0, fval(s.get('distance_pc'), 9999.0) or 9999.0))
    return out


def polygon_area_xy(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return 0.5 * abs(s)


def _ellipse_points(blob: dict, n: int = 72) -> list[tuple[float,float]]:
    x0 = fval(blob.get('X_pc'), fval(blob.get('x'), 0.0)) or 0.0
    y0 = fval(blob.get('Y_pc'), fval(blob.get('y'), 0.0)) or 0.0
    r = blob.get('radius_pc', None)
    if isinstance(r, (list, tuple)) and len(r) >= 2:
        a = float(r[0]); b = float(r[1])
    else:
        a = float(fval(blob.get('a'), 15.0) or 15.0); b = float(fval(blob.get('b'), 8.0) or 8.0)
    ang = math.radians(fval(blob.get('angle_deg'), fval(blob.get('angle'), 0.0)) or 0.0)
    ca, sa = math.cos(ang), math.sin(ang)
    pts=[]
    for i in range(n):
        t=2*math.pi*i/n
        x=a*math.cos(t); y=b*math.sin(t)
        pts.append((x0 + x*ca - y*sa, y0 + x*sa + y*ca))
    return pts


def _radial_envelope_polygon(blobs: list[dict], bins: int = 180, margin: float = 3.0) -> list[list[float]]:
    pts=[]
    for b in blobs:
        pts.extend(_ellipse_points(b, 72))
    if len(pts) < 3:
        return []
    cx=sum(p[0] for p in pts)/len(pts); cy=sum(p[1] for p in pts)/len(pts)
    best=[None]*bins
    for x,y in pts:
        a=(math.atan2(y-cy, x-cx)+2*math.pi)%(2*math.pi)
        k=int(bins*a/(2*math.pi))%bins
        rr=math.hypot(x-cx,y-cy)+margin
        if best[k] is None or rr>best[k]: best[k]=rr
    # fill empty bins by nearest available interpolation
    vals=[v for v in best if v is not None]
    if not vals: return []
    for i in range(bins):
        if best[i] is None:
            # nearest previous/next
            for step in range(1,bins):
                a=(i-step)%bins; b=(i+step)%bins
                if best[a] is not None and best[b] is not None:
                    best[i]=0.5*(best[a]+best[b]); break
                if best[a] is not None:
                    best[i]=best[a]; break
                if best[b] is not None:
                    best[i]=best[b]; break
    # smooth circular radii
    smooth=[]
    for i in range(bins):
        smooth.append(sum(best[(i+j)%bins] for j in range(-3,4))/7.0)
    poly=[]
    for i,r in enumerate(smooth):
        a=2*math.pi*i/bins
        poly.append([round(cx+r*math.cos(a),3), round(cy+r*math.sin(a),3)])
    return poly



# -----------------------------------------------------------------------------
# 3-D cloud isosurface helpers. These are deliberately done in Python so the
# browser receives true mesh geometry, not giant ellipsoids or 2-D contours.

def _mesh_vertex_key(p: tuple[float, float, float], ndigits: int = 3) -> tuple[float, float, float]:
    return (round(float(p[0]), ndigits), round(float(p[1]), ndigits), round(float(p[2]), ndigits))


def _interp_iso(p1, p2, v1: float, v2: float, iso: float) -> tuple[float, float, float]:
    den = float(v2 - v1)
    t = 0.5 if abs(den) < 1e-12 else max(0.0, min(1.0, float((iso - v1) / den)))
    return (
        float(p1[0] + (p2[0] - p1[0]) * t),
        float(p1[1] + (p2[1] - p1[1]) * t),
        float(p1[2] + (p2[2] - p1[2]) * t),
    )


def _add_mesh_vertex(vertices: list[list[float]], lookup: dict, p) -> int:
    key = _mesh_vertex_key(p)
    hit = lookup.get(key)
    if hit is not None:
        return hit
    lookup[key] = len(vertices)
    vertices.append([key[0], key[1], key[2]])
    return lookup[key]


def _polygonise_tetra(points, values, iso: float, vertices: list[list[float]], faces: list[list[int]], lookup: dict) -> None:
    inside = [i for i, v in enumerate(values) if float(v) >= iso]
    outside = [i for i in range(4) if i not in inside]
    if len(inside) == 0 or len(inside) == 4:
        return

    def tri(a, b, c):
        ia = _add_mesh_vertex(vertices, lookup, a)
        ib = _add_mesh_vertex(vertices, lookup, b)
        ic = _add_mesh_vertex(vertices, lookup, c)
        if ia != ib and ib != ic and ia != ic:
            faces.append([ia, ib, ic])

    if len(inside) == 1 or len(inside) == 3:
        inv = len(inside) == 3
        core = outside[0] if inv else inside[0]
        others = inside if inv else outside
        p0 = _interp_iso(points[core], points[others[0]], values[core], values[others[0]], iso)
        p1 = _interp_iso(points[core], points[others[1]], values[core], values[others[1]], iso)
        p2 = _interp_iso(points[core], points[others[2]], values[core], values[others[2]], iso)
        if inv:
            tri(p0, p2, p1)
        else:
            tri(p0, p1, p2)
        return

    if len(inside) == 2:
        a, b = inside
        c, d = outside
        p0 = _interp_iso(points[a], points[c], values[a], values[c], iso)
        p1 = _interp_iso(points[a], points[d], values[a], values[d], iso)
        p2 = _interp_iso(points[b], points[c], values[b], values[c], iso)
        p3 = _interp_iso(points[b], points[d], values[b], values[d], iso)
        tri(p0, p1, p2)
        tri(p2, p1, p3)


def scalar_field_to_mesh(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, *, iso: float, max_faces: int = 2600) -> dict | None:
    field = np.asarray(field, dtype=float)
    if field.ndim != 3 or min(field.shape) < 2 or not np.isfinite(field).any():
        return None
    if float(np.nanmax(field)) <= iso:
        return None
    nx, ny, nz = field.shape
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    lookup: dict = {}
    cube_corners = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1),
    ]
    tets = [
        (0,5,1,6), (0,1,2,6), (0,2,3,6),
        (0,3,7,6), (0,7,4,6), (0,4,5,6),
    ]
    for i in range(nx - 1):
        if len(faces) > max_faces:
            break
        for j in range(ny - 1):
            if len(faces) > max_faces:
                break
            for k in range(nz - 1):
                vals = []
                pts = []
                for di, dj, dk in cube_corners:
                    vals.append(float(field[i+di, j+dj, k+dk]))
                    pts.append(np.array([float(xs[i+di]), float(ys[j+dj]), float(zs[k+dk])], dtype=float))
                if iso < min(vals) or iso > max(vals):
                    continue
                for tet in tets:
                    _polygonise_tetra([pts[t] for t in tet], [vals[t] for t in tet], iso, vertices, faces, lookup)
                    if len(faces) > max_faces:
                        break
                if len(faces) > max_faces:
                    break
    if not vertices or not faces:
        return None
    return {'vertices': vertices, 'faces': faces[:max_faces]}


def group_minkowski_blobs(blobs: list[dict]) -> list[list[dict]]:
    """Group cloud bubbles with the requested Minkowski-ball rule.

    Two bubbles belong to the same cloud when their distance is smaller than
    r1 + r2 + min(r1, r2). The grouping is transitive.
    """
    clean = [b for b in blobs if star_position_tuple(b) is not None]
    if not clean:
        return []
    centers = np.array([star_position_tuple(b) for b in clean], dtype=float)
    radii = np.array([_blob_effective_radius(b) for b in clean], dtype=float)
    used = np.zeros(len(clean), dtype=bool)
    groups: list[list[dict]] = []
    for i in range(len(clean)):
        if used[i]:
            continue
        used[i] = True
        stack = [i]
        group_idx = []
        while stack:
            a = stack.pop()
            group_idx.append(a)
            for j in range(len(clean)):
                if used[j]:
                    continue
                sep = float(np.linalg.norm(centers[a] - centers[j]))
                if sep <= radii[a] + radii[j] + min(radii[a], radii[j]):
                    used[j] = True
                    stack.append(j)
        groups.append([clean[g] for g in group_idx])
    groups.sort(key=len, reverse=True)
    return groups


def cloud_mesh_from_blobs(blobs: list[dict], *, color: str, alpha: float = 0.22, resolution: int = 24, max_faces: int = 2200) -> dict | None:
    usable = [b for b in blobs if star_position_tuple(b) is not None]
    if not usable:
        return None
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for b in usable:
        c = np.array(star_position_tuple(b), dtype=float)
        r = b.get('radius_pc', [10,10,6])
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            rr = np.array([float(r[0]), float(r[1]), float(r[2])], dtype=float)
        else:
            q = _blob_effective_radius(b); rr = np.array([q, q, 0.6*q], dtype=float)
        mins = np.minimum(mins, c - 1.35*rr)
        maxs = np.maximum(maxs, c + 1.35*rr)
    extent = np.maximum(maxs - mins, 1.0)
    # Keep resolution sane but enough to avoid spherical blobs.
    n = int(max(14, min(resolution, 26)))
    nx = max(10, min(n, int(round(n * extent[0] / max(extent)))))
    ny = max(10, min(n, int(round(n * extent[1] / max(extent)))))
    nz = max(8,  min(n, int(round(n * extent[2] / max(extent)))))
    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    F = np.zeros((nx, ny, nz), dtype=float)
    for b in usable:
        cx, cy, cz = star_position_tuple(b)
        r = b.get('radius_pc', [10,10,6])
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            a, bb, cc = max(2.0,float(r[0])), max(2.0,float(r[1])), max(1.5,float(r[2]))
        else:
            q = _blob_effective_radius(b); a = bb = q; cc = max(1.5, 0.6*q)
        ang = math.radians(fval(b.get('angle_deg'), fval(b.get('angle'), 0.0)) or 0.0)
        ca, sa = math.cos(ang), math.sin(ang)
        dx = X - cx; dy = Y - cy; dz = Z - cz
        xr = dx*ca + dy*sa
        yr = -dx*sa + dy*ca
        # This is the Minkowski / Gaussian density union; the outer isosurface
        # follows the combined density envelope, not a fitted ellipsoid.
        F += np.exp(-0.5*((xr/(0.55*a))**2 + (yr/(0.55*bb))**2 + (dz/(0.60*cc))**2))
    gf, _ = get_ndimage_tools()
    if gf is not None:
        try:
            F = gf(F, sigma=0.65)
        except Exception:
            pass
    vmax = float(np.nanmax(F))
    if vmax <= 0:
        return None
    iso = 0.22 * vmax if len(usable) > 2 else 0.28 * vmax
    mesh = scalar_field_to_mesh(F, xs, ys, zs, iso=iso, max_faces=max_faces)
    if mesh is None:
        return None
    mesh.update({'color': color, 'alpha': alpha, 'mode': 'minkowski_gaussian_isodensity', 'bubble_count': len(usable)})
    return mesh


def h5_component_mesh(vals: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, comp_idx: np.ndarray, *, iso: float, color: str = '#b9cad6', alpha: float = 0.24, max_faces: int = 2600) -> dict | None:
    if comp_idx.size == 0:
        return None
    lo = np.maximum(comp_idx.min(axis=0) - 2, 0)
    hi = np.minimum(comp_idx.max(axis=0) + 3, np.array(vals.shape))
    sub = vals[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()
    # Do not include unrelated far-away density in the component crop. The
    # isosurface is the outer boundary of this structure.
    local_mask = np.zeros_like(sub, dtype=bool)
    rel = comp_idx - lo
    local_mask[rel[:,0], rel[:,1], rel[:,2]] = True
    if gaussian_filter is not None:
        try:
            local_mask = gaussian_filter(local_mask.astype(float), sigma=1.0) > 0.02
        except Exception:
            pass
    sub = np.where(local_mask, sub, 0.0)
    if min(sub.shape) < 2 or float(np.nanmax(sub)) <= iso:
        return None
    mesh = scalar_field_to_mesh(sub, xs[lo[0]:hi[0]], ys[lo[1]:hi[1]], zs[lo[2]:hi[2]], iso=iso, max_faces=max_faces)
    if mesh is None:
        return None
    mesh.update({'color': color, 'alpha': alpha, 'mode': 'h5_component_isodensity', 'voxel_count': int(comp_idx.shape[0])})
    return mesh


def build_surface_polygons_from_blobs(blobs: list[dict], *, z_pc: float, color: str, alpha: float = 0.085, max_polygons: int = 3) -> list[dict]:
    """Build 2-D smooth projected cloud surfaces from blob ellipses.

    This mirrors the spirit of solar_map_400pc.py's 2-D cloud fusion but avoids
    a hard dependency on shapely. If matplotlib is available, a smoothed density
    contour is used; otherwise a radial envelope fallback is returned.
    """
    usable=[b for b in blobs if fval(b.get('X_pc'), fval(b.get('x'), None)) is not None and fval(b.get('Y_pc'), fval(b.get('y'), None)) is not None]
    if not usable:
        return []
    allpts=[]
    for b in usable:
        allpts.extend(_ellipse_points(b, 36))
    xs=[p[0] for p in allpts]; ys=[p[1] for p in allpts]
    pad=max(12.0, 0.08*max(max(xs)-min(xs), max(ys)-min(ys), 1.0))
    x0,x1=min(xs)-pad,max(xs)+pad; y0,y1=min(ys)-pad,max(ys)+pad
    nx=ny=120
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'A NumPy version .* is required for this version of SciPy.*', category=UserWarning)
            from scipy.ndimage import gaussian_filter as gf
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        X=np.linspace(x0,x1,nx); Y=np.linspace(y0,y1,ny)
        XX,YY=np.meshgrid(X,Y,indexing='xy')
        F=np.zeros_like(XX,dtype=float)
        for b in usable:
            bx=fval(b.get('X_pc'), fval(b.get('x'), 0.0)) or 0.0
            by=fval(b.get('Y_pc'), fval(b.get('y'), 0.0)) or 0.0
            r=b.get('radius_pc', None)
            if isinstance(r,(list,tuple)) and len(r)>=2:
                a=float(r[0]); bb=float(r[1])
            else:
                a=float(fval(b.get('a'), 15.0) or 15.0); bb=float(fval(b.get('b'), 8.0) or 8.0)
            ang=math.radians(fval(b.get('angle_deg'), fval(b.get('angle'),0.0)) or 0.0)
            ca,sa=math.cos(ang),math.sin(ang)
            dx=XX-bx; dy=YY-by
            xr= dx*ca + dy*sa
            yr=-dx*sa + dy*ca
            sigx=max(4.0,0.55*a); sigy=max(3.0,0.55*bb)
            amp=max(0.7, min(2.5, fval(b.get('members_merged'), 1.0) or 1.0))
            F += amp*np.exp(-0.5*((xr/sigx)**2 + (yr/sigy)**2))
        F=gf(F, sigma=2.2)
        vmax=float(np.nanmax(F))
        if vmax <= 0:
            raise RuntimeError('empty cloud field')
        level=0.22*vmax if len(usable)>2 else 0.30*vmax
        fig=plt.figure(figsize=(2,2))
        ax=fig.add_subplot(111)
        cs=ax.contour(X,Y,F,levels=[level])
        segs=[]
        try:
            segs=[np.asarray(seg) for seg in cs.allsegs[0] if len(seg)>=16]
        finally:
            plt.close(fig)
        polys=[]
        for seg in segs:
            # resample/simplify to keep JSON small
            step=max(1,len(seg)//140)
            pts=[[round(float(x),3),round(float(y),3)] for x,y in seg[::step]]
            area=polygon_area_xy(pts)
            if area>30:
                polys.append((area,pts))
        polys.sort(reverse=True,key=lambda v:v[0])
        out=[]
        for area,pts in polys[:max_polygons]:
            out.append({'points': pts, 'Z_pc': round(float(z_pc),3), 'thickness_pc': 8.0, 'color': color, 'alpha': alpha, 'area_pc2': round(area,2), 'mode': 'density_contour'})
        if out:
            return out
    except Exception:
        pass
    pts=_radial_envelope_polygon(usable, bins=160, margin=4.0)
    if pts:
        return [{'points': pts, 'Z_pc': round(float(z_pc),3), 'thickness_pc': 8.0, 'color': color, 'alpha': alpha, 'area_pc2': round(polygon_area_xy(pts),2), 'mode': 'radial_envelope_fallback'}]
    return []


def nebula_radius(row: dict) -> list[float]:
    d = fval(row.get('distance_pc'), 100.0) or 100.0
    a = d * math.radians(fval(row.get('a_deg'), 8.0) or 8.0)
    b = d * math.radians(fval(row.get('b_deg'), 4.0) or 4.0)
    return [round(max(4.0, a), 3), round(max(3.0, b), 3), round(max(2.0, min(a,b)*0.55), 3)]


def extract_json_array(text: str, token: str, *, start: int = 0, backwards: bool = False):
    pos = text.rfind(token, 0, start) if backwards else text.find(token, start)
    if pos < 0: return None
    i = text.find('[', pos)
    if i < 0: return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '[': depth += 1
        elif text[j] == ']':
            depth -= 1
            if depth == 0:
                return json.loads(text[i:j+1])
    return None


def load_cloud_catalog_from_html(html_path: Path) -> dict[str, list[dict]]:
    if not html_path.exists():
        return {}
    text = html_path.read_text(encoding='utf-8', errors='ignore')
    anchor = text.find('"name": "Major Cloud Catalog"')
    if anchor < 0:
        return {}
    sizes = extract_json_array(text, '"size": ', start=anchor, backwards=True)
    labels = extract_json_array(text, '"text": ', start=anchor)
    xs = extract_json_array(text, '"x": ', start=anchor)
    ys = extract_json_array(text, '"y": ', start=anchor)
    zs = extract_json_array(text, '"z": ', start=anchor)
    if not all(v is not None for v in [sizes, labels, xs, ys, zs]):
        return {}
    groups: dict[str, list[dict]] = {}
    for label, x_raw, y_raw, z_raw, size in zip(labels, xs, ys, zs, sizes):
        # Same remap used in solar_map_400pc.py: X_plot=y_html, Y_plot=x_html.
        row = {
            'catalog_label': str(label),
            'X_pc': float(y_raw),
            'Y_pc': float(x_raw),
            'Z_pc': float(z_raw),
            'size': float(size),
            'r3d': float(math.sqrt(float(x_raw)**2 + float(y_raw)**2 + float(z_raw)**2)),
        }
        groups.setdefault(str(label), []).append(row)
    return groups


def build_regions(yaml_data: dict, existing_regions: list[dict], root: Path, radius_pc: float) -> tuple[list[dict], int, int]:
    regions: list[dict] = []
    neb_rows = dedupe_by_label(list(yaml_data.get('NEBULAE', [])) + list(yaml_data.get('NEBULAE_400PC_EXTRA', [])))
    dust_blobs = yaml_data.get('_DUST_XY_BLOBS', {}) or {}
    aliases = yaml_data.get('CLOUD_CATALOG_ALIASES', {}) or {}
    defaults = yaml_data.get('CLOUD_RENDER_DEFAULTS', {}) or {}
    bubble_scale = fval(defaults.get('bubble_scale_pc'), 3.3) or 3.3
    bubble_min = fval(defaults.get('bubble_min_pc'), 10.0) or 10.0
    html_groups = load_cloud_catalog_from_html(root / 'handbook_distances.html')
    bubble_count = 0

    for row in neb_rows:
        label = clean_label(row.get('label', row.get('name')))
        d = fval(row.get('distance_pc'), 0.0) or 0.0
        if d > radius_pc + 80:
            continue
        coords = coords_for_row(row)
        if coords is None:
            continue
        region: dict[str, Any] = {
            'name': label,
            'catalog_name': str(row.get('name', label)),
            'kind': 'cloud',
            'distance_pc': round(d, 4),
            'ra_deg': fval(row.get('ra_deg'), 0.0) or 0.0,
            'dec_deg': fval(row.get('dec_deg'), 0.0) or 0.0,
            **{k: round(float(v), 5 if k.endswith('_pc') else 7) for k, v in coords.items()},
            'radius_pc': nebula_radius(row),
            'color': str(row.get('color', '#8b6b4a')),
            'source': 'solar_map_400pc_data.yaml',
        }
        # Manual irregular 400 pc dust morphology from the YAML.
        if label in {clean_label(k) for k in dust_blobs.keys()}:
            matching_key = next(k for k in dust_blobs.keys() if clean_label(k) == label)
            region['kind'] = 'dust'
            region['render_mode'] = 'irregular_yaml_blobs_3d'
            region['blobs'] = []
            for i, blob in enumerate(dust_blobs.get(matching_key, [])):
                if 'color' in blob:
                    # dark core; keep it but make it a separate smaller volume
                    alpha = fval(blob.get('alpha'), 0.032) or 0.032
                else:
                    alpha = 0.020
                region['blobs'].append({
                    'X_pc': fval(blob.get('x'), region['X_pc']) or region['X_pc'],
                    'Y_pc': fval(blob.get('y'), region['Y_pc']) or region['Y_pc'],
                    'Z_pc': fval(blob.get('z'), region['Z_pc']) or region['Z_pc'],
                    'radius_pc': [
                        fval(blob.get('a'), 20.0) or 20.0,
                        fval(blob.get('b'), 10.0) or 10.0,
                        fval(blob.get('z_radius'), max(4.0, 0.33 * (fval(blob.get('b'), 10.0) or 10.0))) or 5.0,
                    ],
                    'angle_deg': fval(blob.get('angle'), 0.0) or 0.0,
                    'color': str(blob.get('color', region['color'])),
                    'alpha': alpha,
                    'seed': int(blob.get('seed', i)),
                })
        # Physical handbook bubbles from the Plotly HTML, grouped through aliases.
        raw_aliases = []
        for k, vals in aliases.items():
            if clean_label(k) == label:
                raw_aliases = vals or []
                break
        catalog_blobs = []
        for alias in raw_aliases:
            for pt in html_groups.get(str(alias), []):
                if math.hypot(pt['X_pc'], pt['Y_pc']) <= radius_pc + 80 and pt['r3d'] <= radius_pc + 120:
                    r = max(bubble_min, bubble_scale * math.sqrt(max(pt['size'], 1.0)))
                    catalog_blobs.append({
                        'X_pc': round(pt['X_pc'], 3),
                        'Y_pc': round(pt['Y_pc'], 3),
                        'Z_pc': round(pt['Z_pc'], 3),
                        'radius_pc': [round(r, 3), round(r, 3), round(max(5.0, 0.55*r), 3)],
                        'color': region['color'],
                        'alpha': 0.018,
                        'catalog_label': pt['catalog_label'],
                    })
        cloud_meshes = []
        if catalog_blobs:
            # Keep the real catalogue bubbles for the density field, group them
            # by the Minkowski-ball rule, then extract a true 3-D isodensity
            # mesh for each connected cloud.
            catalog_subset = catalog_blobs[:160]
            groups = group_minkowski_blobs(catalog_subset)
            for g in groups[:5]:
                mesh = cloud_mesh_from_blobs(g, color=region.get('color', '#8b6b4a'), alpha=0.24, resolution=23, max_faces=1800)
                if mesh is not None:
                    cloud_meshes.append(mesh)
            if 'blobs' not in region:
                region['render_mode'] = 'handbook_catalog_minkowski_isodensity_3d'
                region['blobs'] = []
            # Fallback/debug only; renderer now prioritises cloud_meshes.
            region['blobs'].extend(merge_nearby_blobs(catalog_subset))
            bubble_count += len(catalog_subset)
            region['source'] += ' + handbook_distances.html (Minkowski grouped isodensity meshes)'
        if region.get('blobs') and not cloud_meshes:
            for g in group_minkowski_blobs(region.get('blobs', []))[:4]:
                mesh = cloud_mesh_from_blobs(g, color=region.get('color', '#8b6b4a'), alpha=0.22 if region.get('kind') == 'dust' else 0.24, resolution=22, max_faces=1700)
                if mesh is not None:
                    cloud_meshes.append(mesh)
        if cloud_meshes:
            region['cloud_meshes'] = cloud_meshes
        regions.append(region)
    return regions, len(html_groups), bubble_count


def export_dust_clouds_from_h5(h5_path: Path, radius_pc: float, max_clouds: int = 48, percentile: float = 94.5, sigma: float = 1.35) -> tuple[list[dict], str]:
    gaussian_filter_local, ndi_label_local = get_ndimage_tools()
    if not h5_path.exists():
        return [], f'H5 not found: {h5_path.name}'
    if h5py is None:
        return [], 'h5py unavailable; cannot preprocess H5'
    with h5py.File(h5_path, 'r') as h5:
        ds = h5['/stilism/cube_datas'] if '/stilism/cube_datas' in h5 else None
        if ds is None:
            return [], 'No /stilism/cube_datas dataset found'
        gridstep = np.asarray(ds.attrs.get('gridstep_values', [5,5,5]), dtype=float)
        sun = np.asarray(ds.attrs.get('sun_position', [s/2 for s in ds.shape]), dtype=float)
        dx,dy,dz = [float(v) for v in gridstep]
        sx,sy,sz = [float(v) for v in sun]
        hx,hy,hz = [int(math.ceil(radius_pc / v)) for v in (dx,dy,dz)]
        ix0,ix1 = max(0,int(sx-hx)), min(ds.shape[0], int(sx+hx+1))
        iy0,iy1 = max(0,int(sy-hy)), min(ds.shape[1], int(sy+hy+1))
        iz0,iz1 = max(0,int(sz-hz)), min(ds.shape[2], int(sz+hz+1))
        sub = np.asarray(ds[ix0:ix1:2, iy0:iy1:2, iz0:iz1:2], dtype=float)
        xs = (np.arange(ix0, ix1, 2, dtype=float) - sx) * dx
        ys = (np.arange(iy0, iy1, 2, dtype=float) - sy) * dy
        zs = (np.arange(iz0, iz1, 2, dtype=float) - sz) * dz
    vals = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
    if gaussian_filter_local is not None:
        vals = gaussian_filter_local(vals, sigma=float(sigma))
    X,Y,Z = np.meshgrid(xs, ys, zs, indexing='ij')
    rmask = (X*X + Y*Y + Z*Z) <= radius_pc**2
    vals = np.where(rmask, vals, 0.0)
    good = vals[vals > 0]
    if good.size == 0:
        return [], 'H5 read OK but no positive dust voxels in radius'
    thresh = float(np.nanpercentile(good, float(percentile)))
    mask = vals >= thresh
    if ndi_label_local is None:
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return [], 'H5 thresholding yielded no components'
        strengths = vals[idx[:,0], idx[:,1], idx[:,2]]
        pts = np.column_stack([xs[idx[:,0]], ys[idx[:,1]], zs[idx[:,2]], strengths])
        blobs = merge_nearby_blobs([{'X_pc': float(a), 'Y_pc': float(b), 'Z_pc': float(c), 'radius_pc': [14,14,10]} for a,b,c,_ in pts[:800]])
        return blobs[:max_clouds], f'H5 dust cloud fallback export: {len(blobs[:max_clouds])} clouds'
    labels, nlab = ndi_label_local(mask, structure=np.ones((3,3,3), dtype=int))
    clouds = []
    vmax = float(np.nanmax(vals)) or 1.0
    for lab in range(1, nlab + 1):
        idx = np.argwhere(labels == lab)
        if len(idx) < 6:
            continue
        strengths = vals[idx[:,0], idx[:,1], idx[:,2]]
        weight_sum = float(np.sum(strengths))
        if weight_sum <= 0:
            continue
        pts = np.column_stack([xs[idx[:,0]], ys[idx[:,1]], zs[idx[:,2]]])
        ctr = np.average(pts, axis=0, weights=strengths)
        dif = pts - ctr
        std = np.sqrt(np.average(dif**2, axis=0, weights=strengths))
        ext = np.max(np.abs(dif), axis=0) if len(pts) else std
        rx = float(max(10.0, 1.6*std[0] + 0.55*ext[0] + dx))
        ry = float(max(10.0, 1.6*std[1] + 0.55*ext[1] + dy))
        rz = float(max(7.0,  1.4*std[2] + 0.50*ext[2] + dz))
        peak = float(np.max(strengths) / vmax)
        mesh = h5_component_mesh(vals, xs, ys, zs, idx, iso=thresh, color='#b9cad6', alpha=0.24, max_faces=2300)
        row = {
            'X_pc': round(float(ctr[0]), 3),
            'Y_pc': round(float(ctr[1]), 3),
            'Z_pc': round(float(ctr[2]), 3),
            'radius_pc': [round(rx,3), round(ry,3), round(rz,3)],
            'strength': round(peak, 4),
            'alpha': round(float(np.clip(0.080 + 0.080*peak, 0.10, 0.22)), 4),
            'kind': 'dust_h5',
            'voxel_count': int(len(idx)),
        }
        if mesh is not None:
            row['cloud_meshes'] = [mesh]
        clouds.append(row)
    clouds.sort(key=lambda c: (c.get('strength', 0), c.get('voxel_count', 0)), reverse=True)
    clouds = clouds[:max_clouds]
    for c in clouds:
        c['color'] = '#b9cad6'
    return clouds, f'H5 dust isodensity meshes exported: {sum(len(c.get("cloud_meshes", [])) for c in clouds)} meshes from {len(clouds)} components at p{percentile:g}'


def write_outputs(root: Path, stars, const_stars, systems, regions, dust_clouds, meta):
    data_dir = root/'assets'/'data'; js_dir = root/'assets'/'js'
    data_dir.mkdir(parents=True, exist_ok=True); js_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [('bright_stars.json', stars), ('constellation_stars.json', const_stars), ('systems.json', systems), ('regions.json', regions), ('dust_clouds.json', dust_clouds), ('dust_voxels.json', []), ('map_meta.json', meta)]:
        (data_dir/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    payload = {'stars': stars, 'constellationStars': const_stars, 'systems': systems, 'regions': regions, 'dustClouds': dust_clouds, 'dustVoxels': [], 'meta': meta}
    (js_dir/'solar-data.js').write_text('window.SOLAR_MAP_DATA = ' + json.dumps(payload, indent=2, ensure_ascii=False) + ';\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--yaml', type=Path, default=None)
    ap.add_argument('--radius-pc', type=float, default=400.0)
    ap.add_argument('--include-associations', action='store_true')
    ap.add_argument('--top-cluster-members', type=int, default=4)
    ap.add_argument('--star-vmag-max', type=float, default=4.2)
    ap.add_argument('--no-resolve-simbad-stars', action='store_true', help='Disable per-name SIMBAD resolution for YAML stars.')
    ap.add_argument('--no-all-sky-simbad-stars', action='store_true', help='Disable the all-sky SIMBAD TAP bright-star query.')
    ap.add_argument('--refresh-all-sky-star-cache', action='store_true', help='Ignore cached all-sky bright-star JSON and query SIMBAD again.')
    ap.add_argument('--respect-yaml-star-exclusions', action='store_true', help='Apply EXCLUDED_BRIGHT_STAR_NAMES from the thesis YAML. Default is false for web completeness.')
    ap.add_argument('--max-dust-clouds', type=int, default=48)
    ap.add_argument('--h5-percentile', type=float, default=94.5)
    ap.add_argument('--h5-sigma', type=float, default=1.35)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    root = args.root.resolve()
    yaml_path = args.yaml or (root/'solar_map_400pc_data.yaml')
    if not yaml_path.exists():
        yaml_path = root/'scripts'/'solar_map_400pc_data.yaml'
    y = load_yaml(yaml_path)
    existing_stars = load_json(root/'assets'/'data'/'bright_stars.json', [])
    existing_const = load_json(root/'assets'/'data'/'constellation_stars.json', [])
    existing_systems = load_json(root/'assets'/'data'/'systems.json', [])
    existing_regions = load_json(root/'assets'/'data'/'regions.json', [])

    systems = build_systems(y, existing_systems, args.radius_pc, args.include_associations, args.top_cluster_members)
    stars = merge_bright_stars(y, existing_stars, args.radius_pc, vmag_max=args.star_vmag_max, resolve_simbad=not args.no_resolve_simbad_stars, root=root, all_sky_simbad=not args.no_all_sky_simbad_stars, force_all_sky_refresh=args.refresh_all_sky_star_cache, respect_yaml_exclusions=args.respect_yaml_star_exclusions, systems_for_exclusion=systems)
    regions, html_groups, bubble_count = build_regions(y, existing_regions, root, args.radius_pc)
    dust_clouds, h5_status = export_dust_clouds_from_h5(root/'map3D_GAIAdr2_feb2019.h5', args.radius_pc, args.max_dust_clouds, args.h5_percentile, args.h5_sigma)
    meta = {
        'radius_pc': args.radius_pc,
        'yaml_file': str(yaml_path.name),
        'handbook_html_loaded': bool((root/'handbook_distances.html').exists()),
        'handbook_catalog_groups': html_groups,
        'handbook_bubbles_exported': bubble_count,
        'h5_status': h5_status,
        'dust_clouds_exported': len(dust_clouds),
    }
    print(json.dumps({'stars': len(stars), 'systems': len(systems), 'regions': len(regions), **meta}, indent=2, ensure_ascii=False))
    if not args.dry_run:
        write_outputs(root, stars, existing_const, systems, regions, dust_clouds, meta)
        print('Updated assets/data/*.json and assets/js/solar-data.js')


if __name__ == '__main__':
    main()
