"""
Clean automatic X-Y map of nearby stellar systems for thesis figures.

400 pc companion version reconstructed from the clean 200 pc map.
It preserves the final active behaviour of the 200 pc map and extends it to
400 pc with additional clusters, bright stars, Gaia density sampling, and
large irregular dust / OB-association surfaces.

Author: Luis Gonzalez Ramirez / thesis utilities
"""

from __future__ import annotations

import logging
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.simbad import Simbad
import yaml

try:
    import h5py
except Exception:
    h5py = None

try:
    from astroquery.gaia import Gaia as GaiaArchive
except Exception:
    GaiaArchive = None

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.labelsize"] = 16
matplotlib.rcParams["axes.titlesize"] = 18
matplotlib.rcParams["xtick.labelsize"] = 16
matplotlib.rcParams["ytick.labelsize"] = 16
matplotlib.rcParams["legend.fontsize"] = 14


# Gaia density layer colormap.
# Kept in the Python script rather than the YAML because it is a Matplotlib object,
# not plain serializable data.
_clean_GAIA_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "clean_gaia_density",
    [
        (0.00, (1.00, 1.00, 1.00, 0.00)),
        (0.20, (0.90, 0.95, 1.00, 0.10)),
        (0.45, (0.62, 0.76, 0.96, 0.20)),
        (0.70, (0.36, 0.56, 0.88, 0.28)),
        (1.00, (0.20, 0.36, 0.72, 0.36)),
    ],
)


@dataclass(frozen=True)
class GaiaMapConfig:
    radius_pc: float = 400.0
    language: str = "en"
    show_system_regions: bool = True
    show_nebulae: bool = True
    show_gaia_density: bool = True
    show_association_members: bool = False  # False = fast preview; True = detailed association hulls
    region_alpha: float = 0.16
    nebula_alpha: float = 0.14
    max_star_labels: int = 28
    max_stars_total: int = 34
    cache_dir: Path = Path("cache")
    figure_dir: Path = Path("figures")
    system_cache_name: str = 'nearby_systems_400pc.ecsv'
    bright_star_cache_name: str = 'bright_named_stars_400pc_v3.ecsv'
    nebula_cache_name: str = 'nearby_nebulae_400pc.ecsv'
    data_yaml_path: Path = Path('solar_map_400pc_data.yaml')
    cloud_catalog_html: Path | None = Path('handbook_distances.html')
    dust_h5_path: Path | None = Path('map3D_GAIAdr2_feb2019.h5')
    cloud_render_mode: str = 'hybrid'  # manual, catalog, h5, hybrid
    show_physical_dust_projection: bool = True
    dust_projection_mode: str = 'max'   # 'max' = more visual, 'sum' = more physical
    dust_projection_sphere_only: bool = True
    preview_png_dpi: int | None = 150
    save_pdf: bool = False
    draw_label_leaders: bool = True
    fast_label_leaders: bool = True
    dust_projection_alpha: float = 0.30
    dust_contour_alpha: float = 0.58
    dust_contour_lw: float = 1.05
    dust_percentile_lo: float = 58.0
    dust_percentile_hi: float = 99.0
    dust_lowest_contour_level: float = 0.16
    cloud_radius_margin_pc: float = 40.0
    cloud_bubble_scale_pc: float = 3.3
    cloud_bubble_min_pc: float = 10.0


# ---------------------------------------------------------------------
# Final data tables / dictionaries (loaded from YAML)
# ---------------------------------------------------------------------

_YAML_SET_KEYS = {
    'FORCE_BRIGHT_STAR_NAMES',
    'INNER_30PC_LABELLED_NAMES',
    'EXCLUDED_BRIGHT_STAR_NAMES',
    'NEBULA_LABELS_NO_LEADERS_400PC',
}


def _repair_mathtext_label(value):
    if not isinstance(value, str):
        return value
    out = value
    out = out.replace('$\x07lpha$', r'$\alpha$')
    out = out.replace('$\x08eta$', r'$\beta$')
    out = out.replace(r'$\eta$', r'$\eta$')
    out = out.replace(r'$\epsilon$', r'$\epsilon$')
    out = out.replace(r'$\rho$', r'$\rho$')
    out = out.replace('$\r' + 'ho$', r'$\rho$')
    return out


def _repair_mathtext_obj(obj):
    if isinstance(obj, list):
        return [_repair_mathtext_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_repair_mathtext_obj(v) for v in obj)
    if isinstance(obj, set):
        return {_repair_mathtext_obj(v) for v in obj}
    if isinstance(obj, dict):
        return {
            _repair_mathtext_label(k): _repair_mathtext_obj(v)
            for k, v in obj.items()
        }
    return _repair_mathtext_label(obj)


def _repair_mathtext_table(tab):
    """Repair accidental control-character LaTeX labels in Astropy tables.

    The YAML loader repairs dictionaries/lists. This helper keeps the old plotting
    calls safe for Astropy tables returned by the resolver functions.
    """
    if tab is None:
        return None
    try:
        colnames = list(tab.colnames)
    except Exception:
        return tab
    for col in colnames:
        try:
            values = tab[col]
        except Exception:
            continue
        try:
            if getattr(values.dtype, 'kind', '') not in {'U', 'S', 'O'}:
                continue
        except Exception:
            pass
        try:
            tab[col] = [_repair_mathtext_label(v) for v in values]
        except Exception:
            pass
    return tab


def _load_yaml_bundle(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'YAML data file not found: {path}')
    bundle = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(bundle, dict):
        raise ValueError(f'Unexpected YAML content in {path}')
    for key in _YAML_SET_KEYS:
        if key in bundle and not isinstance(bundle[key], set):
            bundle[key] = set(bundle[key])
    return _repair_mathtext_obj(bundle)


def _extend_unique_by_label(target: list[dict], additions: list[dict]) -> None:
    seen = {str(obj.get('label', obj.get('name', ''))) for obj in target}
    for obj in additions:
        key = str(obj.get('label', obj.get('name', '')))
        if key not in seen:
            target.append(obj)
            seen.add(key)


_DATA_BUNDLE = _load_yaml_bundle(Path(__file__).with_name('solar_map_400pc_data.yaml'))
for _key, _value in _DATA_BUNDLE.items():
    globals()[_key] = _value

# Explicit bindings for static checkers and for clearer runtime errors if the YAML
# file is incomplete.  The values still come from solar_map_400pc_data.yaml.
CLEAN_SYSTEM_CACHE = _DATA_BUNDLE.get('CLEAN_SYSTEM_CACHE', 'nearby_systems_400pc.ecsv')
CLEAN_BRIGHT_STAR_CACHE = _DATA_BUNDLE.get('CLEAN_BRIGHT_STAR_CACHE', 'bright_named_stars_400pc_v3.ecsv')
CLEAN_NEBULA_CACHE = _DATA_BUNDLE.get('CLEAN_NEBULA_CACHE', 'nearby_nebulae_400pc.ecsv')
CLEAN_ASSOC_MEMBER_CACHE = _DATA_BUNDLE.get('CLEAN_ASSOC_MEMBER_CACHE', 'association_members_400pc.ecsv')
CLEAN_GAIA_DENSITY_CACHE = _DATA_BUNDLE.get('CLEAN_GAIA_DENSITY_CACHE', 'gaia_density_400pc.ecsv')

NEARBY_SYSTEMS = _DATA_BUNDLE.get('NEARBY_SYSTEMS', [])
BRIGHT_NAMED_STARS = _DATA_BUNDLE.get('BRIGHT_NAMED_STARS', [])
SYSTEMS_400PC_EXTRA = _DATA_BUNDLE.get('SYSTEMS_400PC_EXTRA', [])
BRIGHT_STARS_400PC_EXTRA = _DATA_BUNDLE.get('BRIGHT_STARS_400PC_EXTRA', [])
NEBULAE = _DATA_BUNDLE.get('NEBULAE', [])
NEBULAE_400PC_EXTRA = _DATA_BUNDLE.get('NEBULAE_400PC_EXTRA', [])
NEBULA_COMPONENTS = _DATA_BUNDLE.get('NEBULA_COMPONENTS', {})
NEBULA_LABEL_OFFSETS = _DATA_BUNDLE.get('NEBULA_LABEL_OFFSETS', {})
NEBULA_LABELS_NO_LEADERS_400PC = set(_DATA_BUNDLE.get('NEBULA_LABELS_NO_LEADERS_400PC', []))
ASSOCIATION_MEMBER_SPECS = _DATA_BUNDLE.get('ASSOCIATION_MEMBER_SPECS', {})
MANUAL_STAR_OFFSETS = _DATA_BUNDLE.get('MANUAL_STAR_OFFSETS', {})
MANUAL_LABEL_OVERRIDES = _DATA_BUNDLE.get('MANUAL_LABEL_OVERRIDES', {})
SYSTEM_POSITION_OVERRIDES_400PC = _DATA_BUNDLE.get('SYSTEM_POSITION_OVERRIDES_400PC', {})
SYSTEM_POSITION_SHIFTS_400PC = _DATA_BUNDLE.get('SYSTEM_POSITION_SHIFTS_400PC', {})
SYSTEM_LABEL_OFFSETS = _DATA_BUNDLE.get('SYSTEM_LABEL_OFFSETS', {})
UI_TEXT = _DATA_BUNDLE.get('UI_TEXT', {})
STAR_TEFF = _DATA_BUNDLE.get('STAR_TEFF', {})
SYSTEM_SKY_REGION_SPECS = _DATA_BUNDLE.get('SYSTEM_SKY_REGION_SPECS', {})
_DUST_XY_BLOBS = _DATA_BUNDLE.get('_DUST_XY_BLOBS', {})
REFERENCE_OB_LABELS_400PC = _DATA_BUNDLE.get('REFERENCE_OB_LABELS_400PC', [])
STAR_POSITION_OVERRIDES_400PC = _DATA_BUNDLE.get('STAR_POSITION_OVERRIDES_400PC', {})
STAR_RADIUS_ESTIMATE_RSUN = _DATA_BUNDLE.get('STAR_RADIUS_ESTIMATE_RSUN', {})
FORCE_BRIGHT_STAR_NAMES = set(_DATA_BUNDLE.get('FORCE_BRIGHT_STAR_NAMES', []))
INNER_30PC_LABELLED_NAMES = set(_DATA_BUNDLE.get('INNER_30PC_LABELLED_NAMES', []))
EXCLUDED_BRIGHT_STAR_NAMES = set(_DATA_BUNDLE.get('EXCLUDED_BRIGHT_STAR_NAMES', []))


# Preserve the original behaviour: extend the base lists with the curated 400 pc extras.
_extend_unique_by_label(NEARBY_SYSTEMS, SYSTEMS_400PC_EXTRA)
_extend_unique_by_label(BRIGHT_NAMED_STARS, BRIGHT_STARS_400PC_EXTRA)
_extend_unique_by_label(NEBULAE, NEBULAE_400PC_EXTRA)

_ALLOWED_NEBULA_LABELS = {str(obj['label']) for obj in NEBULAE}

# Cloud label aliases used to translate GalaxyMap / handbook bubble names into
# the labels already used by this figure. These can also live in the YAML file.
CLOUD_CATALOG_ALIASES = _repair_mathtext_obj(_DATA_BUNDLE.get('CLOUD_CATALOG_ALIASES', {
    'Taurus complex': ['Taurus'],
    'Taurus dark clouds': ['Taurus'],
    'Near Perseus dark clouds': ['Perseus', 'Polaris'],
    'Cassiopeia dark clouds': ['Cepheus', 'Polaris'],
    'Cepheus Flare': ['Cepheus'],
    'Cepheus CO void': ['Cepheus'],
    'Perseus cloud': ['Perseus', 'Polaris'],
    'Orion molecular clouds': ['Orion', 'Orion_Lam'],
    'Orion nebula': ['Orion'],
    'California nebula': ['California'],
    'Aquila south rift': ['AqRift', 'Aquila_S'],
    'Ophiuchus dark clouds': ['Ophiuchus', 'OphArc', 'OphNorth'],
    r'$\rho$ Oph complex': ['Ophiuchus', 'L1688'],
    'Lupus clouds': ['Lupus'],
    'Chamaeleon complex': ['Chamaeleon'],
    'Coalsack': ['Coalsack', 'NorthernCoalsack'],
    'Pipe nebula': ['B44', 'B45', 'B59'],
    'ORI OB1': ['Orion', 'Orion_Lam'],
    'VEL OB2': ['VelaC'],
}))

CLOUD_RENDER_DEFAULTS = _DATA_BUNDLE.get('CLOUD_RENDER_DEFAULTS', {
    'bubble_scale_pc': 3.3,
    'bubble_min_pc': 10.0,
    'bubble_resolution': 32,
    'projection_alpha': 0.10,
    'projection_sigma': 1.1,
})

# ---------------------------------------------------------------------
# Physical cloud helpers: GalaxyMap handbook bubbles + optional Lallement dust cube
# ---------------------------------------------------------------------


def _extract_json_array(text: str, token: str, *, start: int = 0, backwards: bool = False):
    pos = text.rfind(token, 0, start) if backwards else text.find(token, start)
    if pos < 0:
        return None
    i = text.find('[', pos)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        ch = text[j]
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return json.loads(text[i:j+1])
    return None


def load_cloud_catalog_from_html(html_path: Path | str) -> dict[str, list[dict]]:
    html_path = Path(html_path)
    if not html_path.exists():
        logging.warning('Cloud HTML not found: %s', html_path)
        return {}
    text = html_path.read_text(encoding='utf-8', errors='ignore')
    anchor = text.find('"name": "Major Cloud Catalog"')
    if anchor < 0:
        logging.warning('Major Cloud Catalog trace not found in %s', html_path)
        return {}

    sizes = _extract_json_array(text, '"size": ', start=anchor, backwards=True)
    labels = _extract_json_array(text, '"text": ', start=anchor)
    xs = _extract_json_array(text, '"x": ', start=anchor)
    ys = _extract_json_array(text, '"y": ', start=anchor)
    zs = _extract_json_array(text, '"z": ', start=anchor)
    if not all(v is not None for v in [sizes, labels, xs, ys, zs]):
        logging.warning('Could not extract all arrays from %s', html_path)
        return {}

    n = min(len(labels), len(xs), len(ys), len(zs), len(sizes))
    groups: dict[str, list[dict]] = {}
    for i in range(n):
        group = str(labels[i])
        # The GalaxyMap/handbook 3D Plotly scene does not use the same visible
        # axis ordering as this thesis X-Y heliocentric map.  Empirically, the
        # cloud points line up with the plotted local-neighbourhood map when the
        # HTML scene coordinates are remapped as:
        #   X_plot = y_html
        #   Y_plot = x_html
        # while z_html is kept only for the 3D radial cut.
        x_raw = float(xs[i])
        y_raw = float(ys[i])
        z_raw = float(zs[i])
        row = {
            'catalog_label': group,
            'x_raw': x_raw,
            'y_raw': y_raw,
            'z_raw': z_raw,
            'x': y_raw,
            'y': x_raw,
            'z': z_raw,
            'size': float(sizes[i]),
            'r3d': float(np.sqrt(x_raw ** 2 + y_raw ** 2 + z_raw ** 2)),
        }
        groups.setdefault(group, []).append(row)
    return groups


def _darken_hex_color(color: str, factor: float = 0.80) -> str:
    color = color.lstrip('#')
    if len(color) != 6:
        return '#5a3921'
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    r = max(0, min(255, int(round(r * factor))))
    g = max(0, min(255, int(round(g * factor))))
    b = max(0, min(255, int(round(b * factor))))
    return f'#{r:02x}{g:02x}{b:02x}'


def _estimate_row_cloud_radius_pc(row) -> float:
    """Projected radius used to reject distant/aliased handbook bubbles."""
    try:
        dist = float(row.get('distance_pc', np.nan))
    except Exception:
        dist = np.nan
    try:
        a_deg = float(row.get('a_deg', np.nan))
    except Exception:
        a_deg = np.nan
    try:
        b_deg = float(row.get('b_deg', np.nan))
    except Exception:
        b_deg = np.nan

    ang = np.nanmax([a_deg, b_deg])
    if np.isfinite(dist) and np.isfinite(ang) and ang > 0:
        radius_pc = dist * np.deg2rad(ang)
        return float(max(55.0, 2.8 * radius_pc + 35.0))
    return 140.0


def _filter_catalog_points_near_row(points: list[dict], row) -> list[dict]:
    """Keep only handbook bubbles spatially close to the intended nebula row."""
    if not points:
        return []
    cx = float(row['X_pc'])
    cy = float(row['Y_pc'])
    max_sep = _estimate_row_cloud_radius_pc(row)
    kept = [p for p in points if np.hypot(float(p['x']) - cx, float(p['y']) - cy) <= max_sep]
    # If the filter is too aggressive for a large structure, keep the original set.
    return kept if len(kept) >= max(2, min(4, len(points))) else points


def draw_catalog_cloud_group(ax, points: list[dict], *, facecolor: str, alpha: float = 0.14,
                             zorder: float = 4.0, bubble_scale_pc: float = 3.3,
                             bubble_min_pc: float = 10.0, resolution: int = 32) -> bool:
    if not points:
        return False
    geoms = []
    radii = []
    for p in points:
        radius = max(float(bubble_min_pc), float(bubble_scale_pc) * np.sqrt(max(float(p.get('size', 1.0)), 1.0)))
        radii.append(radius)
        geoms.append(Point(float(p['x']), float(p['y'])).buffer(radius, resolution=resolution))

    merged = unary_union(geoms)

    # Join nearby catalogue bubbles into a single cloud footprint when they are
    # touching or separated by only a modest gap.  This keeps hybrid-mode
    # molecular clouds looking like coherent regions rather than many isolated
    # circles.
    median_radius = float(np.median(radii)) if radii else float(bubble_min_pc)
    bridge_gap = max(8.0, min(18.0, 0.42 * median_radius))
    smooth_gap = max(2.0, min(7.0, 0.16 * median_radius))
    try:
        merged = merged.buffer(bridge_gap, resolution=resolution)
        merged = merged.buffer(-bridge_gap, resolution=resolution)
        merged = merged.buffer(smooth_gap, resolution=resolution)
        merged = merged.buffer(-smooth_gap, resolution=resolution)
        merged = merged.buffer(0)
    except Exception:
        merged = unary_union(geoms)

    polys = [merged] if isinstance(merged, Polygon) else list(getattr(merged, 'geoms', []))
    if not polys:
        return False
    for poly in polys:
        xp, yp = poly.exterior.xy
        ax.fill(xp, yp, facecolor=facecolor, edgecolor='none', alpha=alpha, zorder=zorder)
    # A few darker cores from the largest catalogue bubbles help readability without reverting to hand-made blobs.
    largest = sorted(points, key=lambda d: float(d.get('size', 0.0)), reverse=True)[:max(1, min(4, len(points)//3 + 1))]
    core_color = _darken_hex_color(facecolor, 0.72)
    for p in largest:
        radius = max(4.0, 0.40 * max(float(bubble_min_pc), float(bubble_scale_pc) * np.sqrt(max(float(p.get('size', 1.0)), 1.0))))
        core = Point(float(p['x']), float(p['y'])).buffer(radius, resolution=max(12, resolution//2))
        xp, yp = core.exterior.xy
        ax.fill(xp, yp, facecolor=core_color, edgecolor='none', alpha=min(0.22, alpha + 0.03), zorder=zorder + 0.12)
    return True


def _iter_h5_datasets(group, prefix=''):
    if h5py is None:
        return
    for key, item in group.items():
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(item, h5py.Dataset):
            yield name, item
        elif isinstance(item, h5py.Group):
            yield from _iter_h5_datasets(item, prefix=name)


def _find_preferred_3d_h5_dataset(h5file):
    preferred = []
    fallback = []
    for name, ds in _iter_h5_datasets(h5file):
        if getattr(ds, 'ndim', 0) != 3:
            continue
        if not np.issubdtype(ds.dtype, np.number):
            continue
        lname = name.lower()
        rec = (name, ds)
        if any(tok in lname for tok in ['dust', 'ebv', 'ext', 'opacity', 'dens', 'cube', 'map']):
            preferred.append(rec)
        else:
            fallback.append(rec)
    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return None, None


def project_lallement_dust_xy(h5_path: Path | str, *, radius_pc: float = 400.0,
                              mode: str = 'max', sigma: float = 1.1,
                              sphere_only: bool = True):
    """Project the Lallement/STILISM local dust cube onto the X-Y plane.

    The known file structure for map3D_GAIAdr2_feb2019.h5 is::
        /stilism/cube_datas
    with attributes:
        gridstep_values = [5, 5, 5] pc
        sun_position = [600.5, 600.5, 80.5] (voxel indices)

    Parameters
    ----------
    mode : {'max', 'sum'}
        'max' highlights thin structures visually; 'sum' is closer to a true
        integrated dust-column projection.
    sphere_only : bool
        If True, only voxels within the 3D sphere of radius ``radius_pc`` are
        included before projection, rather than the full X-Y cylinder.
    """
    if h5py is None:
        logging.info('h5py not available; skipping Lallement dust projection')
        return None
    h5_path = Path(h5_path)
    if not h5_path.exists():
        logging.info('Dust HDF5 file not found: %s', h5_path)
        return None

    try:
        with h5py.File(h5_path, 'r') as h5:
            if '/stilism/cube_datas' in h5:
                ds = h5['/stilism/cube_datas']
                dname = '/stilism/cube_datas'
            else:
                dname, ds = _find_preferred_3d_h5_dataset(h5)
            if ds is None:
                logging.warning('No numeric 3D dataset found in %s', h5_path)
                return None

            gridstep = np.asarray(ds.attrs.get('gridstep_values', [1, 1, 1]), dtype=float)
            sun_position = np.asarray(ds.attrs.get('sun_position', [s / 2.0 for s in ds.shape]), dtype=float)
            if gridstep.size != 3 or sun_position.size != 3:
                logging.warning('Unexpected cube metadata in %s', h5_path)
                return None

            dx, dy, dz = [float(v) for v in gridstep]
            sx, sy, sz = [float(v) for v in sun_position]

            half_x = int(np.ceil(radius_pc / dx))
            half_y = int(np.ceil(radius_pc / dy))
            half_z = int(np.ceil(radius_pc / dz))

            ix0 = max(0, int(np.floor(sx - half_x)))
            ix1 = min(ds.shape[0], int(np.ceil(sx + half_x + 1)))
            iy0 = max(0, int(np.floor(sy - half_y)))
            iy1 = min(ds.shape[1], int(np.ceil(sy + half_y + 1)))
            iz0 = max(0, int(np.floor(sz - half_z)))
            iz1 = min(ds.shape[2], int(np.ceil(sz + half_z + 1)))

            sub = np.asarray(ds[ix0:ix1, iy0:iy1, iz0:iz1], dtype=float)
    except Exception as exc:
        logging.warning('Could not read dust projection from %s: %s', h5_path, exc)
        return None

    x = (np.arange(ix0, ix1, dtype=float) - sx) * dx
    y = (np.arange(iy0, iy1, dtype=float) - sy) * dy
    z = (np.arange(iz0, iz1, dtype=float) - sz) * dz

    sub = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)

    if sphere_only:
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        mask = (X * X + Y * Y + Z * Z) <= float(radius_pc) ** 2
        sub = np.where(mask, sub, np.nan)

    mode = str(mode).lower().strip()
    if mode == 'sum':
        proj = np.nansum(sub * dz, axis=2)
    elif mode == 'max':
        valid_xy = np.any(np.isfinite(sub), axis=2)
        tmp = np.where(np.isfinite(sub), sub, -np.inf)
        proj = np.max(tmp, axis=2)
        proj[~valid_xy] = 0.0
    else:
        raise ValueError("mode must be 'sum' or 'max'")

    proj = gaussian_filter(np.nan_to_num(proj, nan=0.0), sigma=float(sigma))
    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    return {
        'image': proj,
        'extent': extent,
        'dataset_name': dname,
        'gridstep_pc': (dx, dy, dz),
        'sun_position_index': (sx, sy, sz),
        'mode': mode,
        'sphere_only': bool(sphere_only),
    }


def draw_physical_dust_projection(ax, config: GaiaMapConfig, *, zorder: float = 0.35):
    proj = project_lallement_dust_xy(
        config.dust_h5_path,
        radius_pc=float(config.radius_pc),
        mode=str(getattr(config, 'dust_projection_mode', 'max')),
        sigma=float(CLOUD_RENDER_DEFAULTS.get('projection_sigma', 1.3)),
        sphere_only=bool(getattr(config, 'dust_projection_sphere_only', True)),
    )
    if not proj:
        return False

    image = np.asarray(proj['image'], dtype=float)
    good = image[np.isfinite(image) & (image > 0)]
    logging.info('Dust projection pixels >0: %d / %d', int(good.size), int(image.size))
    if good.size == 0:
        return False

    lo = np.nanpercentile(good, float(getattr(config, 'dust_percentile_lo', 52.0)))
    hi = np.nanpercentile(good, float(getattr(config, 'dust_percentile_hi', 98.7)))
    logging.info('Using Lallement dust projection: dataset=%s mode=%s sphere_only=%s',
                 proj.get('dataset_name'), proj.get('mode'), proj.get('sphere_only'))
    norm = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    # Diagonal split requested by the user: line from (100, 400) to (-300, -400),
    # i.e. y = 2 x + 200  <=>  x = 0.5 y - 100.
    # Right of the diagonal: lower visible contour at 0.25.
    # Left  of the diagonal: lower visible contour at 0.35.
    x0, x1, y0, y1 = [float(v) for v in proj['extent']]
    nx, ny = norm.shape
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    Xg, Yg = np.meshgrid(xs, ys, indexing='ij')

    # Remove one large dust patch south-east of VEL OB2 that is visually
    # distracting in this local 400 pc figure and does not match the intended
    # final presentation.
    suppress = ((Xg - 95.0) / 120.0) ** 2 + ((Yg + 320.0) / 88.0) ** 2 <= 1.0
    norm = np.where(suppress, np.nan, norm)

    x_diag = 0.5 * Yg - 100.0
    mask_right = Xg >= x_diag
    mask_left = ~mask_right

    alpha = float(getattr(config, 'dust_projection_alpha', 0.30))
    color_floor = float(getattr(config, 'dust_color_floor', 0.45))
    upper_greys = LinearSegmentedColormap.from_list(
        'upper_greys_dust', plt.cm.Greys(np.linspace(color_floor, 1.0, 256))
    )

    side_specs = [
        ('left', mask_left, float(getattr(config, 'dust_lowest_contour_left', 0.35))),
        ('right', mask_right, float(getattr(config, 'dust_lowest_contour_right', 0.25))),
    ]

    for _, side_mask, low_level in side_specs:
        side = np.where(side_mask, norm, np.nan)
        if not np.isfinite(side).any():
            continue
        levels = np.linspace(low_level, 1.0, 22)
        try:
            ax.contourf(
                side.T, levels=levels, origin='lower', extent=proj['extent'],
                cmap=upper_greys, alpha=alpha, antialiased=True, zorder=zorder
            )
            ax.contour(
                side.T, levels=[low_level], origin='lower', extent=proj['extent'],
                colors=['#8b6b4a'],
                linewidths=float(getattr(config, 'dust_contour_lw', 1.05)),
                alpha=float(getattr(config, 'dust_contour_alpha', 0.58)),
                zorder=zorder + 0.04,
            )
        except Exception as exc:
            logging.warning('Could not draw smooth H5 dust contours (%s); falling back to imshow.', exc)
            fill = np.full_like(side, np.nan, dtype=float)
            inside = np.isfinite(side) & (side >= low_level)
            fill[inside] = color_floor + (1.0 - color_floor) * (side[inside] - low_level) / max(1.0 - low_level, 1e-6)
            ax.imshow(fill.T, origin='lower', extent=proj['extent'], cmap=upper_greys,
                      vmin=color_floor, vmax=1.0, alpha=alpha, zorder=zorder, interpolation='bilinear')
    return True



def draw_physical_cloud_layer(ax, nebulae: Table | None, config: GaiaMapConfig, *, zorder: float = 4.0) -> set[str]:
    drawn: set[str] = set()
    mode = str(getattr(config, 'cloud_render_mode', 'hybrid')).lower()

    # The H5/STILISM projection does not need the nebula table.  Draw it first
    # so it can be tested with show_nebulae=False and cloud_render_mode='h5'.
    if mode in {'h5', 'hybrid'} and getattr(config, 'show_physical_dust_projection', False):
        t0 = time.perf_counter()
        draw_physical_dust_projection(ax, config, zorder=0.35)
        logging.info('Dust H5 projection step finished in %.2f s', time.perf_counter() - t0)

    if nebulae is None or mode not in {'catalog', 'hybrid'}:
        return drawn
    html_path = getattr(config, 'cloud_catalog_html', None)
    if html_path is None:
        return drawn
    t_cat = time.perf_counter()
    catalog_groups = load_cloud_catalog_from_html(html_path)
    logging.info('Cloud HTML catalog parsed in %.2f s; groups=%d', time.perf_counter() - t_cat, len(catalog_groups) if catalog_groups else 0)
    if not catalog_groups:
        return drawn
    radius_pc = float(config.radius_pc) + float(getattr(config, 'cloud_radius_margin_pc', 40.0))
    for row in nebulae:
        label = str(row['label'])
        aliases = CLOUD_CATALOG_ALIASES.get(label, [])
        if not aliases:
            continue
        pts = []
        for alias in aliases:
            pts.extend(catalog_groups.get(alias, []))
        if not pts:
            continue
        pts = [p for p in pts if float(np.hypot(p['x'], p['y'])) <= radius_pc and float(p['r3d']) <= radius_pc + 100.0]
        pts = _filter_catalog_points_near_row(pts, row)
        if not pts:
            continue
        t_cloud = time.perf_counter()
        ok = draw_catalog_cloud_group(
            ax, pts,
            facecolor=str(row['color']),
            alpha=float(getattr(config, 'nebula_alpha', 0.14)),
            zorder=zorder,
            bubble_scale_pc=float(getattr(config, 'cloud_bubble_scale_pc', CLOUD_RENDER_DEFAULTS.get('bubble_scale_pc', 3.3))),
            bubble_min_pc=float(getattr(config, 'cloud_bubble_min_pc', CLOUD_RENDER_DEFAULTS.get('bubble_min_pc', 10.0))),
            resolution=int(CLOUD_RENDER_DEFAULTS.get('bubble_resolution', 32)),
        )
        if ok:
            drawn.add(label)
            logging.info('Catalog cloud drawn: %s using %d bubbles in %.2f s', label, len(pts), time.perf_counter() - t_cloud)
    return drawn

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def tr(language: str, key: str, **kwargs) -> str:
    lang = language if language in UI_TEXT else "en"
    text = UI_TEXT[lang][key]
    return text.format(**kwargs) if kwargs else text

def get_color_from_Teff(Teff):
    """
    Same basic mapping as before, but with a broader white/near-white regime
    around solar-to-A-star temperatures to match the intended visual style.
    """
    r = 255.0
    g = 255.0
    b = 255.0

    if Teff <= 7500:
        r = 255.0
    else:
        rr = Teff / 100.0
        r = 285.1221695283 * ((rr - 70.75) ** (-0.0755148492))
        r = max(0.0, min(255.0, r))

    if Teff <= 7500:
        gg = Teff / 100.0
        g = 105.6708025861 * np.log(gg) - 200.5827433823815
        g = max(0.0, min(255.0, g))
    else:
        gg = Teff / 100.0
        g = 334.1221695283 * ((gg - 70.75) ** (-0.1855148492))
        g = max(0.0, min(255.0, g))

    if Teff >= 7500:
        b = 255.0
    else:
        if Teff <= 3212:
            b = 0.0
        else:
            bb = Teff / 100.0
            b = ((bb - 32.12) ** 1.478)
            b = max(0.0, min(255.0, b))

    rgb = np.array([r, g, b], dtype=float) / 255.0

    # Wider whitish plateau roughly around F/G/A transition for readability.
    # Peak whitening near ~6500 K and extended between ~5200 and ~8200 K.
    t = np.clip(1.0 - np.abs(float(Teff) - 6500.0) / 1700.0, 0.0, 1.0)
    white_strength = 0.42 * t
    rgb = (1.0 - white_strength) * rgb + white_strength * np.ones(3)

    return tuple(np.clip(rgb, 0.0, 1.0))

def teff_for_star(label: str) -> float:
    return float(STAR_TEFF.get(str(label), 5800.0))

def add_stellar_temperature_colorbar(ax_leg, *, x0: float, y0: float, width: float, height: float, language: str) -> None:
    Teff_range = np.linspace(2000, 10000, 1000)
    colors = np.array([get_color_from_Teff(T) for T in Teff_range])
    cmap = ListedColormap(colors)

    bar_ax = ax_leg.inset_axes([x0, y0, width, height])
    grad = np.linspace(0, 1, 1000)[None, :]
    bar_ax.imshow(grad, aspect='auto', cmap=cmap, extent=[2000, 10000, 0, 1])
    bar_ax.set_yticks([])
    bar_ax.set_xticks([2000, 4000, 6000, 8000, 10000])
    bar_ax.tick_params(axis='x', labelsize=8.7, length=2.5, pad=1.5)
    bar_ax.set_xlabel(tr(language, 'temperature'), fontsize=9.2, labelpad=2)
    for spine in bar_ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor('0.35')

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def finite_xyz_mask(tab: Table) -> np.ndarray:
    return (
        np.isfinite(np.asarray(tab["X_pc"], dtype=float))
        & np.isfinite(np.asarray(tab["Y_pc"], dtype=float))
        & np.isfinite(np.asarray(tab["Z_pc"], dtype=float))
    )

def within_xy_radius_mask(tab: Table, radius_pc: float) -> np.ndarray:
    x = np.asarray(tab["X_pc"], dtype=float)
    y = np.asarray(tab["Y_pc"], dtype=float)
    return np.hypot(x, y) <= radius_pc

def table_with_xyz(rows: list[dict]) -> Table:
    tab = Table(rows=rows)
    coord = SkyCoord(
        ra=np.asarray(tab["ra_deg"], dtype=float) * u.deg,
        dec=np.asarray(tab["dec_deg"], dtype=float) * u.deg,
        distance=np.asarray(tab["distance_pc"], dtype=float) * u.pc,
        frame="icrs",
    )
    gal = coord.galactic
    cart = gal.cartesian
    tab["l_deg"] = gal.l.deg
    tab["b_deg"] = gal.b.deg
    tab["X_pc"] = cart.x.to_value(u.pc)
    tab["Y_pc"] = cart.y.to_value(u.pc)
    tab["Z_pc"] = cart.z.to_value(u.pc)
    return tab

def simbad_icrs_position(simbad: Simbad, object_name: str) -> tuple[float, float] | None:
    try:
        res = simbad.query_object(object_name)
    except Exception as exc:
        logging.warning("SIMBAD failed for %s: %s", object_name, exc)
        return None
    if res is None or len(res) == 0:
        logging.warning("SIMBAD could not resolve %s", object_name)
        return None
    row = res[0]
    cols = {c.lower(): c for c in res.colnames}
    if "ra" not in cols or "dec" not in cols:
        logging.warning("No RA/DEC in SIMBAD result for %s", object_name)
        return None
    ra_value = row[cols["ra"]]
    dec_value = row[cols["dec"]]
    try:
        coord = SkyCoord(float(ra_value) * u.deg, float(dec_value) * u.deg, frame="icrs")
    except Exception:
        coord = SkyCoord(ra_value, dec_value, unit=(u.hourangle, u.deg), frame="icrs")
    return coord.ra.deg, coord.dec.deg

def _new_simbad_client_with_parallax() -> Simbad:
    """Return a SIMBAD client configured to request coordinates and parallax when available."""
    simbad = Simbad()
    try:
        simbad.add_votable_fields("parallax")
    except Exception:
        # Older/newer astroquery/SIMBAD field names can differ. Coordinates will
        # still be queried; distance then falls back to the curated value.
        pass
    return simbad


def simbad_icrs_position_and_distance(
    simbad: Simbad,
    object_name: str,
    fallback_distance_pc: float,
) -> tuple[float, float, float, bool]:
    """
    Resolve an object in SIMBAD and use its parallax when SIMBAD provides one.

    Returns
    -------
    ra_deg, dec_deg, distance_pc, used_simbad_parallax
        If the parallax is unavailable/non-positive, distance_pc falls back to
        fallback_distance_pc while coordinates still come from SIMBAD.
    """
    try:
        res = simbad.query_object(object_name)
    except Exception as exc:
        logging.warning("SIMBAD failed for %s: %s", object_name, exc)
        raise

    if res is None or len(res) == 0:
        raise ValueError(f"SIMBAD could not resolve {object_name}")

    row = res[0]
    cols = {c.lower(): c for c in res.colnames}
    if "ra" not in cols or "dec" not in cols:
        raise ValueError(f"No RA/DEC in SIMBAD result for {object_name}")

    ra_value = row[cols["ra"]]
    dec_value = row[cols["dec"]]
    try:
        coord = SkyCoord(float(ra_value) * u.deg, float(dec_value) * u.deg, frame="icrs")
    except Exception:
        coord = SkyCoord(ra_value, dec_value, unit=(u.hourangle, u.deg), frame="icrs")

    parallax_mas = np.nan
    for key in ("plx_value", "plx", "parallax", "plx_value_2"):
        if key in cols:
            parallax_mas = safe_float(row[cols[key]])
            break

    if np.isfinite(parallax_mas) and parallax_mas > 0:
        distance_pc = 1000.0 / parallax_mas
        used_parallax = True
    else:
        distance_pc = float(fallback_distance_pc)
        used_parallax = False

    return coord.ra.deg, coord.dec.deg, distance_pc, used_parallax


def age_to_color(age_myr: float) -> str:
    if not np.isfinite(age_myr):
        return "white"
    if age_myr < 100:
        return "#4cc9f0"
    if age_myr < 600:
        return "#f9c74f"
    return "#f3722c"

def kind_to_marker(kind: str) -> str:
    return {"open_cluster": "o", "association": "D", "sco_cen": "^"}.get(str(kind), "o")

def system_marker_size(label: str, kind: str) -> float:
    """Slightly larger markers for compact catalogue clusters and Sco-Cen."""
    label_plain = str(label).replace("$", "")
    if label_plain == '32 Ori':
        return 290.0
    if str(kind) == "sco_cen":
        return 335.0
    if label_plain.startswith("IC ") or label_plain.startswith("NGC "):
        return 315.0
    if str(kind) == "open_cluster":
        return 255.0
    return 225.0

def system_code_fontsize(code: str) -> float:
    """Readable code text inside markers; smaller for long numeric labels."""
    n = len(str(code))
    if n <= 2:
        return 7.4
    if n == 3:
        return 6.9
    return 6.0

def signed_z(z_pc: float) -> str:
    sign = "+" if z_pc >= 0 else "−"
    return f"{sign}{abs(z_pc):.0f}"

def short_star_label(name: str, z_pc: float) -> str:
    return f"{name} ({signed_z(z_pc)})"

def add_xy_distance_circles(ax, radii=(50, 100, 200)) -> None:
    theta = np.linspace(0, 2 * np.pi, 720)
    for rr in radii:
        ax.plot(rr * np.cos(theta), rr * np.sin(theta), color="0.56", lw=0.9 if rr < max(radii) else 1.15,
                ls="--", alpha=0.45 if rr < max(radii) else 0.70, zorder=0)
        tx = -8 if rr == 100 else 4
        ha = "right" if rr == 100 else "left"
        txt = ax.text(tx, rr + 3, f"{rr} pc", fontsize=8.8, color="0.38", ha=ha, va="bottom", zorder=1)
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)])

def add_occupied_marker_boxes(ax, xs: Iterable[float], ys: Iterable[float], size_px: float = 11.0) -> list:
    from matplotlib.transforms import Bbox
    boxes = []
    for x, y in zip(xs, ys):
        xp, yp = ax.transData.transform((float(x), float(y)))
        boxes.append(Bbox.from_bounds(xp - size_px / 2, yp - size_px / 2, size_px, size_px))
    return boxes

def bboxes_overlap(b1, b2, pad_px: float = 2.0) -> bool:
    return not (b1.x1 + pad_px < b2.x0 or b1.x0 - pad_px > b2.x1 or b1.y1 + pad_px < b2.y0 or b1.y0 - pad_px > b2.y1)

def draw_fast_leader_line(ax, x: float, y: float, dx: float, dy: float, *,
                          color='0.45', alpha=0.35, lw=0.35,
                          zorder: float = 95.0, shrink_data: float = 2.0) -> None:
    """Draw a cheap straight label leader, avoiding Matplotlib arrow_patch.

    ``dx`` and ``dy`` are offset-points, matching ``ax.annotate(...,
    textcoords='offset points')``.  We transform that offset into data
    coordinates and draw a simple line.  This is much faster than arrowprops
    with many labels.
    """
    try:
        fig = ax.figure
        xpix, ypix = ax.transData.transform((float(x), float(y)))
        scale = fig.dpi / 72.0
        x2, y2 = ax.transData.inverted().transform((xpix + float(dx) * scale, ypix + float(dy) * scale))
        vx, vy = x2 - float(x), y2 - float(y)
        rr = float(np.hypot(vx, vy))
        if rr <= 1e-9:
            return
        x1 = float(x) + shrink_data * vx / rr
        y1 = float(y) + shrink_data * vy / rr
        x2 = x2 - shrink_data * vx / rr
        y2 = y2 - shrink_data * vy / rr
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, lw=lw,
                zorder=zorder, solid_capstyle='round')
    except Exception:
        pass


def candidate_offsets_for_position(x: float, y: float) -> list[tuple[int, int]]:
    """
    Candidate label offsets in screen points.

    The offsets are generated around the radial direction from the Sun to the
    star, rather than using the same mirrored pattern everywhere. This gives a
    much more organic label field and uses the empty angular sectors better.
    """
    theta = float(np.arctan2(y, x)) if (abs(x) + abs(y)) > 1e-9 else np.deg2rad(-35.0)
    angle_offsets_deg = [0, 25, -25, 50, -50, 75, -75, 105, -105, 140, -140, 180]
    radii_pts = [18, 28, 40, 56, 76, 100, 126, 154]

    out: list[tuple[int, int]] = []
    for rr in radii_pts:
        for dphi in angle_offsets_deg:
            ang = theta + np.deg2rad(dphi)
            dx = int(np.round(rr * np.cos(ang)))
            dy = int(np.round(rr * np.sin(ang)))
            if abs(dx) < 8 and abs(dy) < 8:
                continue
            pair = (dx, dy)
            if pair not in out:
                out.append(pair)
    return out

def automatic_annotate(ax, items: list[dict], *, fontsize: float, occupied: list | None = None,
                       max_labels: int | None = None, draw_leaders: bool = True) -> list:
    if occupied is None:
        occupied = []
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = 0

    for item in sorted(items, key=lambda d: d.get("priority", 999)):
        if max_labels is not None and placed >= max_labels:
            break
        x, y, text = float(item["x"]), float(item["y"]), str(item["text"])
        offsets = item.get("offsets", candidate_offsets_for_position(x, y))

        # Fast path: avoid Matplotlib arrow_patch and repeated candidate
        # redraws.  We still draw a cheap straight leader line if requested.
        item_draw_leaders = item.get('draw_leaders', draw_leaders)
        if item.get('fast_leaders', True):
            dx, dy = offsets[0] if offsets else (6, 6)
            if item_draw_leaders:
                draw_fast_leader_line(ax, x, y, dx, dy,
                                      color=item.get('leader_color', '0.45'),
                                      alpha=item.get('leader_alpha', 0.35),
                                      lw=item.get('leader_lw', 0.35),
                                      zorder=item.get('leader_zorder', item.get('marker_zorder', 105) - 1.0),
                                      shrink_data=item.get('leader_shrinkB', 3))
            ann = ax.annotate(
                text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=fontsize, ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top", color=item.get("color", "0.10"),
                alpha=item.get("alpha", 0.9), zorder=item.get("zorder", 100),
                arrowprops=None,
            )
            ann.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.95)])
            item["_placed"] = True
            placed += 1
            continue

        # For a few persistent conflicts, enforce the first manual offset.
        if item.get("fixed_offset", False) and offsets:
            dx, dy = offsets[0]
            ann = ax.annotate(
                text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=fontsize, ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top", color=item.get("color", "0.10"),
                alpha=item.get("alpha", 0.9), zorder=item.get("zorder", 100),
                arrowprops=(dict(arrowstyle='-', lw=item.get('leader_lw', 0.35), color=item.get('leader_color', '0.45'), alpha=item.get('leader_alpha', 0.35),
                                 shrinkA=0, shrinkB=item.get('leader_shrinkB', 3)) if draw_leaders else None)
            )
            ann.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.95)])
            fig.canvas.draw()
            occupied.append(ann.get_window_extent(renderer=renderer).expanded(1.05, 1.12))
            item["_placed"] = True
            placed += 1
            continue

        best = None
        best_score = np.inf

        item_draw_leaders = item.get('draw_leaders', draw_leaders)
        for dx, dy in offsets:
            ann = ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                              fontsize=fontsize, ha="left" if dx >= 0 else "right",
                              va="bottom" if dy >= 0 else "top", color=item.get("color", "0.10"),
                              alpha=item.get("alpha", 0.9), zorder=item.get("zorder", 100),
                              arrowprops=(dict(arrowstyle='-', lw=item.get('leader_lw', 0.35), color=item.get('leader_color', '0.45'), alpha=item.get('leader_alpha', 0.35),
                                               shrinkA=0, shrinkB=item.get('leader_shrinkB', 3))
                                          if item_draw_leaders else None))
            ann.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.95)])
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.03, 1.08)
            inside = ax.bbox.contains(bbox.x0, bbox.y0) and ax.bbox.contains(bbox.x1, bbox.y1)
            collision = any(bboxes_overlap(bbox, old, pad_px=2.0) for old in occupied)
            ann.remove()
            score = dx * dx + dy * dy
            if not inside:
                score += 1e8
            if collision:
                score += 1e7
            if score < best_score:
                best = (dx, dy)
                best_score = score
            if inside and not collision:
                break

        if best is None:
            continue
        if best_score >= 1e7 and not item.get("force", False):
            continue

        dx, dy = best
        ann = ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                          fontsize=fontsize, ha="left" if dx >= 0 else "right",
                          va="bottom" if dy >= 0 else "top", color=item.get("color", "0.10"),
                          alpha=item.get("alpha", 0.9), zorder=item.get("zorder", 100),
                          arrowprops=(dict(arrowstyle='-', lw=item.get('leader_lw', 0.35), color=item.get('leader_color', '0.45'), alpha=item.get('leader_alpha', 0.35),
                                           shrinkA=0, shrinkB=item.get('leader_shrinkB', 3))
                                      if item_draw_leaders else None))
        ann.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.95)])
        fig.canvas.draw()
        occupied.append(ann.get_window_extent(renderer=renderer).expanded(1.05, 1.12))
        item["_placed"] = True
        placed += 1
    return occupied

def annotate_label(ax, x, y, text, *, fontsize=8.5, color='0.25',
                   dx=4, dy=4, ha='left', va='bottom', zorder=20,
                   draw_leaders=True, fast_leaders=True,
                   leader_color='0.45', leader_alpha=0.35,
                   leader_lw=0.35, leader_shrinkB=3):
    cfg = MANUAL_LABEL_OVERRIDES.get(text)
    arrowprops = None if fast_leaders else (dict(arrowstyle='-', lw=leader_lw, color=leader_color, alpha=leader_alpha, shrinkA=0, shrinkB=leader_shrinkB) if draw_leaders else None)
    if cfg is not None and 'x' in cfg and 'y' in cfg:
        ann = ax.annotate(text, xy=(x, y), xytext=(cfg['x'], cfg['y']), textcoords='data',
                          fontsize=cfg.get('fontsize', fontsize), color=color,
                          ha=cfg.get('ha', 'left'), va=cfg.get('va', 'center'),
                          zorder=zorder, arrowprops=arrowprops)
    else:
        if cfg is not None:
            dx = cfg.get('dx', dx)
            dy = cfg.get('dy', dy)
            ha = cfg.get('ha', ha)
            va = cfg.get('va', va)
            fontsize = cfg.get('fontsize', fontsize)
        ann = ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords='offset points',
                          fontsize=fontsize, color=color, ha=ha, va=va,
                          zorder=zorder, arrowprops=arrowprops)
    if draw_leaders and fast_leaders:
        if cfg is not None and 'x' in cfg and 'y' in cfg:
            # Data-position labels: draw directly from marker to label anchor.
            ax.plot([float(x), float(cfg['x'])], [float(y), float(cfg['y'])],
                    color=leader_color, alpha=leader_alpha, lw=leader_lw,
                    zorder=zorder - 15.0, solid_capstyle='round')
        else:
            draw_fast_leader_line(ax, x, y, dx, dy, color=leader_color,
                                  alpha=leader_alpha, lw=leader_lw,
                                  zorder=zorder - 15.0,
                                  shrink_data=float(leader_shrinkB))
    ann.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white', alpha=0.95)])
    return ann

def annotate_items(ax, items: list[dict], *, fontsize: float, occupied: list | None = None,
                   max_labels: int | None = None, draw_leaders: bool = True) -> list:
    if occupied is None:
        occupied = []
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    manual_items = []
    auto_items = []
    for item in items:
        key = item.get('label_key', item.get('text'))
        if key in MANUAL_LABEL_OVERRIDES:
            manual_items.append(item)
        else:
            auto_items.append(item)
    for item in sorted(manual_items, key=lambda d: d.get('priority', 999)):
        key = item.get('label_key', item.get('text'))
        visible_text = str(item['text'])
        cfg = MANUAL_LABEL_OVERRIDES.get(key)
        if cfg is not None:
            old_cfg = MANUAL_LABEL_OVERRIDES.get(visible_text)
            MANUAL_LABEL_OVERRIDES[visible_text] = cfg
            ann = annotate_label(ax, float(item['x']), float(item['y']), visible_text,
                                 fontsize=item.get('fontsize', fontsize),
                                 color=item.get('color', '0.10'),
                                 zorder=item.get('zorder', 100),
                                 draw_leaders=item.get('draw_leaders', draw_leaders),
                                 fast_leaders=True,
                                 leader_color=item.get('leader_color', '0.45'),
                                 leader_alpha=item.get('leader_alpha', 0.35),
                                 leader_lw=item.get('leader_lw', 0.35),
                                 leader_shrinkB=item.get('leader_shrinkB', 3))
            if old_cfg is None:
                MANUAL_LABEL_OVERRIDES.pop(visible_text, None)
            else:
                MANUAL_LABEL_OVERRIDES[visible_text] = old_cfg
        else:
            ann = annotate_label(ax, float(item['x']), float(item['y']), visible_text,
                                 fontsize=item.get('fontsize', fontsize),
                                 color=item.get('color', '0.10'),
                                 zorder=item.get('zorder', 100),
                                 draw_leaders=item.get('draw_leaders', draw_leaders),
                                 fast_leaders=True,
                                 leader_color=item.get('leader_color', '0.45'),
                                 leader_alpha=item.get('leader_alpha', 0.35),
                                 leader_lw=item.get('leader_lw', 0.35),
                                 leader_shrinkB=item.get('leader_shrinkB', 3))
        fig.canvas.draw()
        occupied.append(ann.get_window_extent(renderer=renderer).expanded(1.05, 1.12))
        item['_placed'] = True
    occupied = automatic_annotate(ax, auto_items, fontsize=fontsize, occupied=occupied,
                                  max_labels=max_labels, draw_leaders=draw_leaders)
    return occupied

def select_distributed_labelled_stars(
    bright_stars: Table,
    *,
    max_outer_labels: int = 24,
    min_sep_pc: float = 28.0,
) -> Table:
    """
    Select a visually useful, spatially distributed set of star labels.

    Rules:
    - Inside 30 pc, label only the iconic set in INNER_30PC_LABELLED_NAMES.
    - Outside 30 pc, greedily select bright named stars while rewarding empty
      X-Y sectors and penalising labels too close to already selected stars.
    - Mandatory outer stars are included first, then the map is filled by the
      coverage score.
    """
    rows = [row for row in bright_stars]
    selected = []
    selected_names = set()

    def add(row) -> None:
        name = str(row["label"])
        if name not in selected_names:
            selected.append(row)
            selected_names.add(name)

    # Inner iconic labels only.
    for row in sorted(rows, key=lambda r: float(r["vmag"])):
        d = float(row["distance_pc"])
        name = str(row["label"])
        if d <= 30.0 and name in INNER_30PC_LABELLED_NAMES:
            add(row)

    # Mandatory outer labels.
    for row in sorted(rows, key=lambda r: float(r["vmag"])):
        d = float(row["distance_pc"])
        name = str(row["label"])
        if d > 30.0 and name in FORCE_BRIGHT_STAR_NAMES:
            add(row)

    outer_candidates = [
        row for row in rows
        if float(row["distance_pc"]) > 30.0 and str(row["label"]) not in selected_names
    ]

    # Track angular/radial occupancy to fill the full map rather than the core.
    sector_count: dict[tuple[int, int], int] = {}
    for row in selected:
        x, y = float(row["X_pc"]), float(row["Y_pc"])
        rr = np.hypot(x, y)
        theta = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
        sector = int(theta // 45.0)
        ring = 0 if rr < 70 else (1 if rr < 125 else 2)
        sector_count[(sector, ring)] = sector_count.get((sector, ring), 0) + 1

    while outer_candidates and sum(float(r["distance_pc"]) > 30.0 for r in selected) < max_outer_labels:
        best_idx = None
        best_score = -np.inf

        for i, row in enumerate(outer_candidates):
            x, y = float(row["X_pc"]), float(row["Y_pc"])
            rr = np.hypot(x, y)
            theta = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
            sector = int(theta // 45.0)
            ring = 0 if rr < 70 else (1 if rr < 125 else 2)
            vmag = float(row["vmag"])

            if selected:
                dmin = min(np.hypot(x - float(s["X_pc"]), y - float(s["Y_pc"])) for s in selected)
            else:
                dmin = 999.0

            if dmin < min_sep_pc:
                proximity_penalty = 55.0 * (min_sep_pc - dmin) / min_sep_pc
            else:
                proximity_penalty = 0.0

            # Rewards: bright, far from selected labels, and in empty sectors.
            empty_sector_bonus = 42.0 / (1.0 + sector_count.get((sector, ring), 0))
            outer_bonus = 0.08 * rr
            bright_bonus = 22.0 / (vmag + 2.2)

            score = empty_sector_bonus + 0.55 * dmin + outer_bonus + bright_bonus - proximity_penalty

            # Extra encouragement for underused upper-right part of the plot.
            if x > 20 and y > 20:
                score += 20.0

            if score > best_score:
                best_score = score
                best_idx = i

        chosen = outer_candidates.pop(best_idx)
        add(chosen)
        x, y = float(chosen["X_pc"]), float(chosen["Y_pc"])
        rr = np.hypot(x, y)
        theta = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
        sector = int(theta // 45.0)
        ring = 0 if rr < 70 else (1 if rr < 125 else 2)
        sector_count[(sector, ring)] = sector_count.get((sector, ring), 0) + 1

    selected.sort(key=lambda r: float(r["vmag"]))
    return Table(rows=selected, names=bright_stars.colnames)

def system_legend_lines(systems: Table, codes: dict[str, str]) -> list[str]:
    lines = []
    kind_order = {"open_cluster": 0, "sco_cen": 1}
    rows = sorted(list(systems), key=lambda r: (kind_order.get(str(r['kind']), 9), str(r['label'])))
    for row in rows:
        label = str(row['label'])
        z = signed_z(float(row['Z_pc']))
        age = float(row['age_myr'])
        age_txt = f"{age:.0f} Myr" if np.isfinite(age) else "age n/a"
        lines.append(f"{codes[label]:>4s}  {label:<13s}  Z={z:>4s} pc   {age_txt}")
    return lines

def sky_ellipse_xy(ra_deg: float, dec_deg: float, distance_pc: float,
                   a_deg: float, b_deg: float, pa_deg: float = 0.0,
                   n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    from astropy.coordinates import SkyOffsetFrame
    center = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg,
                      distance=float(distance_pc) * u.pc, frame='icrs')
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    lon = float(a_deg) * np.cos(theta)
    lat = float(b_deg) * np.sin(theta)
    off = SkyOffsetFrame(origin=center, rotation=float(pa_deg) * u.deg)
    pts = SkyCoord(lon=lon * u.deg, lat=lat * u.deg,
                   distance=np.full(theta.size, float(distance_pc)) * u.pc,
                   frame=off).transform_to('icrs')
    gal = pts.galactic.cartesian
    return gal.x.to_value(u.pc), gal.y.to_value(u.pc)

def region_alpha_for_kind(kind: str, base_alpha: float = 0.16) -> float:
    if kind == 'open_cluster':
        return base_alpha + 0.04
    if kind == 'sco_cen':
        return base_alpha + 0.03
    if kind == 'association':
        return base_alpha + 0.01
    return base_alpha

def convex_hull_polygon(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 3:
        return pts[:, 0], pts[:, 1]
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        return hull_pts[:, 0], hull_pts[:, 1]
    except Exception:
        return pts[:, 0], pts[:, 1]


def get_system_region_spec(label: str, kind: str) -> dict | None:
    """
    Fallback X-Y footprint for systems without an entry in SYSTEM_SKY_REGION_SPECS.

    Most active systems are rendered through sky-projected ellipses in
    draw_real_projected_region(...). This fallback is only used for any
    system missing from SYSTEM_SKY_REGION_SPECS, so the script does not fail.
    """
    if kind == 'open_cluster':
        return {'a_pc': 12.0, 'b_pc': 9.0, 'angle_deg': 0.0}
    if kind == 'sco_cen':
        return {'a_pc': 34.0, 'b_pc': 16.0, 'angle_deg': -15.0}
    if kind == 'association':
        return {'a_pc': 28.0, 'b_pc': 14.0, 'angle_deg': -15.0}
    return None


def draw_projected_system_region(
    ax,
    *,
    x: float,
    y: float,
    label: str,
    kind: str,
    facecolor,
    edgecolor='none',
    alpha: float = 0.16,
    zorder: float = 8.0,
) -> None:
    """
    Draw a simple fallback ellipse directly in the final X-Y plane.

    This function is needed by draw_real_projected_region(...) whenever a
    system has no sky-footprint entry in SYSTEM_SKY_REGION_SPECS.
    """
    spec = get_system_region_spec(str(label), str(kind))
    if spec is None:
        return

    patch = Ellipse(
        xy=(float(x), float(y)),
        width=2.0 * float(spec['a_pc']),
        height=2.0 * float(spec['b_pc']),
        angle=float(spec.get('angle_deg', 0.0)),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8 if edgecolor != 'none' else 0.0,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)

def draw_real_projected_region(ax, row, *, facecolor: str, alpha: float = 0.16,
                               edgecolor: str = '0.35', linewidth: float = 0.9,
                               zorder: float = 8.0, use_hull: bool = False):
    label = str(row['label'])
    spec = SYSTEM_SKY_REGION_SPECS.get(label)
    if spec is None:
        draw_projected_system_region(ax,
                                     x=float(row['X_pc']), y=float(row['Y_pc']),
                                     label=label, kind=str(row['kind']),
                                     facecolor=facecolor, edgecolor=edgecolor,
                                     alpha=alpha, zorder=zorder)
        return
    xp, yp = sky_ellipse_xy(float(row['ra_deg']), float(row['dec_deg']), float(row['distance_pc']),
                            spec['a_deg'], spec['b_deg'], spec.get('pa_deg', 0.0))
    if use_hull:
        xp, yp = convex_hull_polygon(xp, yp)
    ax.fill(xp, yp, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
            alpha=alpha, zorder=zorder)
    ax.plot(np.r_[xp, xp[0]], np.r_[yp, yp[0]], color=edgecolor, lw=max(0.6, linewidth - 0.2),
            alpha=min(0.75, alpha + 0.12), zorder=zorder + 0.03)

def chaikin_smooth_closed(xp, yp, refinements: int = 3):
    pts = np.column_stack([xp, yp])
    if len(pts) < 3:
        return xp, yp
    for _ in range(refinements):
        new_pts = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        pts = np.asarray(new_pts)
    return pts[:, 0], pts[:, 1]

def draw_member_region(ax, members: Table, *, facecolor: str, edgecolor: str = '0.30',
                       alpha: float = 0.15, zorder: float = 8.0, smooth: bool = True,
                       show_member_points: bool = False):
    if members is None or len(members) == 0:
        return
    xx = np.asarray(members['X_pc'], dtype=float)
    yy = np.asarray(members['Y_pc'], dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[mask], yy[mask]
    if len(xx) < 3:
        if show_member_points:
            ax.scatter(xx, yy, s=12, facecolor=facecolor, edgecolor='0.15', linewidth=0.4,
                       alpha=0.80, zorder=zorder + 0.1)
        return
    xp, yp = convex_hull_polygon(xx, yy)
    if smooth and len(xp) >= 3:
        xp, yp = chaikin_smooth_closed(xp, yp, refinements=3)
    ax.fill(xp, yp, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8,
            alpha=alpha, zorder=zorder)
    ax.plot(np.r_[xp, xp[0]], np.r_[yp, yp[0]], color=edgecolor, lw=0.7,
            alpha=min(alpha + 0.15, 0.55), zorder=zorder + 0.02)
    if show_member_points:
        ax.scatter(xx, yy, s=11, facecolor=facecolor, edgecolor='0.15', linewidth=0.35,
                   alpha=0.62, zorder=zorder + 0.10)

def safe_float(value, default=np.nan) -> float:
    """Convert normal, None, masked and non-finite values safely to float."""
    try:
        if value is None:
            return float(default)
        if np.ma.is_masked(value):
            return float(default)
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)

def safe_int(value, default=-1) -> int:
    """Convert normal, None and masked integer-like values safely."""
    try:
        if value is None:
            return int(default)
        if np.ma.is_masked(value):
            return int(default)
        return int(value)
    except Exception:
        return int(default)

def empty_gaia_density_table() -> Table:
    """Return an empty Gaia-density table with the correct columns and dtypes."""
    return Table(
        names=[
            'source_id', 'distance_pc', 'phot_g_mean_mag', 'bp_rp',
            'ra_deg', 'dec_deg', 'l_deg', 'b_deg', 'X_pc', 'Y_pc', 'Z_pc'
        ],
        dtype=[
            'i8', 'f8', 'f8', 'f8',
            'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'
        ],
    )

def make_unique_system_codes(systems: Table) -> dict[str, str]:
    preferred = {
        'Pleiades': 'P',
        'Hyades': 'H',
        r'$\alpha$ Per': 'aP',
        'Praesepe': 'Pr',
        'IC 2391': '2391',
        'IC 2602': '2602',
        '2451A': '2451A',
        'Mamajek 2': 'Ma2',
        'Platais 3': 'Pl3',
        'Platais 8': 'Pl8',
        'Platais 9': 'Pl9',
        'Cl Alessi 13': 'Al13',
        'Blanco 1': 'B1',
        'IC 4665': '4665',
        'IC 348': '348',
        'Collinder 350': '350',
        'Ruprecht 147': '147',
        'M39': 'M39',
        '1333': '1333',
        'Stock 2': 'St2',
        'Platais 10': 'Pl10',
        'Cl Alessi 9': 'Al9',
        'Cl Alessi 5': 'Al5',
        '1901': '1901',
        'UPK 552': '552',
        'UPK 545': '545',
        'UPK 533': '533',
        'Pozzo 1': 'Poz1',
        '2547': '2547',
        '2451B': '2451B',
        'Collinder 140': '140',
        'Collinder 135': '135',
        'Cl Alessi 3': 'Al3',
        '2516': '2516',
        'Collinder 69': '69',
        'Stock 10': 'St10',
        'UPK 305': '305',
        'Roslund 6': 'R6',
        'UPK 88': '88',
        'Messier 7': 'M7',
        'UPK 612': '612',
        'UPK 624': '624',

        '7058': '7058',
        '6633': '6633',
        'Stephenson 1': 'St1p',
        '101': '101',
        'St1': 'St1',
        'U9': 'U9',
        'U32': 'U32',
        '99': '99',
        '19': '19',
        'RSG1': 'RSG1',
        'U31': 'U31',
        '127': '127',
        '123': '123',
        'RSG5': 'RSG5',
        'U1': 'U1',

        r'$\eta$ Cha': 'eC',
        r'$\epsilon$ Cha': 'epsC',
        r'$\beta$ Pic MG': 'bP',
        'TWA': 'TWA',
        'AB Dor': 'ABD',
        '32 Ori': '32 O',
        'Tuc-Hor': 'TH',
        'Carina': 'Car',
        'Vol-Car': 'VC',
        'Upper Sco': 'US',
        'UCL': 'UCL',
        'LCC': 'LCC',
    }
    codes = {}
    used = set()
    for row in systems:
        label = str(row['label'])
        code = preferred.get(label)
        if code is None:
            if label.startswith('NGC '):
                code = label.replace('NGC ', '').strip()
            else:
                pieces = ''.join(ch for ch in label if ch.isalnum())
                code = pieces[:3] if len(pieces) >= 3 else pieces
                if not code:
                    code = 'SYS'
        base = code
        i = 2
        while code in used:
            code = f'{base}{i}'
            i += 1
        used.add(code)
        codes[label] = code
    return codes

def draw_gaia_density_map(ax, gaia_sample: Table, *, radius_pc: float, zorder: float = 1.0):
    """
    Build a smooth Gaia density-contrast layer in the X-Y plane.

    The map intentionally keeps low-level extended structure so the final
    appearance resembles a broad local-density envelope instead of only a few
    isolated overdensity islands.
    """
    if gaia_sample is None or len(gaia_sample) < 10:
        return None

    x = np.asarray(gaia_sample['X_pc'], dtype=float)
    y = np.asarray(gaia_sample['Y_pc'], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x * x + y * y <= radius_pc * radius_pc)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return None

    rr = np.hypot(x, y)
    if float(radius_pc) > 220.0:
        bins = 440
        sigma_main = 7.0
        sigma_bg = 22.0
        sigma_final = 3.0
        lo_pct, hi_pct = 20.0, 98.3
        # Mild radial de-biasing so the 400 pc map does not pile contrast up
        # only near the external boundary.
        weights = 1.0 / np.sqrt(0.22 + (rr / radius_pc) ** 2)
    else:
        bins = 320
        sigma_main = 5.8
        sigma_bg = 17.0
        sigma_final = 2.6
        lo_pct, hi_pct = 40.0, 98.5
        weights = np.ones_like(x, dtype=float)

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=bins,
        range=[[-radius_pc, radius_pc], [-radius_pc, radius_pc]],
        weights=weights,
    )
    H = gaussian_filter(H.T, sigma=sigma_main)
    if not np.isfinite(H).any() or np.nanmax(H) <= 0:
        return None

    H_bg = gaussian_filter(H, sigma=sigma_bg)
    contrast = H / np.maximum(H_bg, 1e-6)
    contrast = gaussian_filter(contrast, sigma=sigma_final)

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    Rg = np.sqrt(Xg * Xg + Yg * Yg)

    contrast[Rg > radius_pc] = np.nan
    finite = contrast[np.isfinite(contrast)]
    if finite.size < 10:
        return None

    lo = np.nanpercentile(finite, lo_pct)
    hi = np.nanpercentile(finite, hi_pct)
    if not np.isfinite(hi) or hi <= lo:
        return None
    Z = np.clip((contrast - lo) / (hi - lo), 0.0, 1.0)

    levels = [0.06, 0.12, 0.20, 0.30, 0.42, 0.58, 0.76, 1.01]
    ax.contourf(
        Xg, Yg, Z,
        levels=levels,
        cmap=_clean_GAIA_DENSITY_CMAP,
        alpha=(0.26 if float(radius_pc) > 220.0 else 0.20),
        antialiased=True,
        zorder=zorder,
    )
    ax.contour(
        Xg, Yg, Z,
        levels=[0.10, 0.22, 0.40],
        colors=['#8aa8dd'],
        linewidths=0.24,
        alpha=(0.22 if float(radius_pc) > 220.0 else 0.18),
        zorder=zorder + 0.02,
    )

    sm = plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
        cmap=_clean_GAIA_DENSITY_CMAP,
    )
    sm.set_array([])
    return sm

def add_gaia_density_colorbar(ax_leg, *, x0: float, y0: float, width: float, height: float, language: str) -> None:
    label = {'es': 'Contraste Gaia relativo', 'en': 'Relative Gaia contrast'}.get(language, 'Relative Gaia contrast')
    bar_ax = ax_leg.inset_axes([x0, y0, width, height])
    grad = np.linspace(0, 1, 600)
    rgba = _clean_GAIA_DENSITY_CMAP(grad)
    rgba[..., -1] = 0.24
    bar_ax.imshow(rgba[None, :, :], aspect='auto', extent=[0, 1, 0, 1])
    bar_ax.set_yticks([])
    bar_ax.set_xticks([0.0, 0.5, 1.0])
    bar_ax.set_xticklabels(['0', '0.5', '1'])
    bar_ax.tick_params(axis='x', labelsize=8.7, length=2.5, pad=1.5)
    bar_ax.set_xlabel(label, fontsize=9.2, labelpad=2)
    for spine in bar_ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor('0.35')

def resolve_systems(config: GaiaMapConfig, overwrite: bool = False) -> Table:
    """
    Resolve/load the curated nearby systems table.

    For the added 400 pc clusters, coordinates are resolved with SIMBAD and,
    whenever SIMBAD provides a valid parallax, the distance is recomputed as
    d[pc] = 1000 / parallax[mas]. If no parallax is available for a cluster,
    the curated distance_pc in SYSTEMS_400PC_EXTRA is used as a fallback.
    """
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.cache_dir / config.system_cache_name

    if cache_path.exists() and not overwrite:
        logging.info("Loading cached systems: %s", cache_path)
        tab = Table.read(cache_path, format="ascii.ecsv")
        missing = _cache_missing_labels(tab, _expected_system_labels(config))
        if missing:
            logging.info('System cache missing %d labels (%s); rebuilding cache.', len(missing), ', '.join(missing[:12]))
        else:
            return tab

    base_objects = []
    seen = set()
    for obj in list(NEARBY_SYSTEMS) + list(SYSTEMS_400PC_EXTRA):
        label = str(obj.get("label", ""))
        if label in seen:
            continue
        seen.add(label)
        base_objects.append(obj)

    simbad = _new_simbad_client_with_parallax()
    rows = []

    extra_labels = {str(obj["label"]) for obj in SYSTEMS_400PC_EXTRA}

    for obj in base_objects:
        label = str(obj["label"])
        fallback_distance_pc = float(obj["distance_pc"])

        if fallback_distance_pc > float(config.radius_pc) + 60.0:
            continue

        logging.info("Resolving system %s", label)

        is_extra_400pc = label in extra_labels
        resolver = str(obj.get("resolver", "manual"))

        if is_extra_400pc or resolver == "simbad":
            # For new 400 pc clusters, trust SIMBAD for the sky position and use
            # SIMBAD parallax when available. If SIMBAD cannot resolve the name,
            # fall back to the curated RA/Dec/distance stored in the object.
            try:
                ra_deg, dec_deg, distance_pc, used_plx = simbad_icrs_position_and_distance(
                    simbad,
                    str(obj.get("simbad_name", obj["name"])),
                    fallback_distance_pc=fallback_distance_pc,
                )
                obj_out = {**obj, "ra_deg": ra_deg, "dec_deg": dec_deg, "distance_pc": distance_pc}
                obj_out["simbad_parallax_used"] = bool(used_plx)
                if used_plx:
                    logging.info("%s: SIMBAD parallax distance = %.1f pc", label, distance_pc)
                else:
                    logging.info("%s: SIMBAD coordinates; fallback distance = %.1f pc", label, distance_pc)
            except Exception as exc:
                logging.warning("%s: using curated fallback because SIMBAD failed (%s)", label, exc)
                obj_out = {**obj}
                obj_out["simbad_parallax_used"] = False
        elif resolver == "manual":
            obj_out = {**obj}
            obj_out["simbad_parallax_used"] = False
        else:
            pos = simbad_icrs_position(simbad, obj["name"])
            if pos is None:
                continue
            ra_deg, dec_deg = pos
            obj_out = {**obj, "ra_deg": ra_deg, "dec_deg": dec_deg}
            obj_out["simbad_parallax_used"] = False

        if float(obj_out["distance_pc"]) > float(config.radius_pc) + 60.0:
            logging.info("%s skipped after distance check: %.1f pc", label, float(obj_out["distance_pc"]))
            continue

        rows.append(obj_out)

    out = table_with_xyz(rows)
    out.write(cache_path, format="ascii.ecsv", overwrite=True)

    present_extra = [str(v) for v in out["label"] if str(v) in extra_labels]
    logging.info("Saved %d systems to %s; included %d extra 400 pc systems.",
                 len(out), cache_path, len(present_extra))
    return out


def resolve_named_bright_stars(config: GaiaMapConfig, overwrite: bool = False) -> Table:
    """Resolve/load bright named stars and remove excluded labels such as Acrux."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.cache_dir / config.bright_star_cache_name
    if cache_path.exists() and not overwrite:
        logging.info("Loading cached bright stars: %s", cache_path)
        tab = Table.read(cache_path, format="ascii.ecsv")
        if 'label' in tab.colnames:
            tab = tab[np.array([str(lbl) not in EXCLUDED_BRIGHT_STAR_NAMES for lbl in tab['label']])]
        missing = _cache_missing_labels(tab, _expected_bright_star_labels(config))
        if missing:
            logging.info('Bright-star cache missing %d labels; rebuilding cache.', len(missing))
        else:
            return tab

    simbad = Simbad()
    rows = []
    seen_labels: set[str] = set()
    for star in BRIGHT_NAMED_STARS:
        if float(star["distance_pc"]) > float(config.radius_pc):
            continue
        label = str(star["label"])
        if label in seen_labels or label in EXCLUDED_BRIGHT_STAR_NAMES:
            continue
        seen_labels.add(label)
        pos = simbad_icrs_position(simbad, star["name"])
        if pos is None:
            continue
        rows.append({**star, "ra_deg": pos[0], "dec_deg": pos[1]})

    out = table_with_xyz(rows)
    out.write(cache_path, format="ascii.ecsv", overwrite=True)
    logging.info("Saved %d bright stars to %s", len(out), cache_path)
    return out


def resolve_nebulae(config: GaiaMapConfig, overwrite: bool = False) -> Table:
    """Resolve/load the final nebula/dust table; CrA is not included."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.cache_dir / config.nebula_cache_name

    if cache_path.exists() and not overwrite:
        logging.info("Loading cached nebulae: %s", cache_path)
        tab = Table.read(cache_path, format='ascii.ecsv')
    else:
        rows = []
        simbad = Simbad()
        for obj in NEBULAE:
            if obj['resolver'] == 'manual':
                ra_deg, dec_deg = obj['ra_deg'], obj['dec_deg']
            else:
                pos = simbad_icrs_position(simbad, obj['name'])
                if pos is None:
                    continue
                ra_deg, dec_deg = pos
            rows.append({**obj, 'ra_deg': ra_deg, 'dec_deg': dec_deg})
        tab = table_with_xyz(rows)
        tab.write(cache_path, format='ascii.ecsv', overwrite=True)

    if len(tab) == 0 or 'label' not in tab.colnames:
        return tab

    labels = np.asarray([str(v) for v in tab['label']])
    tab = tab[np.asarray([lbl in _ALLOWED_NEBULA_LABELS for lbl in labels], dtype=bool)]

    # Force cached tables to final authoritative values.
    by_label = {str(obj['label']): obj for obj in NEBULAE}
    for row in tab:
        lbl = str(row['label'])
        obj = by_label.get(lbl)
        if obj is None:
            continue
        for col in ['distance_pc', 'ra_deg', 'dec_deg', 'a_deg', 'b_deg', 'pa_deg', 'color']:
            if col in tab.colnames and col in obj:
                row[col] = obj[col]
    return table_with_xyz([dict(row) for row in tab])


def resolve_association_members(config: GaiaMapConfig, overwrite: bool = False) -> Table:
    cache_path = config.cache_dir / CLEAN_ASSOC_MEMBER_CACHE
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not overwrite:
        logging.info('Loading cached association members: %s', cache_path)
        return Table.read(cache_path, format='ascii.ecsv')

    simbad = Simbad()
    rows = []
    dist_map = {str(obj['label']): float(obj['distance_pc']) for obj in NEARBY_SYSTEMS}
    skip_names = {'TWA 30A'}
    for group_label, names in ASSOCIATION_MEMBER_SPECS.items():
        for name in names:
            if name in skip_names:
                continue
            pos = simbad_icrs_position(simbad, name)
            if pos is None:
                continue
            rows.append({
                'group_label': group_label,
                'member_name': name,
                'distance_pc': dist_map.get(group_label, np.nan),
                'ra_deg': pos[0],
                'dec_deg': pos[1],
            })

    if not rows:
        return Table(
            names=['group_label', 'member_name', 'distance_pc', 'ra_deg', 'dec_deg',
                   'l_deg', 'b_deg', 'X_pc', 'Y_pc', 'Z_pc'],
            dtype=['U32', 'U64', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'],
        )

    out = table_with_xyz(rows)
    out.write(cache_path, format='ascii.ecsv', overwrite=True)
    return out


def resolve_gaia_density_sample(config: GaiaMapConfig, overwrite: bool = False) -> Table:
    """
    Gaia density sample with a stable cache name and MemoryError-safe loading.
    """
    cache_path = config.cache_dir / CLEAN_GAIA_DENSITY_CACHE
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    max_cache_mb = 250.0

    if cache_path.exists() and not overwrite:
        try:
            size_mb = cache_path.stat().st_size / (1024.0 * 1024.0)
        except OSError:
            size_mb = np.inf
        if size_mb > max_cache_mb:
            logging.warning(
                'Gaia density cache %s is %.1f MB; skipping it to avoid MemoryError. '
                'Delete this file or rerun with overwrite=True to rebuild it.',
                cache_path, size_mb,
            )
            return empty_gaia_density_table()
        try:
            logging.info('Loading cached Gaia density sample: %s', cache_path)
            return Table.read(cache_path, format='ascii.ecsv')
        except MemoryError:
            logging.warning('MemoryError while reading %s; continuing without Gaia density.', cache_path)
            return empty_gaia_density_table()
        except Exception as exc:
            logging.warning('Could not read Gaia density cache %s (%s); continuing without Gaia density.', cache_path, exc)
            return empty_gaia_density_table()

    if GaiaArchive is None:
        logging.warning('astroquery.gaia unavailable; continuing without Gaia density.')
        return empty_gaia_density_table()

    rows = []
    try:
        if float(config.radius_pc) > 220.0:
            top_n = 260000
            gmag_limit = 15.5
        else:
            top_n = 120000
            gmag_limit = 14.0

        query = f"""
            SELECT TOP {top_n}
                source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp, random_index
            FROM gaiadr3.gaia_source
            WHERE parallax >= {1000.0 / float(config.radius_pc):.6f}
              AND parallax_over_error >= 5
              AND phot_g_mean_mag <= {gmag_limit}
              AND ruwe < 1.4
              AND bp_rp IS NOT NULL
            ORDER BY random_index
        """
        logging.info('Querying Gaia density sample; cache target: %s', cache_path)
        job = GaiaArchive.launch_job(query)
        res = job.get_results()
        logging.info('Gaia density query returned %d rows.', len(res))

        for row in res:
            plx = safe_float(row['parallax'])
            if not np.isfinite(plx) or plx <= 0:
                continue
            ra = safe_float(row['ra'])
            dec = safe_float(row['dec'])
            if not np.isfinite(ra) or not np.isfinite(dec):
                continue
            rows.append({
                'source_id': safe_int(row['source_id']),
                'distance_pc': 1000.0 / plx,
                'phot_g_mean_mag': safe_float(row['phot_g_mean_mag']),
                'bp_rp': safe_float(row['bp_rp']),
                'ra_deg': ra,
                'dec_deg': dec,
            })
    except Exception as exc:
        logging.warning('Gaia density query failed (%s); continuing without Gaia density.', exc)
        return empty_gaia_density_table()

    if not rows:
        return empty_gaia_density_table()

    out = table_with_xyz(rows)
    out.write(cache_path, format='ascii.ecsv', overwrite=True)
    return out


def _draw_sky_nebula_region(ax, row, *, alpha: float = 0.14, zorder: float = 4.0):
    """Fallback renderer for non-special nebulae using sky-projected ellipses/components."""
    label = str(row['label'])
    color = str(row['color'])
    if str(row.get('style', 'region')) == 'point' or label == 'Helix':
        ax.scatter(float(row['X_pc']), float(row['Y_pc']), marker='o', s=78,
                   facecolor=color, edgecolor='0.25', linewidth=1.0, alpha=0.95, zorder=zorder + 0.2)
        ax.scatter(float(row['X_pc']), float(row['Y_pc']), marker='o', s=180,
                   facecolor='none', edgecolor=color, linewidth=0.9, alpha=0.40, zorder=zorder)
        return

    components = NEBULA_COMPONENTS.get(label, [{
        'ra_deg': float(row['ra_deg']), 'dec_deg': float(row['dec_deg']),
        'a_deg': float(row['a_deg']), 'b_deg': float(row['b_deg']),
        'pa_deg': float(row.get('pa_deg', 0.0)),
    }])
    if label == 'California nebula':
        alpha = max(alpha, 0.55)
    for comp in components:
        xp, yp = sky_ellipse_xy(comp['ra_deg'], comp['dec_deg'], float(row['distance_pc']),
                                comp['a_deg'], comp['b_deg'], comp.get('pa_deg', 0.0))
        ax.fill(xp, yp, facecolor=color, edgecolor='none', alpha=alpha, zorder=zorder)
        ax.plot(np.r_[xp, xp[0]], np.r_[yp, yp[0]], color=color, lw=0.7,
                alpha=min(alpha * 1.30, 0.62), zorder=zorder + 0.05)


def _rotated_irregular_blob_xy(
    x0: float,
    y0: float,
    a_pc: float,
    b_pc: float,
    angle_deg: float,
    *,
    n: int = 220,
    seed: int = 0,
    roughness: float = 0.16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an irregular closed blob directly in the final X-Y map plane.

    Parameters
    ----------
    x0, y0 : float
        Blob centre in the plotted heliocentric X-Y coordinates [pc].
    a_pc, b_pc : float
        Semimajor and semiminor axes in plotted parsecs.
    angle_deg : float
        Rotation angle in the plotted X-Y plane.
    n : int
        Number of boundary points.
    seed : int
        Deterministic seed for the edge modulation.
    roughness : float
        Amplitude of the low-order boundary modulation.

    Notes
    -----
    This is intentionally a visual dust morphology layer, not a new physical
    distance estimate. It mimics the broad, patchy dark-cloud appearance in
    the reference image while preserving all original labels and coordinates.
    """
    rng = np.random.default_rng(int(seed))
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)

    # Smooth, deterministic non-elliptical edge.
    mod = np.ones_like(theta)
    for k, amp in ((2, 0.42), (3, 0.30), (5, 0.18), (7, 0.10)):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        mod += float(roughness) * amp * np.sin(k * theta + phase)
    mod = np.clip(mod, 0.66, 1.34)

    x = float(a_pc) * np.cos(theta) * mod
    y = float(b_pc) * np.sin(theta) * mod

    ang = np.deg2rad(float(angle_deg))
    ca, sa = np.cos(ang), np.sin(ang)
    xr = float(x0) + ca * x - sa * y
    yr = float(y0) + sa * x + ca * y
    return xr, yr

def _blob_polygon_xy(blob: dict) -> Polygon:
    xp, yp = _rotated_irregular_blob_xy(
        blob['x'], blob['y'],
        blob['a'], blob['b'],
        blob.get('angle', 0.0),
        seed=blob.get('seed', 0),
        roughness=blob.get('roughness', 0.16),
        n=blob.get('n', 240),
    )
    pts = np.column_stack([xp, yp])
    return Polygon(pts).buffer(0)

def _deduplicate_objects_by_label(seq: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for row in seq:
        key = str(row.get('label', '')).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out

BRIGHT_STARS_400PC_EXTRA = _deduplicate_objects_by_label(BRIGHT_STARS_400PC_EXTRA)
SYSTEMS_400PC_EXTRA = _deduplicate_objects_by_label(SYSTEMS_400PC_EXTRA)



def _cache_missing_labels(tab: Table | None, expected_labels: list[str], label_col: str = 'label') -> list[str]:
    if tab is None:
        return list(expected_labels)
    try:
        present = {str(v) for v in tab[label_col]}
    except Exception:
        return list(expected_labels)
    return [lbl for lbl in expected_labels if str(lbl) not in present]


def _expected_system_labels(config: GaiaMapConfig) -> list[str]:
    labels = []
    for obj in list(NEARBY_SYSTEMS) + list(SYSTEMS_400PC_EXTRA):
        try:
            if float(obj.get('distance_pc', np.inf)) <= float(config.radius_pc) + 60.0:
                labels.append(str(obj.get('label', '')))
        except Exception:
            continue
    return [lbl for lbl in labels if lbl]


def _expected_bright_star_labels(config: GaiaMapConfig) -> list[str]:
    labels = []
    for obj in list(BRIGHT_NAMED_STARS):
        try:
            if float(obj.get('distance_pc', np.inf)) <= float(config.radius_pc):
                labels.append(str(obj.get('label', '')))
        except Exception:
            continue
    return [lbl for lbl in labels if lbl]


def _draw_merged_dust_region_xy(
    ax,
    blobs: list[dict],
    *,
    base_color: str,
    alpha_scale: float = 1.0,
    zorder: float = 4.0,
) -> None:
    polys = []
    for blob in blobs:
        poly = _blob_polygon_xy(blob)
        if not poly.is_empty:
            polys.append(poly)

    if not polys:
        return

    merged = unary_union(polys).buffer(2.0).buffer(-2.0).buffer(0)

    def _draw_single_polygon(poly, face_alpha=0.13, edge_alpha=0.22):
        x, y = poly.exterior.xy
        ax.fill(
            x, y,
            facecolor=base_color,
            edgecolor='none',
            alpha=face_alpha * alpha_scale,
            zorder=zorder,
        )
        ax.plot(
            x, y,
            color=base_color,
            lw=0.8,
            alpha=edge_alpha * alpha_scale,
            zorder=zorder + 0.05,
        )

        # Si quieres respetar huecos internos:
        for interior in poly.interiors:
            xi, yi = interior.xy
            ax.fill(xi, yi, color=ax.get_facecolor(), zorder=zorder + 0.01)

    if isinstance(merged, Polygon):
        _draw_single_polygon(merged)
    elif isinstance(merged, MultiPolygon):
        for poly in merged.geoms:
            _draw_single_polygon(poly)

def _draw_dust_cores_xy(
    ax,
    blobs: list[dict],
    *,
    zorder: float = 4.3,
) -> None:
    for i, blob in enumerate(blobs):
        if 'color' not in blob:
            continue

        xp, yp = _rotated_irregular_blob_xy(
            blob['x'], blob['y'],
            blob['a'], blob['b'],
            blob.get('angle', 0.0),
            seed=blob.get('seed', 100 + i),
            roughness=blob.get('roughness', 0.18),
            n=blob.get('n', 220),
        )

        ax.fill(
            xp, yp,
            facecolor=blob['color'],
            edgecolor='none',
            alpha=blob.get('alpha', 0.18),
            zorder=zorder,
        )

def draw_nebula_region(ax, row, *, alpha: float = 0.14, zorder: float = 4.0):
    """
    Final renderer for nebulae/dust complexes.

    Coalsack, Ophiuchus dark clouds and Aquila south rift are drawn as merged
    irregular X-Y dust surfaces with darker internal cores. All other nebulae
    use the sky-projected ellipse/component fallback.
    """
    label = str(row['label'])

    if label in _DUST_XY_BLOBS:
        blobs = _DUST_XY_BLOBS[label]
        base_blobs = [b for b in blobs if 'color' not in b]
        core_blobs = [b for b in blobs if 'color' in b]
        _draw_merged_dust_region_xy(
            ax,
            base_blobs,
            base_color=str(row['color']),
            alpha_scale=float(alpha) / 0.14,
            zorder=zorder,
        )
        _draw_dust_cores_xy(ax, core_blobs, zorder=zorder + 0.15)
        return

    _draw_sky_nebula_region(ax, row, alpha=alpha, zorder=zorder)

def system_legend_display_name(row) -> str:
    label = str(row['label'])
    name = str(row['name'])

    # ASCC clusters: marker/code short, legend explicit.
    if label in {'101', '99', '19', '127', '123'}:
        return f'ASCC {label}'

    # Show NGC objects by number only, as requested.
    if label.startswith('NGC '):
        return label.replace('NGC ', '').strip()

    # Optional: make other shortened labels clearer in the legend.
    if label == 'St1':
        return 'Stock 1'
    if label in {'U1', 'U9', 'U31', 'U32'}:
        return f'UBC {label[1:]}'
    if label == 'RSG1':
        return 'RSG 1'
    if label == 'RSG5':
        return 'RSG 5'

    return label


def stellar_radius_estimate_rsun(label: str) -> float:
    """Approximate stellar radius from curated spectral-type information or Teff class."""
    key = str(label)
    if key in STAR_RADIUS_ESTIMATE_RSUN:
        return float(STAR_RADIUS_ESTIMATE_RSUN[key])
    teff = teff_for_star(key)
    # Fallback dwarf/giant-free scale by effective-temperature class.  This is
    # only used for visual marker sizes when no curated radius is available.
    if teff >= 20000:
        return 8.0
    if teff >= 10000:
        return 3.0
    if teff >= 7500:
        return 2.0
    if teff >= 6000:
        return 1.2
    if teff >= 5200:
        return 1.0
    if teff >= 4000:
        return 0.8
    return 0.45


def star_marker_size_from_radius(label: str, *, min_size: float = 16.0, max_size: float = 92.0) -> float:
    """Marker area scaled monotonically with estimated stellar radius."""
    radius = np.clip(stellar_radius_estimate_rsun(label), 0.15, 900.0)
    log_r = np.log10(radius)
    log_min = np.log10(0.15)
    log_max = np.log10(900.0)
    t = (log_r - log_min) / (log_max - log_min)
    return float(min_size + np.clip(t, 0.0, 1.0) * (max_size - min_size))


def plot_solar_neighbourhood_xy_auto(
    systems: Table,
    bright_stars: Table,
    config: GaiaMapConfig,
    nebulae: Table | None = None,
    association_members: Table | None = None,
    gaia_density: Table | None = None,
    output_stem: str = 'solar_neighbourhood_200pc_xy_manual_auto_labels_fixed_v2',
) -> None:
    systems = _repair_mathtext_table(systems)
    bright_stars = _repair_mathtext_table(bright_stars)
    nebulae = _repair_mathtext_table(nebulae)
    association_members = _repair_mathtext_table(association_members)

    config.figure_dir.mkdir(parents=True, exist_ok=True)
    r = float(config.radius_pc)

    if r > 220.0:
        star_label_fontsize = 5.4
        nebula_label_fontsize = 7.2
        group_label_fontsize = 7.5
        legend_fontsize = 8.1
        legend_line_step = 0.0215
        figure_size = (16.8, 10.6)
        dpi_out = 240
    else:
        star_label_fontsize = 6.7
        nebula_label_fontsize = 8.1
        group_label_fontsize = 8.7
        legend_fontsize = 10.0
        legend_line_step = 0.0265
        figure_size = (16.4, 10.2)
        dpi_out = 260

    systems = systems[finite_xyz_mask(systems) & within_xy_radius_mask(systems, r + 40.0)]
    systems = systems[np.array([str(lbl) not in {'Argus', 'Columba'} for lbl in systems['label']])]

    if r > 220.0:
        # Visual cleanup for the 400 pc map: gently reposition a few manually-curated
        # clusters / groups so crowded southern clusters and newly added outer objects
        # remain readable without changing the rest of the map.
        for _i, _lbl in enumerate(np.asarray(systems['label'], dtype=str)):
            if _lbl in SYSTEM_POSITION_OVERRIDES_400PC:
                _ov = SYSTEM_POSITION_OVERRIDES_400PC[_lbl]
                systems['X_pc'][_i] = float(_ov['x'])
                systems['Y_pc'][_i] = float(_ov['y'])
            if _lbl in SYSTEM_POSITION_SHIFTS_400PC:
                _sv = SYSTEM_POSITION_SHIFTS_400PC[_lbl]
                systems['X_pc'][_i] = float(systems['X_pc'][_i]) + float(_sv.get('dx', 0.0))
                systems['Y_pc'][_i] = float(systems['Y_pc'][_i]) + float(_sv.get('dy', 0.0))

    all_bright_stars = bright_stars[finite_xyz_mask(bright_stars) & within_xy_radius_mask(bright_stars, r)]
    if r > 220.0 and 'STAR_POSITION_OVERRIDES_400PC' in globals():
        for _i, _lbl in enumerate(np.asarray(all_bright_stars['label'], dtype=str)):
            if _lbl in STAR_POSITION_OVERRIDES_400PC:
                _ov = STAR_POSITION_OVERRIDES_400PC[_lbl]
                all_bright_stars['X_pc'][_i] = float(_ov['x'])
                all_bright_stars['Y_pc'][_i] = float(_ov['y'])
                if 'z' in _ov:
                    all_bright_stars['Z_pc'][_i] = float(_ov['z'])
    labelled_bright_stars = select_distributed_labelled_stars(all_bright_stars, max_outer_labels=(52 if r > 220.0 else 24), min_sep_pc=(24.0 if r > 220.0 else 28.0))

    if nebulae is not None:
        nebulae = nebulae[finite_xyz_mask(nebulae) & within_xy_radius_mask(nebulae, r + 40.0)]
    if association_members is not None and len(association_members) > 0:
        association_members = association_members[finite_xyz_mask(association_members) & within_xy_radius_mask(association_members, r + 40.0)]
    if gaia_density is not None and len(gaia_density) > 0:
        gaia_density = gaia_density[finite_xyz_mask(gaia_density) & within_xy_radius_mask(gaia_density, r)]

    fig = plt.figure(figsize=figure_size, constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[5.42, 2.62], wspace=0.085)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis('off')
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.16, top=0.92)

    language = config.language if config.language in UI_TEXT else 'en'

    density_sm = None
    if getattr(config, 'show_gaia_density', True):
        density_sm = draw_gaia_density_map(ax, gaia_density, radius_pc=r, zorder=0.2)

    add_xy_distance_circles(ax, radii=(50, 100, 200, 400) if r > 220 else (50, 100, 200))
    ax.axhline(0, color='0.72', lw=0.95, alpha=0.85, zorder=0.5)
    ax.axvline(0, color='0.72', lw=0.95, alpha=0.85, zorder=0.5)
    ax.grid(False)
    pad = 16.0
    ax.set_xlim(-r - pad, r + pad)
    ax.set_ylim(-r - pad, r + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$X$ [pc]  $(l=0^\circ)$', fontsize=13)
    ax.set_ylabel(r'$Y$ [pc]  $(l=90^\circ)$', fontsize=13)
    ax.set_title(tr(language, 'title', radius=r), fontsize=18, pad=12)

    dir1 = ax.text(r - 4, -7, r'$l=0^\circ$', ha='right', va='top', fontsize=10.5, color='0.35')
    dir2 = ax.text(4, r - 5, r'$l=90^\circ$', ha='left', va='top', fontsize=10.5, color='0.35')
    for txt in [dir1, dir2]:
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground='white', alpha=0.95)])

    occupied = []

    # Dust / cloud complexes first.
    nebula_items = []
    physically_drawn_nebulae = set()

    # H5-only preview: draw the STILISM/Lallement projection even when the
    # labelled nebula/cloud layer is disabled.
    if not getattr(config, 'show_nebulae', True):
        logging.info('Drawing H5-only dust layer...')
        draw_physical_cloud_layer(ax, None, config, zorder=4.0)

    if nebulae is not None and getattr(config, 'show_nebulae', True):
        logging.info('Drawing nebula/catalog layer...')
        t_neb = time.perf_counter()
        physically_drawn_nebulae = draw_physical_cloud_layer(ax, nebulae, config, zorder=4.0)
        logging.info('Nebula/catalog layer finished in %.2f s', time.perf_counter() - t_neb)
        for row in nebulae:
            label = str(row['label'])
            if label not in physically_drawn_nebulae:
                draw_nebula_region(ax, row, alpha=getattr(config, 'nebula_alpha', 0.16), zorder=4.0)
            if label in {'ORI OB1', 'VEL OB2'}:
                continue
            text = label if label != 'Helix' else f'Helix ({signed_z(float(row["Z_pc"]))} pc)'
            offsets = NEBULA_LABEL_OFFSETS.get(label, (14, 12))
            nebula_items.append({'x': float(row['X_pc']), 'y': float(row['Y_pc']), 'text': text, 'label_key': label, 'priority': 50, 'force': True, 'alpha': 0.96, 'zorder': 55, 'offsets': [offsets] + candidate_offsets_for_position(float(row['X_pc']), float(row['Y_pc'])), 'color': '0.18', 'leader_color': '0.38', 'leader_alpha': 0.55, 'leader_lw': 0.55, 'leader_shrinkB': 2, 'draw_leaders': label not in NEBULA_LABELS_NO_LEADERS_400PC})

    # Add ad hoc Gum nebula label for 400 pc map, matching the small nebula-label
    # style used for labels such as Orion nebula.
    if r > 220.0:
        _gum_txt = ax.text(
            -160, -310, 'Gum nebula',
            fontsize=7.0, color='0.18', alpha=0.96, zorder=55,
            ha='center', va='center',
        )
        _gum_txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white', alpha=0.95)])

    ax.scatter(0, 0, marker='*', s=300, color='gold', edgecolor='black', linewidth=0.95, zorder=200)
    occupied.extend(add_occupied_marker_boxes(ax, [0], [0], size_px=20))

    codes = make_unique_system_codes(systems)
    system_x, system_y = [], []
    group_items = []

    for row in systems:
        x, y = float(row['X_pc']), float(row['Y_pc'])
        age = float(row['age_myr'])
        label = str(row['label'])
        kind = str(row['kind'])
        color = age_to_color(age)

        if kind in {'association', 'sco_cen'}:
            members = None
            if association_members is not None and len(association_members) > 0:
                members = association_members[np.array([str(gl) == label for gl in association_members['group_label']])]
            if members is not None and len(members) >= 3:
                draw_member_region(ax, members, facecolor=color, edgecolor='0.35',
                                   alpha=region_alpha_for_kind(kind, getattr(config, 'region_alpha', 0.16)),
                                   zorder=8.0, smooth=True, show_member_points=False)
                cx = np.nanmean(np.asarray(members['X_pc'], dtype=float))
                cy = np.nanmean(np.asarray(members['Y_pc'], dtype=float))
            else:
                cx, cy = x, y
                if getattr(config, 'show_system_regions', True):
                    draw_real_projected_region(ax, row, facecolor=color, edgecolor='0.35',
                                               alpha=region_alpha_for_kind(kind, getattr(config, 'region_alpha', 0.16)),
                                               zorder=8.0, use_hull=(kind == 'association'))

            offsets = SYSTEM_LABEL_OFFSETS.get(label, (14, 14))
            group_items.append({'x': float(cx), 'y': float(cy), 'text': codes.get(label, label), 'label_key': label, 'priority': 40, 'force': True, 'alpha': 0.96, 'zorder': 82, 'offsets': offsets + candidate_offsets_for_position(float(cx), float(cy)), 'color': '0.12'})
            continue

        # Open clusters
        if getattr(config, 'show_system_regions', True):
            draw_real_projected_region(ax, row, facecolor=color, edgecolor='0.35',
                                       alpha=region_alpha_for_kind(kind, getattr(config, 'region_alpha', 0.16)),
                                       zorder=8.0, use_hull=False)

        system_x.append(x)
        system_y.append(y)
        ax.scatter(x, y, marker=kind_to_marker(kind), s=system_marker_size(label, kind),
                   color=color, edgecolor='black', linewidth=1.05, zorder=70)
        t = ax.text(x, y, codes[label], ha='center', va='center', fontsize=system_code_fontsize(codes[label]),
                    weight='bold', color='black', zorder=80)
        t.set_path_effects([pe.withStroke(linewidth=1.3, foreground='white', alpha=0.65)])

    occupied.extend(add_occupied_marker_boxes(ax, system_x, system_y, size_px=34))
    if nebula_items:
        occupied = annotate_items(ax, nebula_items, fontsize=nebula_label_fontsize, occupied=occupied, max_labels=None, draw_leaders=bool(getattr(config, 'draw_label_leaders', True)))

    # Reference OB-association labels copied by eye from the 400 pc guide image.
    if r > 220.0:
        for _lbl in REFERENCE_OB_LABELS_400PC:
            _txt = ax.text(
                float(_lbl['x']), float(_lbl['y']), str(_lbl['text']),
                ha=_lbl.get('ha', 'center'), va=_lbl.get('va', 'center'),
                fontsize=11.0, weight='bold', color='#b28a00', zorder=86,
            )
            _txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white', alpha=0.92)])
    if group_items:
        occupied = annotate_items(ax, group_items, fontsize=group_label_fontsize, occupied=occupied, max_labels=None, draw_leaders=False)

    # Bright stars.
    bt = labelled_bright_stars[np.argsort(np.asarray(labelled_bright_stars['vmag'], dtype=float))]
    star_x, star_y = [], []
    label_items = []
    for row in bt:
        x, y, z = float(row['X_pc']), float(row['Y_pc']), float(row['Z_pc'])
        vmag = float(row['vmag'])
        label = str(row['label'])
        if label == 'Acrux':
            continue

        size = star_marker_size_from_radius(label, min_size=16.0, max_size=92.0)
        star_x.append(x)
        star_y.append(y)

        manual_offsets = MANUAL_STAR_OFFSETS.get(label)
        candidate_offsets = candidate_offsets_for_position(x, y)
        if manual_offsets is not None:
            candidate_offsets = manual_offsets + candidate_offsets

        label_items.append({
            'x': x, 'y': y,
            'text': short_star_label(label, z),
            'label_key': label,
            'priority': vmag,
            'force': True,
            'alpha': 0.95,
            'zorder': 110,
            'marker_zorder': 105,
            'leader_zorder': 94,
            'marker_size': size,
            'star_color': get_color_from_Teff(teff_for_star(label)),
            'offsets': candidate_offsets,
            'draw_leaders': bool(getattr(config, 'draw_label_leaders', True)),
            'fast_leaders': bool(getattr(config, 'fast_label_leaders', True)),
            'leader_alpha': 0.42,
            'leader_lw': 0.45,
        })

    occupied.extend(add_occupied_marker_boxes(ax, star_x, star_y, size_px=15))
    logging.info('Annotating bright-star labels...')
    t_lab = time.perf_counter()
    annotate_items(ax, label_items, fontsize=star_label_fontsize, occupied=occupied, max_labels=None, draw_leaders=bool(getattr(config, 'draw_label_leaders', True)))
    logging.info('Bright-star labels finished in %.2f s', time.perf_counter() - t_lab)

    for item in label_items:
        if item.get('_placed', False):
            ax.scatter(float(item['x']), float(item['y']), s=float(item['marker_size']),
                       facecolor=item.get('star_color', '0.22'), edgecolor='0.10',
                       linewidth=0.8, alpha=0.96, zorder=105)

    # Compact two-block legend: code / name / Z only.
    ax_leg.text(0.02, 0.985, tr(language, 'systems'), transform=ax_leg.transAxes,
                fontsize=15.0, weight='bold', ha='left', va='top')

    kind_order = {'open_cluster': 0, 'embedded_cluster': 0, 'association': 1, 'sco_cen': 2}
    legend_rows = sorted(list(systems), key=lambda rr: (kind_order.get(str(rr['kind']), 9), str(rr['label'])))
    mid = int(np.ceil(len(legend_rows) / 2.0))
    blocks = [legend_rows[:mid], legend_rows[mid:]]
    colspecs = [
        (0.02, 0.12, 0.47),
        (0.53, 0.63, 0.98),
    ]
    header_y = 0.953
    for (c0, c1, c2), rows_block in zip(colspecs, blocks):
        for xcol, txt, align in [
            (c0, tr(language, 'code'), 'left'),
            (c1, tr(language, 'name'), 'left'),
            (c2, 'Z [pc]', 'right'),
        ]:
            ax_leg.text(xcol, header_y, txt, transform=ax_leg.transAxes,
                        fontsize=10.6, family='monospace', color='0.30', ha=align, va='top')
        yb = 0.925
        for row in rows_block:
            label = str(row['label'])
            z_text = signed_z(float(row['Z_pc']))
            ax_leg.text(c0, yb, codes[label], transform=ax_leg.transAxes,
                        fontsize=legend_fontsize, family='monospace', ha='left', va='top')
            display_name = system_legend_display_name(row)

            ax_leg.text(c1, yb, display_name, transform=ax_leg.transAxes,
                        fontsize=legend_fontsize, family='monospace', ha='left', va='top')
            ax_leg.text(c2, yb, z_text, transform=ax_leg.transAxes,
                        fontsize=legend_fontsize, family='monospace', ha='right', va='top')
            yb -= legend_line_step

    y = 0.925 - legend_line_step * max(len(blocks[0]), len(blocks[1])) - 0.010
    ax_leg.text(0.06, y, tr(language, 'age_colour'), transform=ax_leg.transAxes,
                fontsize=15.0, weight='bold', ha='left', va='top')
    y -= 0.060

    # Age colour: three columns, no overlap.
    age_items = [
        ('#4cc9f0', tr(language, 'age_lt100')),
        ('#f9c74f', tr(language, 'age_100_600')),
        ('#f3722c', tr(language, 'age_ge600')),
    ]
    xcols = [0.05, 0.31, 0.57]
    for (col, txt), xx in zip(age_items, xcols):
        ax_leg.scatter(xx, y, transform=ax_leg.transAxes, marker='s',
                       s=86, facecolor=col, edgecolor='black', clip_on=False)
        ax_leg.text(xx + 0.050, y, txt, transform=ax_leg.transAxes,
                    fontsize=9.0, va='center', ha='left')

    y -= 0.085
    ax_leg.text(0.06, y, tr(language, 'stellar_colour'), transform=ax_leg.transAxes,
                fontsize=15.0, weight='bold', ha='left', va='top')
    add_stellar_temperature_colorbar(ax_leg, x0=0.06, y0=y - 0.090, width=0.74, height=0.030,
                                     language=language)

    if density_sm is not None:
        y -= 0.150
        ax_leg.text(0.06, y, 'Gaia density', transform=ax_leg.transAxes,
                    fontsize=15.0, weight='bold', ha='left', va='top')
        add_gaia_density_colorbar(ax_leg, x0=0.06, y0=y - 0.090, width=0.74, height=0.030,
                                  language=language)

    note = tr(language, 'note')
    fig.text(0.49, 0.070, note, ha='center', va='center', fontsize=8.8, color='0.28')

    png_path = config.figure_dir / f'{output_stem}.png'
    pdf_path = config.figure_dir / f'{output_stem}.pdf'
    dpi_save = int(getattr(config, 'preview_png_dpi', dpi_out) or dpi_out)
    logging.info('Saving PNG at dpi=%d...', dpi_save)
    t_save = time.perf_counter()
    fig.savefig(png_path, dpi=dpi_save)
    logging.info('Saved %s in %.2f s', png_path, time.perf_counter() - t_save)
    if bool(getattr(config, 'save_pdf', False)):
        logging.info('Saving PDF...')
        t_pdf = time.perf_counter()
        fig.savefig(pdf_path)
        logging.info('Saved %s in %.2f s', pdf_path, time.perf_counter() - t_pdf)
    plt.close(fig)

def main() -> None:
    setup_logging()
    config = GaiaMapConfig(
        radius_pc=400.0,
        language='es',
        show_system_regions=True,
        show_nebulae=True,
        show_gaia_density=True,
        show_association_members=True,
        region_alpha=0.16,
        nebula_alpha=0.17,
        system_cache_name=CLEAN_SYSTEM_CACHE,
        bright_star_cache_name=CLEAN_BRIGHT_STAR_CACHE,
        nebula_cache_name=CLEAN_NEBULA_CACHE,
        data_yaml_path=Path('solar_map_400pc_data.yaml'),
        cloud_catalog_html=Path('handbook_distances.html'),
        dust_h5_path=Path('map3D_GAIAdr2_feb2019.h5'),
        cloud_render_mode='hybrid',  # use 'hybrid' or 'catalog' to draw molecular-cloud bubbles too,
        # then pass a green facecolor to draw_catalog_cloud_group(...) if desired,
        show_physical_dust_projection=True,
        dust_projection_mode='max',
        dust_projection_sphere_only=True,
        preview_png_dpi=180,
        save_pdf=False,
        draw_label_leaders=True,
        fast_label_leaders=True,
        dust_projection_alpha=0.30,
        dust_contour_alpha=0.58,
        dust_contour_lw=1.05,
        dust_percentile_lo=52.0,
        dust_percentile_hi=98.7,
        dust_lowest_contour_level=0.16,
    )

    # FAST_PREVIEW: use existing caches when available. Set overwrite=True only
    # when you want to refresh SIMBAD/Gaia-derived positions.
    t0 = time.perf_counter()
    systems = resolve_systems(config=config, overwrite=False)
    logging.info('Systems ready in %.2f s', time.perf_counter() - t0)

    t0 = time.perf_counter()
    bright_stars = resolve_named_bright_stars(config=config, overwrite=False)
    logging.info('Bright stars ready in %.2f s', time.perf_counter() - t0)

    # Only load the nebula table if the labelled/catalog cloud layer is enabled.
    # H5-only projection does not need it.
    need_nebula_table = bool(config.show_nebulae and str(config.cloud_render_mode).lower() in {'manual', 'catalog', 'hybrid'})
    t0 = time.perf_counter()
    nebulae = resolve_nebulae(config=config, overwrite=False) if need_nebula_table else None
    logging.info('Nebula table ready/skipped in %.2f s', time.perf_counter() - t0)

    association_members = resolve_association_members(config=config, overwrite=False) if config.show_association_members else None
    gaia_density = resolve_gaia_density_sample(config=config, overwrite=False) if config.show_gaia_density else empty_gaia_density_table()

    t0 = time.perf_counter()
    plot_solar_neighbourhood_xy_auto(
        systems=systems,
        bright_stars=bright_stars,
        nebulae=nebulae,
        association_members=association_members,
        gaia_density=gaia_density,
        config=config,
        output_stem='solar_map_400pc_xy',
    )
    logging.info('Plot/render/save finished in %.2f s', time.perf_counter() - t0)

if __name__ == '__main__':
    main()
