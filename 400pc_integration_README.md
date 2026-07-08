# 400 pc web-map data bridge

The web map does not read `map3D_GAIAdr2_feb2019.h5` or the Plotly handbook HTML directly in the browser. Those are scientific source files and must be preprocessed into JSON.

Files used by `scripts/export_400pc_web_json.py`:

- `solar_map_400pc_data.yaml`: systems, bright-star lists, nebula definitions, manual dust blobs, label/style metadata.
- `handbook_distances.html`: GalaxyMap/handbook molecular-cloud bubble catalogue. Parsed into 3D translucent cloud bubbles in `assets/data/regions.json`.
- `map3D_GAIAdr2_feb2019.h5`: optional Lallement/STILISM dust cube. If present in the project root, the exporter writes a decimated dust point layer to `assets/data/dust_voxels.json`.

Run:

```bash
python -m pip install -r requirements-data.txt
python scripts/export_400pc_web_json.py
```

For the H5 layer, place `map3D_GAIAdr2_feb2019.h5` in the project root before running the exporter.
