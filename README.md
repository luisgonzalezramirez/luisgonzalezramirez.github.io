# Data source seeds

These CSV files are the human-readable input tables for `scripts/generate_solar_json.py`.

## v0.15 policy

- Bright field stars: resolved by object name with SIMBAD when possible, with seed distances as fallback.
- Open-cluster centres: resolved by SIMBAD where useful, but the seed distance is normally preferred/required because cluster entries do not always have a meaningful parallax.
- Named cluster members: resolved by SIMBAD from `cluster_members_seed.csv`.
- If a named cluster member cannot be resolved, it is **skipped by default**. This avoids drawing fake named stars in the web map.
- Use `--allow-visual-fallback` only for visual demos; it writes deterministic fake offsets and marks them as `visual_fallback`.

For a scientifically stronger next step, replace or supplement `cluster_members_seed.csv` with Gaia DR3 membership tables rather than using only named bright members.


## Current named-member coverage

The seed table now includes bright named members/candidates for:

- Pleiades
- Hyades
- Praesepe
- α Per
- IC 2391
- IC 2602
- NGC 2451A

Pleiades has curated `ra_deg`, `dec_deg`, `distance_pc` values so it can be rendered immediately.
The other clusters are intended to be resolved by SIMBAD when `scripts/generate_solar_json.py` is run with internet access.
Unresolved entries remain listed as pending seeds and are not plotted as named stars.


## v1.2 bilingual update

- Added functional English/Spanish language switching across the main page, Chronos presentation and 3D map pages.
- Replaced language labels with flag buttons.
- Added Spanish slide deck support through `assets/img/talk_es/diapo_1.PNG` ... `diapo_12.PNG` and English slide deck support through `assets/img/talk_en/slide_01.PNG` ... `slide_12.PNG`.
- Reordered the 3D map navigation so Home appears first.
- Kept scientific labels and object names in the map in English while translating interface text and descriptions.
