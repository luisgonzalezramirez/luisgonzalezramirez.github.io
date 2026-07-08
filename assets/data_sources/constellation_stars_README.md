# Constellation-star layer

The three-brightest-stars-per-constellation layer is generated from HYG rather than hand-written.

Run:

```bash
python scripts/generate_constellation_stars_from_hyg.py --download --radius-pc 300 --top-n 3
```

or point it to a downloaded HYG CSV file:

```bash
python scripts/generate_constellation_stars_from_hyg.py --hyg-csv path/to/hygdata_v41.csv
```

The generated stars are saved in `assets/data/constellation_stars.json` and bundled into `assets/js/solar-data.js`.
