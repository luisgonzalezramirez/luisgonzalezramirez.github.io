
# Luis González Ramírez personal website — v1 content update

This folder contains the updated landing page and asset structure.

## Main changes

- Removed the institutional logo strip (CAB / INTA / CSIC / etc.) from the landing page.
- Added a full astrophotography gallery with placeholder images for:
  M8, M13, M16, M42, M1, Moon, Milky Way from Tendilla, M81–M82, M51, M51 field galaxies and M51 zoom / IC 4278.
- Added bilingual CV selector with embedded PDF viewer.
- Added publication cards, A&A/ADS links, local PDF slots and QR codes.
- Added GitHub, ORCID, Instagram and ADS profile links.

## Files to replace with real content

Keep the same filenames or edit `assets/js/gallery.js`.

### PDFs
Add these files:

```text
assets/pdf/CV_Luis_Gonzalez_Ramirez_EN.pdf
assets/pdf/CV_Luis_Gonzalez_Ramirez_ES.pdf
assets/pdf/chronos_aanda_2026.pdf
assets/pdf/hsa_proceedings_2025.pdf
```

## Run locally

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Important note about the map

This package includes the uploaded data files and `map.html`. In this turn, the full `assets/js/solar-map.js` renderer was not included among the sandbox files, so I left a clear placeholder file. Replace it with the real `solar-map.js` from the current map prototype to restore the interactive Three.js map.
