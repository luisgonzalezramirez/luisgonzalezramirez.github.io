# Fixes applied in v8

## Index page
- Removed the circular mark to the left of the name in the navigation bar.
- Kept scroll reveal only on `index.html`, with earlier fade-in/fade-out so sections do not remain hidden until centred.
- Made CV PDF rendering lazy to avoid the brief white/blank flash when jumping to the CV from the top menu.
- Added keyboard navigation for the CV preview with left/right arrow keys when the CV section is visible.
- Replaced the social icons with the user-supplied PNG assets and removed the white icon tile background.
- Corrected the M51 field text from IC 4277 to IC 4263.

## Chronos presentation
- Removed the PDF/PowerPoint embedded viewer.
- Rebuilt `talk_chronos.html` as an in-page PNG slide deck using `assets/img/talk/slide_01.png` ... `slide_12.png`.
- Added left/right keyboard navigation for the presentation.
- Removed the visible filename from the viewer.
- Removed the white PDF border/background.
- Added a Home link in the top navigation.

## 3D map
- Kept the working Three.js map baseline.
- Flipped the Galactic Y rendering sign so the XY projection matches the intended convention: X to the Galactic centre and Y toward l = 90 degrees.
- Kept only XY-plane distance rings.
- Enlarged implicit cloud-surface search bounds to reduce clipped cloud surfaces.
- Added map references for Lallement et al. (2019) dust and Zucker et al. (2020) molecular-cloud distances.

## v9 map-only fixes

- Renamed the map layer label from `Dust / molecular clouds` to `Zucker molecular clouds`.
- Disabled explicit wireframe overlays on molecular-cloud/dust meshes so the surfaces render as continuous translucent bodies.
- Changed implicit cloud materials to uniform translucent surfaces, reducing visible mesh-triangle faceting.
- Reworked the implicit-cloud ray solver to avoid collapsed/cut surfaces when the density centre is outside the chosen isosurface. The new solver scans for entry/exit along each ray and falls back to a smooth radial envelope, producing closed surfaces instead of clipped sheets.
- Increased cloud-surface sampling/detail and allowed larger search radii only for the surface construction, without changing catalogue positions or nominal cloud sizes.

- Replaced blob-only cloud rendering with precomputed marching-cubes `cloud_meshes` in `assets/data/regions.json` for soft clouds, preventing large molecular/dust regions from being cut off.

- Added comet C/2025 A6 Lemmon as the second astrophotography card, expecting `assets/img/astro/a6_lemmon.jpg`.
- Optimised the 3D atlas without changing its content: indexed cloud mesh geometry, capped device-pixel ratio, disabled antialiasing, and throttled expensive label/constant-size updates.

- Added an Outreach section for Astroafición with the supplied activity photo and company logo.
- The outreach photo is rendered as a static content image, not as an astrophotography gallery item/modal image.
- Added an Outreach link to the main navigation.

- Added a static telescope presentation image to the Astrophotography section (`assets/img/astro/tel.jpg`).
- Added multiple static Outreach/Astroafición presentation images using the uploaded filenames, without gallery modal/open-new-tab behaviour.

- Restored the 3D map assets from v10 and applied only lightweight performance changes: lower pixel ratio, no antialias, indexed cloud meshes, throttled label updates.
- Reworked Outreach layout: small square Astroafición image on the left, logo/text/button on the right, and three square photos below.
- Kept Astrophotography and Outreach section images static/non-clickable.

- Restored the map data and cloud surfaces exactly from WEB_fixed_v10, replacing later experimental cloud-surface versions.
- Applied only safe performance optimisations to the v10 map renderer: lower pixel ratio, no antialias, indexed cloud meshes, and lighter label overlay updates.

- v16: Kept the v10 cloud mesh data intact and applied only a local Z-axis thickness factor in `renderCloudMesh`, preserving XY positions/orientation while making cloud surfaces less flattened.

- Replaced Outreach main Astroafición image with a cache-safe filename `assets/img/outreach/astro_main_updated.jpg` so browsers do not keep showing the previous `astro.JPG`.
- Map files left unchanged in this version.

- Map forcibly restored from WEB_fixed_v10.zip: `map.html`, `assets/data/`, `assets/js/solar-data.js`, and `assets/js/solar-map.js`.

- Rebuilt Zucker/dust region `cloud_meshes` from the JSON blobs with a real Python 3D density field and marching-cubes surfaces.
- XY centres/scales are preserved; Z radii are inflated locally before meshing so the clouds are not paper-thin.
- `solar-data.js` regenerated from JSON. `renderCloudMesh` now uses indexed BufferGeometry for speed.
- No Lallement H5 file was present in the package/session, so the true STILISM H5 is not reprocessed here; the included script is ready for the same mesh-export path once the H5 is supplied.

- Added `scripts/export_stars_only.py` to refresh only bright stars while preserving cloud/map surfaces.
- Added astrophotography thumbnails and changed gallery cards to load thumbnails while keeping full images for the modal.
- Replaced continuous scroll animation with one-shot IntersectionObserver reveal to reduce stutter.
- Added CSS containment/content-visibility for gallery cards.

- Restored smooth fade-in/fade-out reveal animations for gallery cards and sections using IntersectionObserver, not scroll listeners.
- Kept thumbnail loading/async decoding for the astrophotography gallery to avoid scroll jank.
- Removed the v20 animation suppression/content-visibility block that caused abrupt appearance/disappearance.

- Rewrote `scripts/export_stars_only.py` to run the full exporter inside a temporary copy, then copy back only `bright_stars.json` and the `stars` field in `solar-data.js`. This avoids Windows/OneDrive permission errors and prevents non-star map layers from being regenerated.
