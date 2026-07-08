import json, math, os, shutil, zipfile, stat
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

BASE=Path('/mnt/data/WEB_v18_restore_v10_map')
OUT=Path('/mnt/data/WEB_v19_cloud_pipeline')

def rm(p):
    if not p.exists(): return
    if p.is_file() or p.is_symlink():
        try: p.unlink()
        except PermissionError:
            os.chmod(p, stat.S_IWUSR|stat.S_IRUSR); p.unlink()
        return
    shutil.rmtree(p, ignore_errors=True)

rm(OUT)
shutil.copytree(BASE, OUT)
regions_path=OUT/'assets/data/regions.json'
regions=json.load(open(regions_path,encoding='utf-8'))

# --- robust 3D cloud mesh generator ---
def eff_radii(radius, kind='cloud'):
    if not isinstance(radius,(list,tuple)):
        rx=ry=rz=float(radius or 8)
    else:
        rx=float(radius[0] if len(radius)>0 else 8)
        ry=float(radius[1] if len(radius)>1 else rx)
        rz=float(radius[2] if len(radius)>2 else min(rx,ry)*0.35)
    rx=max(rx,2.5); ry=max(ry,2.5)
    # The previous visualisation inherited extremely thin z radii.  This is the
    # only deliberate morphology change: keep XY sizes but give every molecular
    # / dust component a physically visible line-of-sight thickness.
    xy_geom=math.sqrt(rx*ry)
    if kind == 'dust':
        rz_eff=max(rz*3.2, 0.42*xy_geom, 10.0)
    else:
        rz_eff=max(rz*3.0, 0.36*xy_geom, 8.0)
    rz_eff=min(rz_eff, max(rx,ry)*0.85)
    return rx,ry,rz_eff

def make_mesh_for_region(region):
    blobs=region.get('blobs') or []
    if not blobs:
        return []
    kind=region.get('kind','cloud')
    parsed=[]
    mins=np.array([1e9,1e9,1e9], dtype=float)
    maxs=-mins
    for b in blobs:
        x=float(b.get('X_pc', region.get('X_pc',0)) or 0)
        y=float(b.get('Y_pc', region.get('Y_pc',0)) or 0)
        z=float(b.get('Z_pc', region.get('Z_pc',0)) or 0)
        rx,ry,rz=eff_radii(b.get('radius_pc', region.get('radius_pc',[8,8,8])), kind)
        angle=math.radians(float(b.get('angle_deg',0) or 0))
        ca,sa=math.cos(angle),math.sin(angle)
        w=float(b.get('alpha', region.get('alpha', 0.08)) or 0.08)
        # alpha values in YAML/JSON are display opacities, not masses, so keep
        # weights in a narrow range; otherwise one blob dominates the surface.
        w=max(0.75, min(1.45, 8.0*w))
        parsed.append((x,y,z,rx,ry,rz,ca,sa,w))
        pad=np.array([rx,ry,rz])*2.25
        mins=np.minimum(mins, np.array([x,y,z])-pad)
        maxs=np.maximum(maxs, np.array([x,y,z])+pad)
    extent=float(np.max(maxs-mins))
    # Keep maps usable in the browser. Coarser for huge complexes, sharper for small ones.
    step=float(np.clip(extent/74.0, 3.8, 7.2))
    # Expand again to guarantee the isosurface dies inside the box, not at the box wall.
    margin=max(18.0, 2.0*step)
    mins-=margin; maxs+=margin
    xs=np.arange(mins[0], maxs[0]+step, step, dtype=np.float32)
    ys=np.arange(mins[1], maxs[1]+step, step, dtype=np.float32)
    zs=np.arange(mins[2], maxs[2]+step, step, dtype=np.float32)
    # Guard against pathological grids
    if xs.size*ys.size*zs.size > 1_800_000:
        step *= (xs.size*ys.size*zs.size/1_800_000)**(1/3)
        xs=np.arange(mins[0], maxs[0]+step, step, dtype=np.float32)
        ys=np.arange(mins[1], maxs[1]+step, step, dtype=np.float32)
        zs=np.arange(mins[2], maxs[2]+step, step, dtype=np.float32)
    X,Y,Z=np.meshgrid(xs,ys,zs,indexing='ij')
    F=np.zeros(X.shape, dtype=np.float32)
    for x,y,z,rx,ry,rz,ca,sa,w in parsed:
        dx=X-x; dy=Y-y; dz=Z-z
        xp=ca*dx+sa*dy
        yp=-sa*dx+ca*dy
        q=(xp/rx)**2+(yp/ry)**2+(dz/rz)**2
        # Smooth Gaussian density. Clip far tails for speed/contrast.
        contrib=np.exp(-0.5*q).astype(np.float32)
        contrib[q>18]=0
        F += w*contrib
    F=gaussian_filter(F, sigma=0.75)
    # Chosen so q~2.1-2.4 contour; preserves scale but closes surfaces robustly.
    level=0.34 if kind=='dust' else 0.36
    if float(F.max()) <= level:
        level=float(F.max())*0.48
    verts, faces, norms, vals = marching_cubes(F, level=level, spacing=(step,step,step), allow_degenerate=False)
    verts[:,0]+=float(xs[0]); verts[:,1]+=float(ys[0]); verts[:,2]+=float(zs[0])
    # Flip winding if needed? Three.js double-sided, so not important.
    # Quantise for JSON size.
    vertices=[[round(float(a),3),round(float(b),3),round(float(c),3)] for a,b,c in verts]
    faces=[[int(a),int(b),int(c)] for a,b,c in faces]
    alpha=0.18 if kind=='dust' else 0.22
    return [{'vertices':vertices, 'faces':faces, 'alpha':alpha, 'mesh_source':'python_density_marching_cubes'}]

counts=[]
for r in regions:
    if r.get('kind') in ('cloud','dust') and r.get('blobs'):
        try:
            meshes=make_mesh_for_region(r)
            if meshes:
                r['cloud_meshes']=meshes
                # Keep the original blobs for traceability/fallback, but the renderer will use cloud_meshes first.
                counts.append((r['name'],len(meshes),len(meshes[0]['vertices']),len(meshes[0]['faces'])))
        except Exception as e:
            print('FAILED', r.get('name'), e)

json.dump(regions, open(regions_path,'w',encoding='utf-8'), separators=(',',':'))

# regenerate solar-data.js from JSON files
base=OUT/'assets/data'
obj={
    'stars': json.load(open(base/'bright_stars.json',encoding='utf-8')),
    'constellationStars': json.load(open(base/'constellation_stars.json',encoding='utf-8')),
    'systems': json.load(open(base/'systems.json',encoding='utf-8')),
    'regions': json.load(open(base/'regions.json',encoding='utf-8')),
    'dustClouds': json.load(open(base/'dust_clouds.json',encoding='utf-8')),
    'mapMeta': json.load(open(base/'map_meta.json',encoding='utf-8')),
}
with open(OUT/'assets/js/solar-data.js','w',encoding='utf-8') as f:
    f.write('window.SOLAR_MAP_DATA = ')
    json.dump(obj,f,separators=(',',':'))
    f.write(';\n')

# Patch JS renderer to index vertices instead of duplicating triangles, but do not alter orientation or map controls.
js_path=OUT/'assets/js/solar-map.js'
js=js_path.read_text(encoding='utf-8')
old="""  function renderCloudMesh(targetGroup, meshData, fallbackColor, opts = {}) {
    const verts = meshData.vertices || [];
    const faces = meshData.faces || [];
    if (!verts.length || !faces.length) return null;
    const color = meshData.color ? cssHexToThreeColor(meshData.color, fallbackColor) : fallbackColor;
    const positions = [];
    faces.forEach((face) => {
      face.forEach((idx) => {
        const v = verts[idx];
        if (!v) return;
        const p = galToVector({ X_pc: Number(v[0]), Y_pc: Number(v[1]), Z_pc: Number(v[2]) });
        positions.push(p.x, p.y, p.z);
      });
    });
    if (!positions.length) return null;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.computeVertexNormals();"""
new="""  function renderCloudMesh(targetGroup, meshData, fallbackColor, opts = {}) {
    const verts = meshData.vertices || [];
    const faces = meshData.faces || [];
    if (!verts.length || !faces.length) return null;
    const color = meshData.color ? cssHexToThreeColor(meshData.color, fallbackColor) : fallbackColor;
    const positions = [];
    verts.forEach((v) => {
      const p = galToVector({ X_pc: Number(v[0]), Y_pc: Number(v[1]), Z_pc: Number(v[2]) });
      positions.push(p.x, p.y, p.z);
    });
    const indices = [];
    faces.forEach((face) => {
      if (face.length >= 3) indices.push(Number(face[0]), Number(face[1]), Number(face[2]));
    });
    if (!positions.length || !indices.length) return null;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();"""
if old in js:
    js=js.replace(old,new)
else:
    print('renderCloudMesh block not found')
# Keep good enough pixel ratio but not huge
js=js.replace("const fullscreenBoost = document.fullscreenElement ? 1.75 : 1.5;\n    return Math.min(base, fullscreenBoost);",
              "const fullscreenBoost = document.fullscreenElement ? 1.25 : 1.0;\n    return Math.min(base, fullscreenBoost);")
js=js.replace("antialias: true,", "antialias: false,")
js_path.write_text(js,encoding='utf-8')

# Add the generator script to the package for reproducibility.
scripts=OUT/'scripts'; scripts.mkdir(exist_ok=True)
shutil.copy2('/mnt/data/build_v19.py', scripts/'build_cloud_meshes_python_density.py')

with open(OUT/'FIXES_APPLIED.md','a',encoding='utf-8') as f:
    f.write('\n- Rebuilt Zucker/dust region `cloud_meshes` from the JSON blobs with a real Python 3D density field and marching-cubes surfaces.\n')
    f.write('- XY centres/scales are preserved; Z radii are inflated locally before meshing so the clouds are not paper-thin.\n')
    f.write('- `solar-data.js` regenerated from JSON. `renderCloudMesh` now uses indexed BufferGeometry for speed.\n')
    f.write('- No Lallement H5 file was present in the package/session, so the true STILISM H5 is not reprocessed here; the included script is ready for the same mesh-export path once the H5 is supplied.\n')

print('meshes')
for c in counts:
    print(c)

# zip
ZIP=Path('/mnt/data/WEB_fixed_v19.zip')
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        z.write(p, p.relative_to(OUT.parent))
print('zip', ZIP, ZIP.stat().st_size/1024/1024)
