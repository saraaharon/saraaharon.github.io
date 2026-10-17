"""Build 15-min isochrone bands around airports (nearest-airport drive-time surface, clipped to M2)."""
import json, numpy as np, pandas as pd, shapely
from pyproj import Transformer
from shapely.geometry import shape, Polygon, mapping
from shapely.ops import unary_union, transform
from shapely import STRtree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from contourpy import contour_generator, FillType

U = './'   # folder with uatM2.json, airports_access_py.csv, population_grid_3844.csv
STEP, CELL = 15, 1000.0
to_m  = Transformer.from_crs(4326, 3844, always_xy=True).transform
to_ll = Transformer.from_crs(3844, 4326, always_xy=True).transform

AP = {'BCM':(26.9103,46.5219),'SCV':(26.3541,47.6875),'IAS':(27.6206,47.1785),'OTP':(26.0850,44.5711),
      'BBU':(26.1021,44.5032),'CND':(28.4883,44.3622),'TCE':(28.7143,45.0625),'GHV':(25.5300,45.7844),
      'TGM':(24.4125,46.4677),'CLJ':(23.6862,46.7852),'BAY':(23.4701,47.6584)}

# --- M2 shape (EPSG:3844)
d = json.load(open(U+'uatM2.json'))
m2 = unary_union([transform(to_m, shape(f['geometry'])).buffer(0) for f in d['features']]).buffer(0)

# --- drive-time points (+ airports anchored at 0 min)
a = pd.read_csv(U+'airports_access_py.csv')
px, py = to_m(a.lon.values, a.lat.values)
ax, ay = to_m(*zip(*AP.values()))
X = np.r_[px, ax]; Y = np.r_[py, ay]; Z = np.r_[a.minutes.values, np.zeros(len(AP))]
LAB = np.r_[a.nearest_id.values, np.array(list(AP.keys()))]

# --- grid over M2
minx, miny, maxx, maxy = m2.bounds
xs = np.arange(minx-5000, maxx+5000, CELL); ys = np.arange(miny-5000, maxy+5000, CELL)
GX, GY = np.meshgrid(xs, ys)
pts = np.c_[X, Y]
f = LinearNDInterpolator(pts, Z)(GX, GY)
nn = NearestNDInterpolator(pts, Z)(GX, GY)
f = np.where(np.isnan(f), nn, f)
f = gaussian_filter(f, 1.0)

# --- catchments: nearest point's airport, smoothed indicator per airport
_, idx = cKDTree(pts).query(np.c_[GX.ravel(), GY.ravel()])
lab = LAB[idx].reshape(GX.shape)
inside = shapely.contains_xy(m2.buffer(3000), GX, GY)
print('M2 cells:', inside.sum(), '| cells whose nearest airport is NOT one of the 11:', (inside & ~np.isin(lab, list(AP))).sum())
ind = {k: gaussian_filter((lab == k).astype(float), 2.0) for k in AP}
stack = np.stack([ind[k] for k in AP]); win = np.array(list(AP))[stack.argmax(0)]

def polys(z, lo, hi):
    g = contour_generator(xs, ys, z, fill_type=FillType.OuterOffset)
    pl, of = g.filled(lo, hi)
    out = []
    for p, o in zip(pl, of):
        rings = [p[o[i]:o[i+1]] for i in range(len(o)-1)]
        if len(rings[0]) >= 4: out.append(Polygon(rings[0], [r for r in rings[1:] if len(r) >= 4]))
    return unary_union([q.buffer(0) for q in out]) if out else Polygon()

catch = {k: polys((win == k).astype(float), 0.5, 1.5).intersection(m2) for k in AP}

nb = int(np.ceil(np.nanmax(f[inside]) / STEP))
bands = [polys(f, -1 if k == 0 else k*STEP, (k+1)*STEP) for k in range(nb)]

# --- population (2021 1 km grid, x/y already in 3844)
pg = pd.read_csv(U+'population_grid_3844.csv', usecols=['x_3844','y_3844','population'])
P = shapely.points(pg.x_3844.values, pg.y_3844.values); pop_all = pg.population.values

recs = []
for k in AP:
    for b, bp in enumerate(bands):
        g = bp.intersection(catch[k])
        if g.is_empty or g.area < 2e5: continue
        recs.append((k, b, g))
tree = STRtree([r[2] for r in recs])
pi, gi = tree.query(P, predicate='within')
pop = np.zeros(len(recs)); np.add.at(pop, gi, pop_all[pi])
m2_pts = shapely.contains_xy(m2, pg.x_3844.values, pg.y_3844.values)
print('M2 pop (grid):', pop_all[m2_pts].sum(), '| pop captured by band polygons:', pop.sum())

def rnd(c): return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 4), round(c[1], 4)]
feats = []
for (k, b, g), p in zip(recs, pop):
    gl = transform(to_ll, g.simplify(250, preserve_topology=True))
    gj = mapping(gl)
    feats.append({'type':'Feature','properties':{'ap':k,'band':b,'from':b*STEP,'to':(b+1)*STEP,
        'pop':int(p),'km2':round(g.area/1e6,1)},
        'geometry':{'type':gj['type'],'coordinates':rnd(json.loads(json.dumps(gj['coordinates'])))}})
outline = transform(to_ll, m2.simplify(400, preserve_topology=True))
oj = mapping(outline)
open('data/isochrones.js','w').write(
    'window.ISO='+json.dumps({'type':'FeatureCollection','features':feats},separators=(',',':'))+';\n'
    'window.ISO_META='+json.dumps({'step':STEP,'bands':nb,'totalPop':int(pop_all[m2_pts].sum())})+';\n'
    'window.M2_OUTLINE='+json.dumps({'type':'Feature','properties':{},'geometry':{'type':oj['type'],'coordinates':rnd(json.loads(json.dumps(oj['coordinates'])))}},separators=(',',':'))+';')
import os; print('features', len(feats), 'bands', nb, 'size KB', os.path.getsize('data/isochrones.js')//1024)
print(pd.DataFrame([r['properties'] for r in feats]).groupby('ap')[['pop','km2']].sum())
