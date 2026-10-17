#!/usr/bin/env python3
"""
roadnet_access.py - drive time to the NEAREST facility for every origin, in pure Python.
No OSRM, no Docker, no internet: it builds a road graph from an OpenStreetMap .pbf and runs ONE
multi-source Dijkstra from all facilities, so 69,000 origins cost the same as 3,000.

Needs: pip install osmium numpy scipy        (geopandas only if origins are a shapefile)
Keeps nearest_access.py's column names, so analyze_pop_access.py works on the output.

First run builds and caches the graph (several minutes, a few GB RAM for Romania). Later runs load the cache:
  python roadnet_access.py --pbf romania-261008.osm.pbf --graph romania_roads.npz \
      --origins population_grid.csv --facilities universities.csv --out uni_pop_access.csv
  python roadnet_access.py --graph romania_roads.npz --origins lau_centroids.csv \
      --facilities airports.csv --out airports_access_py.csv

Model (close to the OSRM car profile so results are comparable):
  * default speeds by road class (km/h), a maxspeed tag is used at 80% of its value
  * oneway streets and roundabouts respected; access=no/private skipped; ferries included at 10 km/h
  * origins/facilities snap to the nearest road node; the snapping distance is driven at --offroad-kmh
  * output has minutes, km_straight (straight-line to the chosen facility) and snap_km; road km is not computed
"""
import argparse, csv, math, os, re, sys, time
from array import array
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

SPEEDS = {"motorway": 90, "motorway_link": 45, "trunk": 85, "trunk_link": 40, "primary": 65,
          "primary_link": 30, "secondary": 55, "secondary_link": 25, "tertiary": 40,
          "tertiary_link": 20, "unclassified": 25, "residential": 25, "living_street": 10, "road": 20}
SERVICE_KMH = 15
FERRY_KMH = 10
MAXSPEED_FACTOR = 0.8
R_EARTH = 6371000.0


# ------------------------------------------------------------------ graph building
def _make_handler(include_service):
    import osmium

    class Roads(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.w = array("q"), array("q"), array("f")
            self.nref, self.nlon, self.nlat = array("q"), array("d"), array("d")
            self.ways = 0

        def way(self, w):
            t = w.tags
            hw = t.get("highway")
            if hw:
                speed = SPEEDS.get(hw, SERVICE_KMH if (hw == "service" and include_service) else None)
                if speed is None:
                    return
                ms = t.get("maxspeed", "")
                m = re.match(r"^\s*(\d{1,3})\s*(km/h)?\s*$", ms)
                if m and int(m.group(1)) > 0:
                    speed = int(m.group(1)) * MAXSPEED_FACTOR
            elif t.get("route") == "ferry":
                speed = FERRY_KMH
            else:
                return
            if t.get("access") in ("no", "private") or t.get("motor_vehicle") in ("no", "private") or t.get("area") == "yes":
                return
            ow = t.get("oneway", "")
            fwd, bwd = True, True
            if ow in ("yes", "true", "1") or t.get("junction") in ("roundabout", "circular") or (hw in ("motorway", "motorway_link") and ow != "no"):
                bwd = False
            elif ow == "-1":
                fwd = False
            pts = []
            for n in w.nodes:
                loc = n.location
                if loc.valid():
                    pts.append((n.ref, loc.lon, loc.lat))
            if len(pts) < 2:
                return
            self.ways += 1
            ms_per = speed / 3.6
            for ref, lon, lat in pts:
                self.nref.append(ref); self.nlon.append(lon); self.nlat.append(lat)
            for (r1, x1, y1), (r2, x2, y2) in zip(pts, pts[1:]):
                if r1 == r2:
                    continue
                dx = math.radians(x2 - x1) * math.cos(math.radians((y1 + y2) / 2))
                dy = math.radians(y2 - y1)
                sec = R_EARTH * math.hypot(dx, dy) / ms_per
                if fwd:
                    self.a.append(r1); self.b.append(r2); self.w.append(sec)
                if bwd:
                    self.a.append(r2); self.b.append(r1); self.w.append(sec)

    return Roads()


def build_graph(pbf, include_service=False):
    t0 = time.time()
    print(f"reading {pbf} (this takes a while for a whole country)...", flush=True)
    h = _make_handler(include_service)
    h.apply_file(pbf, locations=True)
    print(f"  {h.ways} road ways, {len(h.a)} directed segments, {time.time() - t0:.0f}s", flush=True)
    nref = np.frombuffer(h.nref, dtype=np.int64)
    uniq, first = np.unique(nref, return_index=True)
    lon = np.frombuffer(h.nlon, dtype=np.float64)[first]
    lat = np.frombuffer(h.nlat, dtype=np.float64)[first]
    a = np.searchsorted(uniq, np.frombuffer(h.a, dtype=np.int64)).astype(np.int64)
    b = np.searchsorted(uniq, np.frombuffer(h.b, dtype=np.int64)).astype(np.int64)
    w = np.frombuffer(h.w, dtype=np.float32).copy()
    n = len(uniq)
    # keep the cheapest of parallel edges
    order = np.argsort(w, kind="stable")
    key = a[order] * n + b[order]
    _, idx = np.unique(key, return_index=True)
    keep = order[idx]
    a, b, w = a[keep], b[keep], w[keep]
    # keep only the largest connected piece (drops isolated roads/islands that trap snapping)
    comp = connected_components(csr_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n)),
                                directed=True, connection="weak")[1]
    big = np.bincount(comp).argmax()
    ok = comp == big
    remap = np.full(n, -1, dtype=np.int64); remap[ok] = np.arange(ok.sum())
    e = ok[a] & ok[b]
    print(f"  graph: {ok.sum()} nodes, {e.sum()} edges ({n - ok.sum()} nodes dropped as disconnected)", flush=True)
    return lon[ok], lat[ok], remap[a[e]].astype(np.int32), remap[b[e]].astype(np.int32), w[e]


def load_or_build(pbf, graph_path, rebuild, include_service):
    if graph_path and os.path.exists(graph_path) and not rebuild:
        z = np.load(graph_path)
        print(f"loaded graph {graph_path}: {len(z['lon'])} nodes, {len(z['a'])} edges")
        return z["lon"], z["lat"], z["a"], z["b"], z["w"]
    if not pbf:
        sys.exit("No cached graph found: give --pbf to build one.")
    g = build_graph(pbf, include_service)
    if graph_path:
        np.savez(graph_path, lon=g[0], lat=g[1], a=g[2], b=g[3], w=g[4])
        print(f"graph saved -> {graph_path}")
    return g


# ------------------------------------------------------------------ analysis
def hav_km(lon1, lat1, lon2, lat2):
    p = np.pi / 180
    dx = (lon1 - lon2) * p * np.cos((lat1 + lat2) / 2 * p)
    dy = (lat1 - lat2) * p
    return 6371.0 * np.hypot(dx, dy)


def nearest_by_road(graph, O, F, offroad_kmh, max_snap_km, snap_k=5, window_km=0.5):
    lon, lat, a, b, w = graph
    n = len(lon)
    c0 = math.cos(math.radians(float(lat.mean())))
    tree = cKDTree(np.c_[lon * c0, lat])

    def snap(pts, k=1):
        d, i = tree.query(np.c_[pts[:, 0] * c0, pts[:, 1]], k=k)
        if k == 1:
            d, i = d[:, None], i[:, None]
        return i, d * 111.195          # km (approx.)

    o_ll = np.array([[o[1], o[2]] for o in O]); f_ll = np.array([[f[1], f[2]] for f in F])
    oi, osnap = snap(o_ll, max(1, min(snap_k, n))); fi, fsnap = snap(f_ll); fi, fsnap = fi[:, 0], fsnap[:, 0]
    if fsnap.max() > 1:
        print(f"WARNING: {(fsnap > 1).sum()} facilities are more than 1 km from a road (max {fsnap.max():.1f} km)")
    # reversed graph: Dijkstra from the facilities gives the time FROM every node TO the nearest facility
    R = csr_matrix((w.astype(np.float64), (b, a)), shape=(n, n))
    node_to_fac = {}
    for k, node in enumerate(fi):
        node_to_fac.setdefault(int(node), k)
    srcs = np.array(sorted(node_to_fac), dtype=np.int32)
    t0 = time.time()
    dist, _, src = dijkstra(R, directed=True, indices=srcs, min_only=True, return_predecessors=True)
    print(f"multi-source Dijkstra over {n} nodes: {time.time() - t0:.0f}s", flush=True)

    # each origin tries its snap_k nearest road nodes (no more than window_km farther than the closest one)
    # and keeps the lowest total time; this avoids snapping onto a dead-end or one-way stub
    with np.errstate(invalid="ignore"):
        tot = dist[oi] + osnap * 1000 / (offroad_kmh / 3.6)
    tot[~np.isfinite(dist[oi])] = np.inf
    tot[osnap > osnap[:, :1] + window_km] = np.inf      # only nodes about as close as the nearest one
    best = tot.argmin(axis=1)
    ar = np.arange(len(O))
    node_o, snap_o, t_o = oi[ar, best], osnap[ar, best], tot[ar, best]
    rows = []
    for k, o in enumerate(O):
        node = int(node_o[k])
        if not np.isfinite(t_o[k]) or src[node] < 0:
            rows.append((o, "", "", "", "", round(float(snap_o[k]), 3), "NO_ROUTE")); continue
        j = node_to_fac[int(src[node])]
        km_s = float(hav_km(o[1], o[2], F[j][1], F[j][2]))
        status = "Ok" if snap_o[k] <= max_snap_km else "FAR_FROM_ROAD"
        rows.append((o, F[j][0], F[j][3], round(float(t_o[k] / 60), 2), round(km_s, 2), round(float(snap_o[k]), 3), status))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pbf"); ap.add_argument("--graph", default="roads_graph.npz")
    ap.add_argument("--rebuild", action="store_true"); ap.add_argument("--include-service", action="store_true")
    ap.add_argument("--origins"); ap.add_argument("--id-field", default="id")
    ap.add_argument("--lon", default="lon"); ap.add_argument("--lat", default="lat")
    ap.add_argument("--facilities")
    ap.add_argument("--fac-id", default="id"); ap.add_argument("--fac-name", default="name")
    ap.add_argument("--fac-lon", default="lon"); ap.add_argument("--fac-lat", default="lat")
    ap.add_argument("--out")
    ap.add_argument("--offroad-kmh", type=float, default=20.0)
    ap.add_argument("--max-snap-km", type=float, default=5.0)
    ap.add_argument("--snap-k", type=int, default=5, help="road nodes tried per origin (best total time wins)")
    ap.add_argument("--snap-window-km", type=float, default=0.5, help="only try nodes within this much of the closest one")
    a = ap.parse_args()

    graph = load_or_build(a.pbf, a.graph, a.rebuild, a.include_service)
    if not (a.origins and a.facilities and a.out):
        print("Graph ready. Add --origins, --facilities and --out to run an analysis.")
        return
    from nearest_access import read_csv, read_origins   # same folder
    O = read_origins(a.origins, a.id_field, a.lon, a.lat)
    F = read_csv(a.facilities, a.fac_id, a.fac_lon, a.fac_lat, a.fac_name)
    print(f"{len(O)} origins, {len(F)} facilities")
    rows = nearest_by_road(graph, O, F, a.offroad_kmh, a.max_snap_km, a.snap_k, a.snap_window_km)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "lon", "lat", "nearest_id", "nearest_name", "minutes", "km", "km_straight", "snap_km", "status"])
        for o, fid, fname, mins, kms, snap_km, status in rows:
            wr.writerow([o[0], o[1], o[2], fid, fname, mins, "", kms, snap_km, status])
    bad = sum(1 for r in rows if r[-1] != "Ok")
    print(f"done -> {a.out}  ({bad} rows not Ok)")


if __name__ == "__main__":
    main()
