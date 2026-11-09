"""Pure / network logic for the bus route optimizer (no Streamlit imports here,
so it can be unit-tested). Network calls are isolated in small functions that
tests can monkeypatch."""
from __future__ import annotations

import io
import math
import re
import time

import numpy as np
import pandas as pd
import requests
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

EARTH_R = 6371000.0
# Without Google, a straight line under-estimates walking. Typical urban grid detour ≈ 1.3.
WALK_DETOUR_FACTOR = 1.3
NOMINATIM_HEADERS = {"User-Agent": "BusRouteOptimizerApp/2.0 (school route planner)"}
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter"]
GOOGLE_DM_MAX_DESTS = 25  # Distance Matrix: ≤25 destinations, ≤100 elements per request


# ---------------------------------------------------------------- geometry

def haversine_matrix(coords1, coords2):
    coords1, coords2 = np.asarray(coords1, float), np.asarray(coords2, float)
    lat1, lon1 = np.radians(coords1[:, 0])[:, None], np.radians(coords1[:, 1])[:, None]
    lat2, lon2 = np.radians(coords2[:, 0])[None, :], np.radians(coords2[:, 1])[None, :]
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    return EARTH_R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def in_bbox(lat, lon, bbox):
    """bbox = (min_lat, min_lon, max_lat, max_lon)."""
    return bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]


def pad_bbox(bbox, meters):
    dlat = meters / 111_320.0
    dlon = meters / (111_320.0 * max(math.cos(math.radians((bbox[0] + bbox[2]) / 2)), 0.1))
    return (bbox[0] - dlat, bbox[1] - dlon, bbox[2] + dlat, bbox[3] + dlon)


def decode_polyline(polyline_str):
    index, lat, lng, coords = 0, 0, 0, []
    while index < len(polyline_str):
        deltas = []
        for _ in range(2):
            shift, result = 0, 0
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            deltas.append(~(result >> 1) if result & 1 else result >> 1)
        lat += deltas[0]
        lng += deltas[1]
        coords.append((lat / 1e5, lng / 1e5))
    return coords


# ---------------------------------------------------------------- input

def read_table(raw: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        for enc in ("utf-8-sig", "cp1255", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc)
            except UnicodeDecodeError:
                continue
    return pd.read_excel(io.BytesIO(raw))


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or not str(v).strip()


def build_passenger_rows(df: pd.DataFrame, address_col: str, name_col: str | None):
    """Row-wise so names never drift from their addresses (the old code dropna'd
    the two columns independently, shifting names after any blank cell)."""
    rows = []
    for i, rec in enumerate(df.to_dict("records")):
        addr = rec.get(address_col)
        if _blank(addr):
            continue
        name = rec.get(name_col) if name_col else None
        rows.append({
            "name": f"Passenger {i + 1}" if _blank(name) else str(name).strip(),
            "address": str(addr).strip(),
        })
    return rows


def clean_street(address: str, city: str) -> str:
    """'הצדף 18/4 מיקוד 8103821' -> 'הצדף 18'. Removes zip, trailing city and
    apartment number so the geocoder gets just street + building."""
    s = str(address)
    s = re.sub(r"מיקוד\s*\d*", " ", s)
    s = re.sub(r"\b\d{5,7}\b", " ", s)  # bare zip codes
    if city and city.strip():
        s = re.sub(rf"[,\s]*{re.escape(city.strip())}\s*$", " ", s.strip())
    s = re.sub(r"(\d+)\s*/\s*\d+", r"\1", s)  # building/apartment
    s = re.sub(r"\s*(דירה|כניסה|קומה)\s*\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def street_variants(street: str) -> list[str]:
    """Spelling variants OSM is likely to use: ש"י→שי / dropped initials,
    שד→שדרות, without רבי/הרב/רחוב prefixes. Original first."""
    out = [street]

    def add(v):
        v = re.sub(r"\s+", " ", v).strip()
        if v and v not in out:
            out.append(v)

    add(re.sub(r"^(שד'?|שד\.)\s+", "שדרות ", street))
    add(re.sub(r"^(רבי|הרב|רחוב|רח'|רח\.)\s+", "", street))
    add(re.sub(r'["״]', "", street))
    add(re.sub(r'\S+["״]\S*\s*', "", street))
    return out


def _norm(s: str) -> str:
    return re.sub(r"[\s\-'\"׳״]", "", str(s)).lower()


def google_result_in_city(result: dict, city_names) -> bool:
    wanted = {_norm(c) for c in city_names if c}
    for comp in result.get("address_components", []):
        if {"locality", "administrative_area_level_2", "postal_town"} & set(comp.get("types", [])):
            if _norm(comp.get("long_name", "")) in wanted or _norm(comp.get("short_name", "")) in wanted:
                return True
    return False


# ---------------------------------------------------------------- geocoding

def geocode_city(city: str, gmaps_key: str = "", lang: str = "he"):
    """Returns {'bbox': (minlat,minlon,maxlat,maxlon), 'names': [...]} or None."""
    if not city.strip():
        return None
    names = [city.strip()]
    if gmaps_key:
        for lg in (lang, "en"):
            try:
                res = requests.get(
                    "https://maps.googleapis.com/maps/api/geocode/json",
                    params={"address": city, "components": "country:IL", "key": gmaps_key, "language": lg},
                    timeout=10,
                ).json()
            except requests.RequestException:
                break
            if res.get("status") == "OK":
                r = res["results"][0]
                names.append(r["address_components"][0]["long_name"])
                vp = r["geometry"].get("bounds") or r["geometry"]["viewport"]
                bbox = (vp["southwest"]["lat"], vp["southwest"]["lng"], vp["northeast"]["lat"], vp["northeast"]["lng"])
        if len(names) > 1:
            return {"bbox": bbox, "names": list(dict.fromkeys(names))}
    try:
        data = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"city": city, "countrycodes": "il", "format": "json", "limit": 1,
                    "addressdetails": 1, "accept-language": "he,en"},
            headers=NOMINATIM_HEADERS, timeout=10,
        ).json()
    except (requests.RequestException, ValueError):
        return None
    if not data:
        return None
    bb = [float(x) for x in data[0]["boundingbox"]]  # [minlat, maxlat, minlon, maxlon]
    return {"bbox": (bb[0], bb[2], bb[1], bb[3]), "names": names}


def geocode_address(address: str, city: str = "", gmaps_key: str = "", lang: str = "he", city_info=None):
    """Returns {'lat','lon','source'} or None. With a city, any result outside it
    is rejected, and so are street-less results that only hit the city centre."""
    street = clean_street(address, city) if city else str(address).strip()
    bbox = pad_bbox(city_info["bbox"], 300) if city_info else None
    names = city_info["names"] if city_info else [city]

    def ok(lat, lon):
        return bbox is None or in_bbox(lat, lon, bbox)

    if gmaps_key:
        params = {"address": f"{street}, {city}" if city else street, "key": gmaps_key,
                  "language": lang, "region": "il"}
        params["components"] = f"country:IL|locality:{city}" if city else "country:IL"
        try:
            res = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params, timeout=10).json()
        except (requests.RequestException, ValueError):
            res = {}
        for r in res.get("results", []):
            types = set(r.get("types", []))
            if types & {"locality", "political", "postal_code"} and not types & {"street_address", "premise", "route", "subpremise"}:
                continue  # only matched the city/zip, not the street
            loc = r["geometry"]["location"]
            if city and not (google_result_in_city(r, names) or ok(loc["lat"], loc["lng"])):
                continue
            if not ok(loc["lat"], loc["lng"]):
                continue
            return {"lat": float(loc["lat"]), "lon": float(loc["lng"]), "source": "google"}

    for n, variant in enumerate(street_variants(street) if city else [street]):
        if n:
            time.sleep(1.0)  # Nominatim usage policy: ≤1 request/second
        params = {"format": "json", "limit": 3, "countrycodes": "il", "addressdetails": 1, "accept-language": "he,en"}
        if city:
            params.update({"street": variant, "city": city})
        else:
            params["q"] = variant
        if bbox:
            params.update({"viewbox": f"{bbox[1]},{bbox[2]},{bbox[3]},{bbox[0]}", "bounded": 1})
        try:
            data = requests.get("https://nominatim.openstreetmap.org/search", params=params,
                                headers=NOMINATIM_HEADERS, timeout=10).json()
        except (requests.RequestException, ValueError):
            continue
        for d in data or []:
            lat, lon = float(d["lat"]), float(d["lon"])
            if ok(lat, lon):
                return {"lat": lat, "lon": lon, "source": "osm"}
    return None


# ---------------------------------------------------------------- bus stops

STOP_COLS = ["stop_id", "name", "lat", "lon"]


def fetch_bus_stops_osm(bbox, lang="he") -> pd.DataFrame:
    b = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    query = f"""[out:json][timeout:25];
    ( node["highway"="bus_stop"]({b});
      node["public_transport"="platform"]["bus"="yes"]({b}); );
    out body;"""
    elements = None
    # Overpass rejects requests without a meaningful User-Agent (406/429) — the
    # old code sent none, so it always got zero stops. Try mirrors in turn.
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=NOMINATIM_HEADERS, timeout=60)
            if r.ok:
                elements = r.json().get("elements", [])
                break
        except (requests.RequestException, ValueError):
            continue
    if elements is None:
        return pd.DataFrame(columns=STOP_COLS)
    stops = []
    for el in elements:
        t = el.get("tags", {})
        name = t.get(f"name:{lang}") or t.get("name:he") or t.get("name") or t.get("name:en") or f"Stop #{el['id']}"
        ref = t.get("ref")
        stops.append({"stop_id": f"osm:{el['id']}", "name": f"{name} ({ref})" if ref else name,
                      "lat": float(el["lat"]), "lon": float(el["lon"])})
    return pd.DataFrame(stops, columns=STOP_COLS)


def _grid_centers(bbox, radius_m):
    step = radius_m * 1.4  # circles of radius r on a 1.4r grid overlap fully
    lat_steps = max(1, math.ceil((bbox[2] - bbox[0]) * 111_320 / step))
    mid = math.radians((bbox[0] + bbox[2]) / 2)
    lon_steps = max(1, math.ceil((bbox[3] - bbox[1]) * 111_320 * math.cos(mid) / step))
    for i in range(lat_steps):
        for j in range(lon_steps):
            yield (bbox[0] + (i + 0.5) * (bbox[2] - bbox[0]) / lat_steps,
                   bbox[1] + (j + 0.5) * (bbox[3] - bbox[1]) / lon_steps)


def fetch_bus_stops_google(bbox, gmaps_key, lang="he", radius_m=800, max_calls=40) -> pd.DataFrame:
    """Google Places (New) Nearby Search for bus stops over a grid covering bbox."""
    stops = []
    for n, (lat, lon) in enumerate(_grid_centers(bbox, radius_m)):
        if n >= max_calls:
            break
        try:
            res = requests.post(
                "https://places.googleapis.com/v1/places:searchNearby",
                headers={"X-Goog-Api-Key": gmaps_key,
                         "X-Goog-FieldMask": "places.id,places.displayName,places.location"},
                json={"includedTypes": ["bus_stop", "bus_station"], "maxResultCount": 20,
                      "languageCode": lang,
                      "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lon},
                                                         "radius": float(radius_m)}}},
                timeout=10,
            ).json()
        except (requests.RequestException, ValueError):
            continue
        for p in res.get("places", []):
            stops.append({"stop_id": f"g:{p['id']}", "name": p.get("displayName", {}).get("text", "Bus stop"),
                          "lat": p["location"]["latitude"], "lon": p["location"]["longitude"]})
    return pd.DataFrame(stops, columns=STOP_COLS)


def merge_stops(frames, dedupe_m=20) -> pd.DataFrame:
    """Concatenate stop sources (earlier frames win) and drop points within dedupe_m."""
    df = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) \
        if any(f is not None and not f.empty for f in frames) else pd.DataFrame(columns=STOP_COLS)
    if df.empty:
        return df
    df = df.drop_duplicates(subset="stop_id").reset_index(drop=True)
    coords = df[["lat", "lon"]].to_numpy()
    keep = []
    for i in range(len(df)):
        if keep and haversine_matrix(coords[i:i + 1], coords[keep]).min() < dedupe_m:
            continue
        keep.append(i)
    return df.iloc[keep].reset_index(drop=True)


# ---------------------------------------------------------------- walking distances

def _wp(lat, lon):
    return {"waypoint": {"location": {"latLng": {"latitude": float(lat), "longitude": float(lon)}}}}


def _google_walk_block(origins, dests, key):
    """Routes API computeRouteMatrix (the legacy Distance Matrix API is often
    not enabled on new keys)."""
    r = requests.post(
        "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix",
        headers={"X-Goog-Api-Key": key,
                 "X-Goog-FieldMask": "originIndex,destinationIndex,distanceMeters,condition"},
        json={"origins": [_wp(*o) for o in origins], "destinations": [_wp(*d) for d in dests],
              "travelMode": "WALK"},
        timeout=20,
    )
    out = np.full((len(origins), len(dests)), np.nan)
    if not r.ok:
        return out
    for el in r.json():
        if el.get("condition") == "ROUTE_EXISTS" and "distanceMeters" in el:
            out[el.get("originIndex", 0), el.get("destinationIndex", 0)] = el["distanceMeters"]
    return out


def walking_matrix(p_coords, s_coords, max_walk_m, gmaps_key="", min_candidates=3):
    """passengers x stops walking distance (m). Starts from air distance x detour
    factor; with a key, replaces it with real Google walking distance for every
    stop within max_walk by air (walk ≥ air, so farther stops can't qualify),
    plus each passenger's few nearest stops so far-away homes still get a real
    number."""
    p_coords, s_coords = np.asarray(p_coords, float), np.asarray(s_coords, float)
    air = haversine_matrix(p_coords, s_coords)
    mat = air * WALK_DETOUR_FACTOR
    if not gmaps_key or len(s_coords) == 0:
        return mat
    for i in range(len(p_coords)):
        cand = np.where(air[i] <= max_walk_m)[0]
        if len(cand) == 0:  # nobody in range: get real numbers for the nearest few
            cand = np.argsort(air[i])[:min_candidates]
        cand = cand[np.argsort(air[i, cand])][:GOOGLE_DM_MAX_DESTS]
        try:
            block = _google_walk_block(p_coords[i:i + 1], s_coords[cand], gmaps_key)
        except (requests.RequestException, ValueError):
            continue
        got = ~np.isnan(block[0])
        mat[i, cand[got]] = block[0, got]
    return mat


def _osrm_table(pts):
    locs = ";".join(f"{lon},{lat}" for lat, lon in pts)
    try:
        res = requests.get(f"https://router.project-osrm.org/table/v1/driving/{locs}",
                           params={"annotations": "distance"}, timeout=20).json()
    except (requests.RequestException, ValueError):
        return None
    if res.get("code") != "Ok":
        return None
    m = np.array(res["distances"], dtype=float)
    return None if np.isnan(m).any() else m


def _google_drive_block(origins, dests, key):
    r = requests.post(
        "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix",
        headers={"X-Goog-Api-Key": key,
                 "X-Goog-FieldMask": "originIndex,destinationIndex,distanceMeters,condition"},
        json={"origins": [_wp(*o) for o in origins],
              # sideOfRoad: arrive on the stop's side, so no U-turn to reach the far-side stop
              "destinations": [{"waypoint": {**_wp(*d)["waypoint"], "sideOfRoad": True}} for d in dests],
              "travelMode": "DRIVE"},
        timeout=30,
    )
    out = np.full((len(origins), len(dests)), np.nan)
    if r.ok:
        for el in r.json():
            if el.get("condition") == "ROUTE_EXISTS" and "distanceMeters" in el:
                out[el.get("originIndex", 0), el.get("destinationIndex", 0)] = el["distanceMeters"]
    return out


def driving_matrix(pts, gmaps_key="", road_factor=1.4):
    """Square road-distance matrix (meters, asymmetric). Google Routes (25x25
    blocks) → OSRM table → air distance x road_factor, filling gaps per cell."""
    pts = [tuple(map(float, p)) for p in pts]
    fallback = haversine_matrix(pts, pts) * road_factor
    m = np.full_like(fallback, np.nan)
    if gmaps_key:
        for i in range(0, len(pts), 25):
            for j in range(0, len(pts), 25):
                try:
                    m[i:i + 25, j:j + 25] = _google_drive_block(pts[i:i + 25], pts[j:j + 25], gmaps_key)
                except (requests.RequestException, ValueError):
                    pass
    if np.isnan(m).any() and len(pts) <= 100:
        osrm = _osrm_table(pts)
        if osrm is not None:
            m = np.where(np.isnan(m), osrm, m)
    m = np.where(np.isnan(m), fallback, m)
    np.fill_diagonal(m, 0)
    return m


# ---------------------------------------------------------------- stop selection

def _prune_stops(walk_mat, selected, max_walk_m):
    """Drop a stop when everyone it serves still has another selected stop within
    max_walk (or no farther than today, for people already over the limit).
    Least-used stops go first, so close-together stops merge into one."""
    selected = list(selected)
    changed = True
    while changed and len(selected) > 1:
        changed = False
        sub = walk_mat[:, selected]
        nearest = np.argmin(sub, axis=1)
        current = sub[np.arange(len(sub)), nearest]
        for j in np.argsort(np.bincount(nearest, minlength=len(selected)), kind="stable"):
            served = nearest == j
            others = [k for k in range(len(selected)) if k != j]
            alt = sub[served][:, others].min(axis=1) if served.any() else np.array([])
            if np.all(alt <= np.maximum(max_walk_m, current[served])):
                selected.pop(j)
                changed = True
                break
    return selected


def select_optimal_stops(passengers_df, bus_stops_df, walk_mat, max_walk_m):
    """Greedy set cover over walking distance, then each passenger goes to the
    nearest *selected* stop. Passengers with no stop within max_walk get their
    nearest stop and are flagged over_max_walk."""
    passengers_df = passengers_df.reset_index(drop=True).copy()
    if bus_stops_df is None or bus_stops_df.empty:
        bus_stops_df = pd.DataFrame({
            "stop_id": [f"V_{i}" for i in range(len(passengers_df))],
            "name": passengers_df["address"].values,
            "lat": passengers_df["lat"].values, "lon": passengers_df["lon"].values,
        })
        walk_mat = np.where(np.eye(len(passengers_df), dtype=bool), 0.0,
                            haversine_matrix(passengers_df[["lat", "lon"]].values,
                                             bus_stops_df[["lat", "lon"]].values) * WALK_DETOUR_FACTOR)
    walk_mat = np.asarray(walk_mat, float)
    within = walk_mat <= max_walk_m
    uncovered = np.ones(len(passengers_df), bool)
    selected = []
    while uncovered.any():
        cover = (within & uncovered[:, None]).sum(axis=0)
        if cover.max() > 0:
            best_count = cover.max()
            tied = np.where(cover == best_count)[0]
            sums = np.where(within[:, tied] & uncovered[:, None], walk_mat[:, tied], 0).sum(axis=0)
            s = int(tied[np.argmin(sums)])
            uncovered &= ~within[:, s]
        else:
            p = int(np.where(uncovered)[0][0])
            s = int(np.argmin(walk_mat[p]))
            uncovered[p] = False
        if s not in selected:
            selected.append(s)

    selected = _prune_stops(walk_mat, selected, max_walk_m)
    sub = walk_mat[:, selected]
    nearest = np.argmin(sub, axis=1)
    dists = sub[np.arange(len(sub)), nearest]
    stops = bus_stops_df.iloc[selected].reset_index(drop=True).copy()
    counts = np.bincount(nearest, minlength=len(selected))
    used = np.where(counts > 0)[0]
    remap = {old: new for new, old in enumerate(used)}
    stops = stops.iloc[used].reset_index(drop=True)
    stops["passengers_count"] = counts[used].astype(int)

    passengers_df["assigned_stop_idx"] = [remap[i] for i in nearest]
    passengers_df["assigned_stop_name"] = stops["name"].values[passengers_df["assigned_stop_idx"]]
    passengers_df["walk_distance_m"] = np.round(dists).astype(int)
    passengers_df["over_max_walk"] = dists > max_walk_m
    return stops, passengers_df


# ---------------------------------------------------------------- routing

def solve_vrp(stops, dest, origin, num_routes, demands, capacity=None, time_limit_s=3, dist_matrix=None):
    """Multi-vehicle routing: each bus starts at origin (or anywhere, if None),
    visits a subset of stops and ends at dest. Balances route lengths via a span
    cost so a single bus doesn't take everything. Returns a list of stop-index
    lists in visiting order (empty routes dropped)."""
    n = len(stops)
    if n == 0:
        return []
    k = max(1, min(int(num_routes), n))
    dest_i, start_i = n, n + 1
    pts = list(stops) + [dest] + [origin if origin else dest]
    # dist_matrix: road distances over [stops..., dest, start]; air distance otherwise.
    dist = np.round(np.asarray(dist_matrix, float) if dist_matrix is not None
                    else haversine_matrix(pts, pts)).astype(np.int64)
    if origin is None:
        dist[start_i, :] = 0  # free start next to the first stop
    dist[:, start_i] = 10**7  # never return to start
    dist[start_i, dest_i] = 10**7 if n >= k else 0  # don't let a bus skip all stops

    manager = pywrapcp.RoutingIndexManager(n + 2, k, [start_i] * k, [dest_i] * k)
    routing = pywrapcp.RoutingModel(manager)

    def cost(fi, ti):
        return int(dist[manager.IndexToNode(fi), manager.IndexToNode(ti)])

    cb = routing.RegisterTransitCallback(cost)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.AddDimension(cb, 0, 10**8, True, "Distance")
    if k > 1:
        routing.GetDimensionOrDie("Distance").SetGlobalSpanCostCoefficient(100)

    if capacity:
        dem = list(demands) + [0, 0]
        dcb = routing.RegisterUnaryTransitCallback(lambda i: int(dem[manager.IndexToNode(i)]))
        routing.AddDimensionWithVehicleCapacity(dcb, 0, [int(capacity)] * k, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s
    sol = routing.SolveWithParameters(params)
    if not sol:
        return None  # infeasible (e.g. not enough seats)
    routes = []
    for v in range(k):
        idx, r = routing.Start(v), []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node < n:
                r.append(node)
            idx = sol.Value(routing.NextVar(idx))
        if r:
            routes.append(r)
    return routes


def get_road_geometry(coords_list, gmaps_key=""):
    if len(coords_list) < 2:
        return list(coords_list)
    if gmaps_key and len(coords_list) <= 27:  # Routes API: ≤25 intermediates
        try:
            r = requests.post(
                "https://routes.googleapis.com/directions/v2:computeRoutes",
                headers={"X-Goog-Api-Key": gmaps_key, "X-Goog-FieldMask": "routes.polyline.encodedPolyline"},
                json={"origin": _wp(*coords_list[0])["waypoint"], "destination": _wp(*coords_list[-1])["waypoint"],
                      "intermediates": [{**_wp(*c)["waypoint"], "sideOfRoad": True} for c in coords_list[1:-1]],
                      "travelMode": "DRIVE"},
                timeout=15,
            )
            if r.ok and r.json().get("routes"):
                return decode_polyline(r.json()["routes"][0]["polyline"]["encodedPolyline"])
        except (requests.RequestException, ValueError, KeyError):
            pass
    locs = ";".join(f"{lon},{lat}" for lat, lon in coords_list)
    try:
        res = requests.get(f"https://router.project-osrm.org/route/v1/driving/{locs}",
                           params={"overview": "full", "geometries": "geojson"}, timeout=10).json()
        if res.get("code") == "Ok":
            return [(lat, lon) for lon, lat in res["routes"][0]["geometry"]["coordinates"]]
    except (requests.RequestException, ValueError):
        pass
    return list(coords_list)


# ---------------------------------------------------------------- misc

def search_passengers(df, query):
    q = query.strip()
    return df[df["name"].astype(str).str.contains(q, case=False, regex=False, na=False)
              | df["address"].astype(str).str.contains(q, case=False, regex=False, na=False)]
