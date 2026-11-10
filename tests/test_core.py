import math
import numpy as np
import pandas as pd
import pytest

import route_core as rc


# ---------- input parsing ----------

def test_build_passengers_keeps_name_address_alignment_when_names_missing():
    df = pd.DataFrame({"name": ["A", None, "C"], "addr": ["x 1", "y 2", "z 3"]})
    rows = rc.build_passenger_rows(df, "addr", "name")
    assert [(r["name"], r["address"]) for r in rows] == [("A", "x 1"), ("Passenger 2", "y 2"), ("C", "z 3")]


def test_build_passengers_skips_blank_addresses_without_shifting_names():
    df = pd.DataFrame({"name": ["A", "B", "C"], "addr": ["x 1", None, "z 3"]})
    rows = rc.build_passenger_rows(df, "addr", "name")
    assert [(r["name"], r["address"]) for r in rows] == [("A", "x 1"), ("C", "z 3")]


def test_read_table_handles_utf8_bom_hebrew_csv():
    raw = "שם,כתובת\nדני,הרצל 1 יבנה\n".encode("utf-8-sig")
    df = rc.read_table(raw, "Names.CSV")
    assert list(df.columns) == ["שם", "כתובת"]


def test_read_table_handles_cp1255_csv():
    raw = "שם,כתובת\nדני,הרצל 1\n".encode("cp1255")
    df = rc.read_table(raw, "a.csv")
    assert df.iloc[0, 0] == "דני"


# ---------- address / city ----------

def test_strip_zip_and_city_from_address():
    assert rc.clean_street("הצדף 18/4 מיקוד 8103821", "יבנה") == "הצדף 18"
    assert rc.clean_street("הצדף 18/4 יבנה", "יבנה") == "הצדף 18"
    assert rc.clean_street('ש"י עגנון 9/24', "יבנה") == 'ש"י עגנון 9'


def test_clean_street_drops_apartment_part():
    # "18/4" = building 18 apartment 4; geocoders want the building only
    assert rc.clean_street("הצדף 18/4", "") == "הצדף 18"




def test_point_in_bbox():
    assert rc.in_bbox(31.87, 34.74, (31.85, 34.70, 31.90, 34.78))
    assert not rc.in_bbox(31.80, 34.65, (31.85, 34.70, 31.90, 34.78))


# ---------- stops ----------

def test_merge_stops_dedupes_close_points():
    a = pd.DataFrame([{"stop_id": "o1", "name": "A", "lat": 31.0, "lon": 34.0}])
    b = pd.DataFrame([{"stop_id": "g1", "name": "A'", "lat": 31.00005, "lon": 34.0},
                      {"stop_id": "g2", "name": "B", "lat": 31.01, "lon": 34.0}])
    merged = rc.merge_stops([a, b], dedupe_m=20)
    assert sorted(merged["stop_id"]) == ["g2", "o1"]


def test_select_stops_uses_walk_matrix_not_air_distance():
    passengers = pd.DataFrame({"lat": [0.0], "lon": [0.0], "address": ["p"]})
    stops = pd.DataFrame({"stop_id": ["near_air", "near_walk"], "name": ["S1", "S2"],
                          "lat": [0.0, 0.0], "lon": [0.001, 0.002]})
    walk = np.array([[900.0, 300.0]])  # S1 is closer by air but across a fence
    sel, p = rc.select_optimal_stops(passengers, stops, walk, max_walk_m=400)
    assert list(sel["name"]) == ["S2"]
    assert p.loc[0, "walk_distance_m"] == 300
    assert not p.loc[0, "over_max_walk"]


def test_select_stops_flags_passenger_beyond_max_walk():
    passengers = pd.DataFrame({"lat": [0.0], "lon": [0.0], "address": ["p"]})
    stops = pd.DataFrame({"stop_id": ["s"], "name": ["S"], "lat": [0.0], "lon": [0.01]})
    sel, p = rc.select_optimal_stops(passengers, stops, np.array([[1200.0]]), max_walk_m=400)
    assert p.loc[0, "over_max_walk"]


def test_select_stops_greedy_covers_everyone():
    rng = np.random.default_rng(0)
    passengers = pd.DataFrame({"lat": rng.random(20), "lon": rng.random(20), "address": ["p"] * 20})
    stops = pd.DataFrame({"stop_id": [str(i) for i in range(8)], "name": [f"S{i}" for i in range(8)],
                          "lat": rng.random(8), "lon": rng.random(8)})
    walk = rng.random((20, 8)) * 1000
    sel, p = rc.select_optimal_stops(passengers, stops, walk, max_walk_m=500)
    assert p["assigned_stop_idx"].between(0, len(sel) - 1).all()
    assert sel["passengers_count"].sum() == 20


# ---------- routing ----------

def test_solve_vrp_visits_every_stop_once_and_ends_at_destination():
    stops = [(0.0, 0.01 * i) for i in range(1, 9)]
    dest = (0.0, 0.0)
    routes = rc.solve_vrp(stops, dest, origin=None, num_routes=2, demands=[1] * 8)
    visited = sorted(i for r in routes for i in r)
    assert visited == list(range(8))
    assert all(len(r) > 0 for r in routes)


def test_solve_vrp_respects_capacity():
    stops = [(0.0, 0.001 * i) for i in range(1, 7)]
    routes = rc.solve_vrp(stops, (0.0, 0.0), None, num_routes=2, demands=[10] * 6, capacity=30)
    assert all(sum(10 for _ in r) <= 30 for r in routes)


def test_solve_vrp_single_stop():
    assert rc.solve_vrp([(0.0, 0.01)], (0.0, 0.0), None, num_routes=3, demands=[1]) == [[0]]


# ---------- google walking matrix ----------





def test_search_is_literal_not_regex():
    df = pd.DataFrame({"name": ["a(b"], "address": ["x"]})
    assert len(rc.search_passengers(df, "a(")) == 1


def test_street_variants_cover_common_israeli_spellings():
    assert "שי עגנון 9" in rc.street_variants('ש"י עגנון 9')
    assert "עגנון 9" in rc.street_variants('ש"י עגנון 9')
    assert "שדרות דואני 10" in rc.street_variants("שד דואני 10")
    assert "יהודה הלוי 21" in rc.street_variants("רבי יהודה הלוי 21")
    assert rc.street_variants("הצדף 18")[0] == "הצדף 18"


# ---------- v2: driving matrix + stop pruning ----------

def test_select_stops_prunes_redundant_stop():
    # Greedy picks C first (covers p2,p3 with the smallest sum), then needs A for p0
    # and B for p1 — after which C is redundant: p2 can walk to A, p3 to B.
    passengers = pd.DataFrame({"lat": [0] * 4, "lon": [0] * 4, "address": list("abcd")})
    stops = pd.DataFrame({"stop_id": list("ABC"), "name": list("ABC"), "lat": [0] * 3, "lon": [0] * 3})
    walk = np.array([[100, 900, 900],
                     [900, 100, 900],
                     [300, 900, 10],
                     [900, 300, 10]], float)
    sel, p = rc.select_optimal_stops(passengers, stops, walk, max_walk_m=350)
    assert sorted(sel["name"]) == ["A", "B"]
    assert p["walk_distance_m"].max() <= 350


def test_solve_vrp_uses_given_distance_matrix():
    # Air distance says 0->1->2, but the road matrix makes 0->2->1 far cheaper.
    stops = [(0.0, 0.001), (0.0, 0.002), (0.0, 0.003)]
    n = 3
    big = 10_000
    m = np.full((n + 2, n + 2), big)
    np.fill_diagonal(m, 0)
    m[4, 0] = 1; m[0, 2] = 1; m[2, 1] = 1; m[1, 3] = 1  # start(4)->0->2->1->dest(3)
    routes = rc.solve_vrp(stops, (0.0, 0.0), (0.0, 0.004), 1, [1, 1, 1], dist_matrix=m)
    assert routes == [[0, 2, 1]]




# ---------- v3: walking paths + google tiles ----------

class _Resp:
    def __init__(self, payload, ok=True):
        self._p, self.ok = payload, ok

    def json(self):
        return self._p










def test_route_aware_swap_replaces_off_route_stop():
    # Route: S0 -> S1 -> dest. Kid k1 uses S1, which sits on a 1km side street;
    # candidate C (index 2) covers k1 equally well and lies on the main road.
    walk = np.array([[50, 900, 900],     # k0 -> S0
                     [900, 200, 250]],   # k1 -> S1 or C
                    float)
    # pool order: [S0, S1, C, dest]
    M = np.array([[0, 1000, 100, 1100],
                  [1000, 0, 1000, 1000],
                  [100, 1000, 0, 1000],
                  [0, 0, 0, 0]], float)
    routes = [[0, 1]]
    new_routes, changed = rc.improve_stops_for_route(routes, walk, M, dest_idx=3, max_walk_m=350)
    assert changed and new_routes == [[0, 2]]


def test_route_aware_swap_keeps_walk_limit():
    walk = np.array([[50, 900, 900], [900, 200, 600]], float)  # C would exceed 350m for k1
    M = np.array([[0, 1000, 100, 1100], [1000, 0, 1000, 1000], [100, 1000, 0, 1000], [0, 0, 0, 0]], float)
    new_routes, changed = rc.improve_stops_for_route([[0, 1]], walk, M, dest_idx=3, max_walk_m=350)
    assert not changed and new_routes == [[0, 1]]


# ---------- v4: headings / u-turns ----------

def test_bearing_cardinal():
    assert rc.bearing((0, 0), (0.01, 0)) == pytest.approx(0, abs=1)      # north
    assert rc.bearing((0, 0), (0, 0.01)) == pytest.approx(90, abs=1)     # east
    assert rc.bearing((0, 0), (-0.01, 0)) == pytest.approx(180, abs=1)


def test_count_uturns_detects_reversal_not_corners():
    straight = [(0, 0), (0.001, 0), (0.002, 0), (0.003, 0)]
    corner = [(0, 0), (0.001, 0), (0.001, 0.001), (0.001, 0.002)]          # 90° turn
    uturn = [(0, 0), (0.001, 0), (0.002, 0), (0.0021, 0.00003), (0.001, 0.00003), (0, 0.00003)]
    assert rc.count_uturns(straight) == 0
    assert rc.count_uturns(corner) == 0
    assert rc.count_uturns(uturn) == 1






def _circle(center, r_deg, start_deg, end_deg, step=15):
    out = []
    a = start_deg
    while (end_deg - a) * (1 if end_deg > start_deg else -1) >= 0:
        out.append((center[0] + r_deg * math.cos(math.radians(a)), center[1] + r_deg * math.sin(math.radians(a))))
        a += abs(step) if end_deg > start_deg else -abs(step)
    return out


def test_roundabout_left_turn_is_not_a_uturn():
    # approach northbound, go 3/4 around a 20m roundabout (counter-clockwise), exit westbound
    r = 0.00018
    path = [(-0.0015, 0), (-0.0008, 0), (-r, 0)] + _circle((0, 0), r, 180, -90, step=-15)[1:] + [(0, -0.0008), (0, -0.0015)]
    assert rc.count_uturns(path) == 0


def test_roundabout_full_circle_back_is_a_uturn():
    r = 0.00018
    path = [(-0.0015, 0), (-0.0008, 0), (-r, 0)] + _circle((0, 0), r, 180, -180, step=-15)[1:] + [(-0.0008, 0.00002), (-0.0015, 0.00002)]
    assert rc.count_uturns(path) == 1



# ---------- free routing (OSRM / FOSSGIS) ----------

def test_walking_matrix_only_queries_candidates_within_air_radius(monkeypatch):
    calls = []

    def fake(src, dst):
        calls.append((len(src), len(dst)))
        return np.full((len(src), len(dst)), 123.0)

    monkeypatch.setattr(rc, "_foot_table", fake)
    p = np.array([[0.0, 0.0]])
    s = np.array([[0.0, 0.001], [0.0, 0.5], [0.0, 0.6], [0.0, 0.7], [0.0, 0.8]])
    m = rc.walking_matrix(p, s, max_walk_m=400, min_candidates=1)
    assert m[0, 0] == 123.0
    assert m[0, 1] > 400  # never queried, stays as inflated air distance
    assert calls == [(1, 1)]


def test_walking_matrix_falls_back_when_router_down(monkeypatch):
    monkeypatch.setattr(rc, "_foot_table", lambda src, dst: None)
    p = np.array([[0.0, 0.0]])
    s = np.array([[0.0, 0.001]])
    air = rc.haversine_matrix(p, s)[0, 0]
    assert rc.walking_matrix(p, s, 400)[0, 0] == pytest.approx(air * rc.WALK_DETOUR_FACTOR)


def test_driving_matrix_sends_curb_and_bearings(monkeypatch):
    seen = {}

    def fake(bases, src, dst, extra=None):
        seen.update(extra)
        return np.ones((len(src), len(dst)))
    monkeypatch.setattr(rc, "_osrm_table", fake)
    m = rc.driving_matrix([(0, 0), (0, 0.01)], headings=[90, None])
    assert seen["approaches"] == ["curb"] * 4
    assert seen["bearings"] == ["90,45", "", "90,45", ""]
    assert m[0, 0] == 0 and m[0, 1] == 1


def test_driving_matrix_falls_back_to_scaled_air(monkeypatch):
    monkeypatch.setattr(rc, "_osrm_table", lambda *a, **k: None)
    pts = [(0.0, 0.0), (0.0, 0.01)]
    m = rc.driving_matrix(pts)
    assert m[0, 1] == pytest.approx(rc.haversine_matrix(pts, pts)[0, 1] * 1.4)


def test_osrm_table_parses_null_as_nan(monkeypatch):
    monkeypatch.setattr(rc, "_osrm_get", lambda *a, **k: {"code": "Ok", "distances": [[0, None]]})
    m = rc._osrm_table(["x"], [(0, 0)], [(0, 1), (0, 2)])
    assert m[0, 0] == 0 and np.isnan(m[0, 1])


def test_walking_path_reads_geojson(monkeypatch):
    monkeypatch.setattr(rc, "_osrm_get", lambda *a, **k: {"routes": [{"geometry": {"coordinates": [[34.7, 31.8], [34.71, 31.81]]}}]})
    assert rc.walking_path((31.8, 34.7), (31.81, 34.71)) == [(31.8, 34.7), (31.81, 34.71)]
    monkeypatch.setattr(rc, "_osrm_get", lambda *a, **k: None)
    assert rc.walking_path((0, 0), (0, 1)) is None


def test_stop_heading_from_last_leg(monkeypatch):
    sent = {}

    def fake(bases, service, coords, params, timeout=40):
        sent.update(params)
        return {"routes": [{"geometry": {"coordinates": [[0, 0], [0.001, 0], [0.002, 0]]}}]}  # eastbound
    monkeypatch.setattr(rc, "_osrm_get", fake)
    assert rc.stop_heading((0, 0.002)) == pytest.approx(90, abs=2)
    assert sent["approaches"] == "unrestricted;curb"


def test_no_google_left():
    import pathlib
    for f in ("route_core.py", "streamlit_app.py"):
        src = pathlib.Path(f).read_text()
        assert "googleapis.com" not in src, f


def test_osm_stops_disk_cache_survives_outage(monkeypatch, tmp_path):
    good = pd.DataFrame([{"stop_id": "osm:1", "name": "A (123)", "lat": 31.05, "lon": 34.05}])
    monkeypatch.setattr(rc, "_overpass_stops", lambda bbox, lang: good)
    bbox = (31.0, 34.0, 31.1, 34.1)
    assert len(rc.fetch_bus_stops_osm(bbox, cache_dir=tmp_path)) == 1
    monkeypatch.setattr(rc, "_overpass_stops", lambda bbox, lang: None)  # server down
    again = rc.fetch_bus_stops_osm(bbox, cache_dir=tmp_path)
    assert list(again["stop_id"]) == ["osm:1"]
    # a bbox inside a cached one is served from it too
    inner = rc.fetch_bus_stops_osm((31.01, 34.01, 31.09, 34.09), cache_dir=tmp_path)
    assert len(inner) == 1
