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


def test_city_matches_google_result_by_locality():
    result = {"address_components": [
        {"long_name": "Yavne", "short_name": "Yavne", "types": ["locality", "political"]}]}
    assert rc.google_result_in_city(result, ["יבנה", "Yavne"])
    assert not rc.google_result_in_city(result, ["אשדוד", "Ashdod"])


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

def test_walking_matrix_only_queries_candidates_within_air_radius(monkeypatch):
    calls = []

    def fake(origins, dests, key):
        calls.append((len(origins), len(dests)))
        return np.full((len(origins), len(dests)), 123.0)

    monkeypatch.setattr(rc, "_google_walk_block", fake)
    p = np.array([[0.0, 0.0]])
    s = np.array([[0.0, 0.001], [0.0, 0.5]])  # second stop ~55km away
    m = rc.walking_matrix(p, s, max_walk_m=400, gmaps_key="k")
    assert m[0, 0] == 123.0
    assert m[0, 1] > 400  # never queried, stays as inflated air distance
    assert calls == [(1, 1)]


def test_walking_matrix_without_key_uses_detour_factor():
    p = np.array([[0.0, 0.0]])
    s = np.array([[0.0, 0.001]])
    air = rc.haversine_matrix(p, s)[0, 0]
    m = rc.walking_matrix(p, s, 400, gmaps_key="")
    assert m[0, 0] == pytest.approx(air * rc.WALK_DETOUR_FACTOR)


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


def test_driving_matrix_falls_back_to_scaled_air(monkeypatch):
    monkeypatch.setattr(rc, "_osrm_table", lambda pts: None)
    pts = [(0.0, 0.0), (0.0, 0.01)]
    m = rc.driving_matrix(pts, gmaps_key="")
    assert m.shape == (2, 2) and m[0, 1] > rc.haversine_matrix(pts, pts)[0, 1]


# ---------- v3: walking paths + google tiles ----------

class _Resp:
    def __init__(self, payload, ok=True):
        self._p, self.ok = payload, ok

    def json(self):
        return self._p


def test_walking_path_decodes_google_polyline(monkeypatch):
    monkeypatch.setattr(rc.requests, "post", lambda *a, **k: _Resp(
        {"routes": [{"polyline": {"encodedPolyline": "_p~iF~ps|U_ulLnnqC"}}]}))
    path = rc.walking_path((38.5, -120.2), (40.7, -120.95), "k")
    assert path[0] == (38.5, -120.2) and path[-1] == (40.7, -120.95)


def test_walking_path_without_key_or_on_error_is_none(monkeypatch):
    assert rc.walking_path((0, 0), (0, 1), "") is None
    monkeypatch.setattr(rc.requests, "post", lambda *a, **k: _Resp({}, ok=False))
    assert rc.walking_path((0, 0), (0, 1), "k") is None


def test_google_tile_session(monkeypatch):
    monkeypatch.setattr(rc.requests, "post", lambda *a, **k: _Resp({"session": "S1"}))
    assert rc.google_tile_session("k", "he") == "S1"
    monkeypatch.setattr(rc.requests, "post", lambda *a, **k: _Resp({}, ok=False))
    assert rc.google_tile_session("k", "he") is None


def test_google_tile_session_sends_referer(monkeypatch):
    seen = {}

    def fake(*a, **k):
        seen.update(k.get("headers") or {})
        return _Resp({"session": "S"})
    monkeypatch.setattr(rc.requests, "post", fake)
    rc.google_tile_session("k", "he", referer="https://app.example/")
    assert seen.get("Referer") == "https://app.example/"
