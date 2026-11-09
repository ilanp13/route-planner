import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import io
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

st.set_page_config(page_title="Bus Route Optimizer | מתכנן קווי איסוף", layout="wide")

TRANSLATIONS = {
    "he": {
        "dir": "rtl",
        "align": "right",
        "title": "🚌 מערכת אופטימיזציית קווי איסוף ותחנות אוטובוס",
        "subtitle": "ניהול, תכנון ושיתוף קווי איסוף לתלמידים/נוסעים (הלוך וחזור).",
        "settings": "הגדרות מערכת",
        "mode_select": "בחר מצב תצוגה",
        "mode_admin": "🔧 מצב מנהל (העלאה וחישוב מחדש)",
        "mode_viewer": "👀 מצב צפייה ושיתוף (למשתמשים / הורים)",
        "search_label": "🔍 חפש לפי שם ילד או כתובת מגורים...",
        "search_results_title": "תוצאות חיפוש אישיות:",
        "my_route": "קו משויך:",
        "my_stop": "תחנת איסוף:",
        "my_walk": "מרחק הליכה מהבית:",
        "api_key_input": "Google Maps API Key (אופציונלי)",
        "api_active": "✅ מחובר ל-Google Maps API",
        "api_fallback": "ℹ️ עובד במצב חינמי (OSM/OSRM).",
        "upload": "העלה קובץ כתובות (Excel או CSV)",
        "select_col": "בחר עמודת כתובות מגורים",
        "select_name_col": "בחר עמודת שמות הילדים/נוסעים (אופציונלי)",
        "city_hint": "עיר ברירת מחדל",
        "dest_input": "כתובת יעד סופי (חובה)",
        "origin_input": "כתובת תחנת מוצא (אופציונלי)",
        "num_routes": "מספר קווי נסיעה",
        "max_walk": "מרחק הליכה מקסימלי מהבית לתחנה (מטרים)",
        "run_btn": "חשב מסלולים אופטימליים 🚀",
        "err_no_file": "אנא העלה קובץ כתובות והזן כתובת יעד סופי.",
        "status_geocoding": "ממיר כתובות לקואורדינטות...",
        "status_stops": "שולף תחנות אוטובוס...",
        "status_clustering": "מחלק לקווים באופן מאוזן ומשייך נוסעים...",
        "status_routing": "מחשב מסלולי נסיעה אופטימליים...",
        "done": "החישוב הסתיים בהצלחה!",
        "map_title": "מפת קווים, תחנות וקווי הליכה",
        "tab_outbound": "מסלולי הלוך",
        "tab_inbound": "מסלולי חזור",
        "tab_passengers": "טבלת שיוך מלאה",
        "route_name": "קו",
        "download_btn": "📥 הורד תוכנית קווים ושיוך נוסעים (Excel)",
        "col_stop_order": "סדר תחנה",
        "col_stop_name": "שם תחנה / מיקום",
        "col_passengers_count": "מספר נוסעים",
        "col_address": "כתובת מגורים",
        "col_name": "שם הילד/נוסע",
        "col_assigned_stop": "תחנה משויכת",
        "col_walk_dist": "מרחק הליכה (מ')",
        "col_route": "קו משויך",
        "origin_label": "תחנת מוצא",
        "dest_label": "תחנת יעד סופי",
        "unassigned_warn": "כתובות שלא זוהו על המפה:"
    },
    "en": {
        "dir": "ltr",
        "align": "left",
        "title": "🚌 Bus Stop & Route Optimization System",
        "subtitle": "Management and sharing portal for passenger/student pickup routes.",
        "settings": "System Settings",
        "mode_select": "Select View Mode",
        "mode_admin": "🔧 Admin Mode (Upload & Compute)",
        "mode_viewer": "👀 Viewer Mode (For Parents / Users)",
        "search_label": "🔍 Search by passenger name or address...",
        "search_results_title": "Personal Search Result:",
        "my_route": "Assigned Route:",
        "my_stop": "Assigned Bus Stop:",
        "my_walk": "Walking Distance from Home:",
        "api_key_input": "Google Maps API Key (Optional)",
        "api_active": "✅ Connected to Google Maps API",
        "api_fallback": "ℹ️ Using free OSM/OSRM mode.",
        "upload": "Upload Addresses File (Excel or CSV)",
        "select_col": "Select residential address column",
        "select_name_col": "Select passenger name column (Optional)",
        "city_hint": "Default City",
        "dest_input": "Final Destination Address (Required)",
        "origin_input": "Origin Station Address (Optional)",
        "num_routes": "Number of Routes",
        "max_walk": "Max walking distance to stop (meters)",
        "run_btn": "Calculate Optimal Routes 🚀",
        "err_no_file": "Please upload an address file and enter a destination address.",
        "status_geocoding": "Geocoding addresses...",
        "status_stops": "Fetching bus stops...",
        "status_clustering": "Balancing routes and assigning passengers...",
        "status_routing": "Computing optimal vehicle routes...",
        "done": "Calculation completed successfully!",
        "map_title": "Routes, Stops & Walking Paths Map",
        "tab_outbound": "Outbound Routes",
        "tab_inbound": "Inbound Routes",
        "tab_passengers": "Full Passenger Assignments",
        "route_name": "Route",
        "download_btn": "📥 Download Full Route Plan (Excel)",
        "col_stop_order": "Stop Order",
        "col_stop_name": "Stop Name / Location",
        "col_passengers_count": "Passengers Count",
        "col_address": "Residential Address",
        "col_name": "Passenger Name",
        "col_assigned_stop": "Assigned Bus Stop",
        "col_walk_dist": "Walking Distance (m)",
        "col_route": "Assigned Route",
        "origin_label": "Origin Station",
        "dest_label": "Final Destination",
        "unassigned_warn": "Addresses that could not be geocoded:"
    }
}

lang_choice = st.sidebar.radio("Language / שפה", ["עברית", "English"], horizontal=True)
lang_code = "he" if lang_choice == "עברית" else "en"
T = TRANSLATIONS[lang_code]

st.markdown(f"""
<style>
    .stApp {{
        direction: {T['dir']};
        text-align: {T['align']};
    }}
    div[data-testid="stSidebarUserContent"] {{
        direction: {T['dir']};
        text-align: {T['align']};
    }}
    th, td {{
        text-align: {T['align']} !important;
    }}
</style>
""", unsafe_allow_html=True)

secret_api_key = ""
try:
    secret_api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
except Exception:
    pass

def haversine_matrix(coords1, coords2):
    lat1, lon1 = np.radians(coords1[:, 0])[:, None], np.radians(coords1[:, 1])[:, None]
    lat2, lon2 = np.radians(coords2[:, 0])[None, :], np.radians(coords2[:, 1])[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 6371000.0 * 2.0 * np.arcsin(np.sqrt(a))

def decode_polyline(polyline_str):
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}
    while index < len(polyline_str):
        for unit in ['latitude', 'longitude']:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20:
                    break
            if result & 1:
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)
        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append((lat / 100000.0, lng / 100000.0))
    return coordinates

@st.cache_data(show_spinner=False)
def geocode_address(address, city_hint="", gmaps_key="", lang="he"):
    query = address.strip()
    if city_hint and city_hint.strip() not in query:
        query = f"{query}, {city_hint.strip()}"

    if gmaps_key:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": query, "key": gmaps_key, "language": lang, "region": "il"}
        try:
            res = requests.get(url, params=params, timeout=10).json()
            if res.get("status") == "OK" and res.get("results"):
                loc = res["results"][0]["geometry"]["location"]
                return float(loc["lat"]), float(loc["lng"])
        except Exception:
            pass

    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "BusRouteOptimizerApp/1.0"}
    params = {"q": query, "format": "json", "limit": 1, "accept-language": "he,en"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=10)
        data = res.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def fetch_bus_stops(min_lat, min_lon, max_lat, max_lon, lang="he"):
    pad = 0.015
    bbox = f"{min_lat - pad},{min_lon - pad},{max_lat + pad},{max_lon + pad}"
    query = f"""
    [out:json][timeout:25];
    (
      node["highway"="bus_stop"]({bbox});
      node["public_transport"="platform"]["bus"="yes"]({bbox});
    );
    out body;
    """
    try:
        res = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=30)
        elements = res.json().get("elements", [])
        stops = []
        for el in elements:
            tags = el.get("tags", {})
            name = (
                tags.get(f"name:{lang}")
                or tags.get("name:he")
                or tags.get("name:en")
                or tags.get("name")
                or f"Stop #{el['id']}"
            )
            stops.append({
                "stop_id": str(el["id"]),
                "name": name,
                "lat": float(el["lat"]),
                "lon": float(el["lon"])
            })
        return pd.DataFrame(stops).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def select_optimal_stops(passengers_df, bus_stops_df, max_walk_m):
    p_coords = passengers_df[["lat", "lon"]].values

    if bus_stops_df.empty:
        bus_stops_df = pd.DataFrame({
            "stop_id": [f"V_{i}" for i in range(len(passengers_df))],
            "name": passengers_df["address"].values,
            "lat": passengers_df["lat"].values,
            "lon": passengers_df["lon"].values
        })

    s_coords = bus_stops_df[["lat", "lon"]].values
    dist_mat = haversine_matrix(p_coords, s_coords)

    uncovered = set(range(len(passengers_df)))
    selected_stop_indices = []

    while uncovered:
        best_stop = None
        best_covered = set()
        best_dist_sum = float("inf")

        for s_idx in range(len(bus_stops_df)):
            covered_now = {p_idx for p_idx in uncovered if dist_mat[p_idx, s_idx] <= max_walk_m}
            if len(covered_now) > len(best_covered):
                best_covered = covered_now
                best_stop = s_idx
                best_dist_sum = sum(dist_mat[p_idx, s_idx] for p_idx in covered_now)
            elif len(covered_now) == len(best_covered) and len(covered_now) > 0:
                dist_sum = sum(dist_mat[p_idx, s_idx] for p_idx in covered_now)
                if dist_sum < best_dist_sum:
                    best_stop = s_idx
                    best_dist_sum = dist_sum

        if best_stop is not None and len(best_covered) > 0:
            selected_stop_indices.append(best_stop)
            uncovered -= best_covered
        else:
            p_idx = next(iter(uncovered))
            nearest_stop = int(np.argmin(dist_mat[p_idx, :]))
            if nearest_stop not in selected_stop_indices:
                selected_stop_indices.append(nearest_stop)
            uncovered.remove(p_idx)

    selected_stops = bus_stops_df.iloc[selected_stop_indices].copy().reset_index(drop=True)
    sel_coords = selected_stops[["lat", "lon"]].values
    final_dist_mat = haversine_matrix(p_coords, sel_coords)

    nearest_sel_idx = np.argmin(final_dist_mat, axis=1)
    min_dists = np.min(final_dist_mat, axis=1)

    passengers_df = passengers_df.copy()
    passengers_df["assigned_stop_idx"] = nearest_sel_idx
    passengers_df["assigned_stop_name"] = [selected_stops.loc[i, "name"] for i in nearest_sel_idx]
    passengers_df["walk_distance_m"] = np.round(min_dists).astype(int)

    counts = passengers_df["assigned_stop_idx"].value_counts()
    selected_stops["passengers_count"] = selected_stops.index.map(lambda i: int(counts.get(i, 0)))
    selected_stops = selected_stops[selected_stops["passengers_count"] > 0].copy()

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_stops.index)}
    selected_stops = selected_stops.reset_index(drop=True)
    passengers_df["assigned_stop_idx"] = passengers_df["assigned_stop_idx"].map(old_to_new)

    return selected_stops, passengers_df

def solve_tsp_open_or_fixed(coords_list, has_fixed_start):
    n = len(coords_list)
    if n <= 2:
        return list(range(n))

    coords_arr = np.array(coords_list)
    dist_mat = np.round(haversine_matrix(coords_arr, coords_arr)).astype(int)

    if has_fixed_start:
        manager = pywrapcp.RoutingIndexManager(n, 1, [0], [n - 1])
    else:
        dummy_mat = np.zeros((n + 1, n + 1), dtype=int)
        dummy_mat[:n, :n] = dist_mat
        dummy_mat[n, n - 1] = 10**9
        dist_mat = dummy_mat
        manager = pywrapcp.RoutingIndexManager(n + 1, 1, [n], [n - 1])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        f_node = manager.IndexToNode(from_index)
        t_node = manager.IndexToNode(to_index)
        return int(dist_mat[f_node][t_node])

    transit_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 2

    solution = routing.SolveWithParameters(search_params)
    if solution:
        order = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node < n:
                order.append(node)
            index = solution.Value(routing.NextVar(index))
        order.append(n - 1)
        return order
    return list(range(n))

def get_road_geometry(coords_list, gmaps_key=""):
    if len(coords_list) < 2:
        return coords_list

    if gmaps_key and len(coords_list) <= 25:
        origin = f"{coords_list[0][0]},{coords_list[0][1]}"
        destination = f"{coords_list[-1][0]},{coords_list[-1][1]}"
        waypoints = "|".join([f"{lat},{lon}" for lat, lon in coords_list[1:-1]])
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {"origin": origin, "destination": destination, "key": gmaps_key, "mode": "driving"}
        if waypoints:
            params["waypoints"] = waypoints
        try:
            res = requests.get(url, params=params, timeout=10).json()
            if res.get("status") == "OK" and res.get("routes"):
                poly = res["routes"][0]["overview_polyline"]["points"]
                return decode_polyline(poly)
        except Exception:
            pass

    locs = ";".join([f"{lon},{lat}" for lat, lon in coords_list])
    url = f"http://router.project-osrm.org/route/v1/driving/{locs}?overview=full&geometries=geojson"
    try:
        res = requests.get(url, timeout=10).json()
        if res.get("code") == "Ok":
            geojson_coords = res["routes"][0]["geometry"]["coordinates"]
            return [(lat, lon) for lon, lat in geojson_coords]
    except Exception:
        pass
    return coords_list

st.title(T["title"])
st.write(T["subtitle"])

with st.sidebar:
    st.header(T["settings"])
    app_mode = st.radio(T["mode_select"], [T["mode_viewer"], T["mode_admin"]])

    if app_mode == T["mode_admin"]:
        if secret_api_key:
            active_api_key = secret_api_key
            st.success(T["api_active"])
        else:
            user_key = st.text_input(T["api_key_input"], type="password")
            active_api_key = user_key.strip()
            if active_api_key:
                st.success(T["api_active"])
            else:
                st.info(T["api_fallback"])

        uploaded_file = st.file_uploader(T["upload"], type=["xlsx", "xls", "csv"])
        city_hint = st.text_input(T["city_hint"], value="")
        dest_address = st.text_input(T["dest_input"], value="")
        origin_address = st.text_input(T["origin_input"], value="")
        num_routes = st.number_input(T["num_routes"], min_value=1, max_value=10, value=2, step=1)
        max_walk = st.slider(T["max_walk"], min_value=100, max_value=1000, value=350, step=50)

df_input = None
address_col = None
name_col = None

if app_mode == T["mode_admin"] and uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)
    address_col = st.sidebar.selectbox(T["select_col"], options=df_input.columns.tolist())
    name_cols_opt = ["-- None --"] + df_input.columns.tolist()
    name_col_choice = st.sidebar.selectbox(T["select_name_col"], options=name_cols_opt)
    name_col = name_col_choice if name_col_choice != "-- None --" else None

if app_mode == T["mode_admin"] and st.sidebar.button(T["run_btn"], type="primary", use_container_width=True):
    if df_input is None or not dest_address.strip():
        st.error(T["err_no_file"])
    else:
        with st.status(T["status_geocoding"], expanded=True) as status:
            dest_coords = geocode_address(dest_address, city_hint, active_api_key, lang_code)
            if not dest_coords:
                st.error(f"Could not locate destination / לא ניתן לאתר את כתובת היעד: {dest_address}")
                st.stop()

            origin_coords = (
                geocode_address(origin_address, city_hint, active_api_key, lang_code)
                if origin_address.strip() else None
            )

            raw_addresses = df_input[address_col].dropna().astype(str).tolist()
            raw_names = (
                df_input[name_col].dropna().astype(str).tolist()
                if name_col else [f"Passenger {i+1}" for i in range(len(raw_addresses))]
            )

            unique_addrs = list(dict.fromkeys(raw_addresses))
            addr_to_coords = {}
            failed_addrs = []

            prog = st.progress(0)
            for i, addr in enumerate(unique_addrs):
                coords = geocode_address(addr, city_hint, active_api_key, lang_code)
                if coords:
                    addr_to_coords[addr] = coords
                else:
                    failed_addrs.append(addr)
                prog.progress((i + 1) / len(unique_addrs))
                if not active_api_key:
                    time.sleep(0.35)

            valid_rows = []
            for idx, addr in enumerate(raw_addresses):
                if addr in addr_to_coords:
                    valid_rows.append({
                        "name": raw_names[idx] if idx < len(raw_names) else f"Passenger {idx+1}",
                        "address": addr,
                        "lat": addr_to_coords[addr][0],
                        "lon": addr_to_coords[addr][1]
                    })
            passengers_df = pd.DataFrame(valid_rows)

            if passengers_df.empty:
                st.error("No valid residential addresses could be geocoded.")
                st.stop()

            status.update(label=T["status_stops"])
            all_lats = passengers_df["lat"].tolist() + [dest_coords[0]]
            all_lons = passengers_df["lon"].tolist() + [dest_coords[1]]
            bus_stops_df = fetch_bus_stops(min(all_lats), min(all_lons), max(all_lats), max(all_lons), lang_code)

            status.update(label=T["status_clustering"])
            selected_stops, passengers_df = select_optimal_stops(passengers_df, bus_stops_df, max_walk)

            status.update(label=T["status_routing"])
            actual_k = min(int(num_routes), len(selected_stops))
            
            # --- חלוקה מאוזנת בקפידה (Balanced Split) למניעת קו בודד ---
            if actual_k > 1 and len(selected_stops) >= actual_k:
                # מיון התחנות לפי קו רוחב/אורך כדי לחלק אותן שווה בשווה בצורה מרחבית או שוויונית
                sorted_indices = np.argsort(selected_stops["lat"].values)
                cluster_labels = np.zeros(len(selected_stops), dtype=int)
                chunk_size = int(np.ceil(len(selected_stops) / actual_k))
                for r_idx in range(actual_k):
                    start_i = r_idx * chunk_size
                    end_i = min((r_idx + 1) * chunk_size, len(selected_stops))
                    cluster_labels[sorted_indices[start_i:end_i]] = r_idx
                selected_stops["route_id"] = cluster_labels
            else:
                selected_stops["route_id"] = np.arange(len(selected_stops)) % actual_k

            stop_to_route = selected_stops["route_id"].to_dict()
            passengers_df["route_id"] = passengers_df["assigned_stop_idx"].map(stop_to_route) + 1
            selected_stops["route_id"] = selected_stops["route_id"] + 1

            routes_outbound = {}
            routes_inbound = {}
            routes_geometry = {}

            for r_id in range(1, actual_k + 1):
                r_stops = selected_stops[selected_stops["route_id"] == r_id].copy().reset_index(drop=True)
                nodes = []
                if origin_coords:
                    nodes.append({
                        "name": f"{T['origin_label']}: {origin_address}",
                        "lat": origin_coords[0],
                        "lon": origin_coords[1],
                        "passengers": 0
                    })
                for _, row in r_stops.iterrows():
                    nodes.append({
                        "name": row["name"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "passengers": int(row["passengers_count"])
                    })
                nodes.append({
                    "name": f"{T['dest_label']}: {dest_address}",
                    "lat": dest_coords[0],
                    "lon": dest_coords[1],
                    "passengers": 0
                })

                coords_list = [(n["lat"], n["lon"]) for n in nodes]
                ordered_indices = solve_tsp_open_or_fixed(coords_list, has_fixed_start=(origin_coords is not None))
                ordered_nodes = [nodes[idx] for idx in ordered_indices]

                routes_outbound[r_id] = pd.DataFrame([
                    {
                        T["col_stop_order"]: idx + 1,
                        T["col_stop_name"]: n["name"],
                        T["col_passengers_count"]: n["passengers"],
                        "lat": n["lat"],
                        "lon": n["lon"]
                    }
                    for idx, n in enumerate(ordered_nodes)
                ])

                routes_inbound[r_id] = pd.DataFrame([
                    {
                        T["col_stop_order"]: idx + 1,
                        T["col_stop_name"]: n["name"],
                        T["col_passengers_count"]: n["passengers"],
                        "lat": n["lat"],
                        "lon": n["lon"]
                    }
                    for idx, n in enumerate(reversed(ordered_nodes))
                ])

                ordered_coords = [(n["lat"], n["lon"]) for n in ordered_nodes]
                routes_geometry[r_id] = get_road_geometry(ordered_coords, active_api_key)

            st.session_state["results"] = {
                "passengers_df": passengers_df,
                "selected_stops": selected_stops,
                "routes_outbound": routes_outbound,
                "routes_inbound": routes_inbound,
                "routes_geometry": routes_geometry,
                "dest_coords": dest_coords,
                "origin_coords": origin_coords,
                "failed_addrs": failed_addrs
            }
            status.update(label=T["done"], state="complete", expanded=False)

if "results" in st.session_state:
    res = st.session_state["results"]
    p_df = res["passengers_df"]
    s_df = res["selected_stops"]
    r_out = res["routes_outbound"]
    r_in = res["routes_inbound"]
    r_geom = res["routes_geometry"]

    # --- אזור חיפוש אישי למשתמשים/הורים ---
    st.markdown("---")
    search_query = st.text_input(T["search_label"], value="")
    if search_query.strip():
        matched = p_df[
            p_df["name"].str.contains(search_query, case=False, na=False) |
            p_df["address"].str.contains(search_query, case=False, na=False)
        ]
        if not matched.empty:
            st.success(T["search_results_title"])
            for _, row in matched.iterrows():
                r_num = int(row["route_id"])
                st.info(
                    f"👤 **{row['name']}** | 🏠 {row['address']} <br>"
                    f"🚌 **{T['my_route']}** קו {r_num} | "
                    f"🚏 **{T['my_stop']}** {row['assigned_stop_name']} | "
                    f"🚶‍♂️ **{T['my_walk']}** {row['walk_distance_m']} מטר",
                    icon="📍"
                )
        else:
            st.warning("לא נמצאו תוצאות תואמות לחיפוש.")
    st.markdown("---")

    if res["failed_addrs"]:
        st.warning(f"{T['unassigned_warn']} {', '.join(res['failed_addrs'])}")

    st.subheader(T["map_title"])
    center_lat = p_df["lat"].mean()
    center_lon = p_df["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

    route_colors = {1: "#2563eb", 2: "#dc2626", 3: "#16a34a", 4: "#9333ea", 5: "#ea580c", 6: "#0891b2"}

    for r_num, geom in r_geom.items():
        color = route_colors.get(r_num, "#334155")
        folium.PolyLine(
            geom, color=color, weight=5, opacity=0.85,
            tooltip=f"{T['route_name']} {r_num}"
        ).add_to(m)

        df_route = r_out[r_num]
        for _, stop_row in df_route.iterrows():
            folium.CircleMarker(
                location=[stop_row["lat"], stop_row["lon"]],
                radius=9,
                color=color,
                fill=True,
                fill_color="#ffffff",
                fill_opacity=1.0,
                popup=f"<b>{T['route_name']} {r_num} (#{stop_row[T['col_stop_order']]})</b><br>{stop_row[T['col_stop_name']]}<br>👥 {stop_row[T['col_passengers_count']]}"
            ).add_to(m)

    for _, p_row in p_df.iterrows():
        stop_row = s_df.loc[p_row["assigned_stop_idx"]]
        r_num = int(stop_row["route_id"])
        line_color = route_colors.get(r_num, "#64748b")

        folium.PolyLine(
            [(p_row["lat"], p_row["lon"]), (stop_row["lat"], stop_row["lon"])],
            color=line_color, weight=2, dash_array="4, 4", opacity=0.8
        ).add_to(m)
        
        folium.CircleMarker(
            location=[p_row["lat"], p_row["lon"]],
            radius=4, color=line_color, fill=True, fill_color=line_color, fill_opacity=0.9,
            tooltip=f"{p_row['name']} - {p_row['address']} ({T['route_name']} {r_num}, הליכה: {p_row['walk_distance_m']}m)"
        ).add_to(m)

    folium.Marker(
        location=res["dest_coords"],
        tooltip=T["dest_label"],
        icon=folium.Icon(color="black", icon="flag")
    ).add_to(m)

    if res["origin_coords"]:
        folium.Marker(
            location=res["origin_coords"],
            tooltip=T["origin_label"],
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)

    st_folium(m, width=None, height=540, returned_objects=[])

    tab1, tab2, tab3 = st.tabs([T["tab_outbound"], T["tab_inbound"], T["tab_passengers"]])

    with tab1:
        cols = st.columns(len(r_out))
        for idx, (r_num, df_r) in enumerate(r_out.items()):
            with cols[idx]:
                st.markdown(f"**{T['route_name']} {r_num}**")
                st.dataframe(df_r.drop(columns=["lat", "lon"]), use_container_width=True, hide_index=True)

    with tab2:
        cols = st.columns(len(r_in))
        for idx, (r_num, df_r) in enumerate(r_in.items()):
            with cols[idx]:
                st.markdown(f"**{T['route_name']} {r_num}**")
                st.dataframe(df_r.drop(columns=["lat", "lon"]), use_container_width=True, hide_index=True)

    with tab3:
        display_p_df = pd.DataFrame({
            T["col_name"]: p_df["name"],
            T["col_address"]: p_df["address"],
            T["col_route"]: p_df["route_id"].map(lambda x: f"{T['route_name']} {x}"),
            T["col_assigned_stop"]: p_df["assigned_stop_name"],
            T["col_walk_dist"]: p_df["walk_distance_m"]
        })
        st.dataframe(display_p_df, use_container_width=True, hide_index=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for r_num in r_out:
            r_out[r_num].drop(columns=["lat", "lon"]).to_excel(
                writer, sheet_name=f"Route {r_num} - Outbound", index=False
            )
            r_in[r_num].drop(columns=["lat", "lon"]).to_excel(
                writer, sheet_name=f"Route {r_num} - Inbound", index=False
            )
        display_p_df.to_excel(writer, sheet_name="Passengers Assignment", index=False)

    st.download_button(
        label=T["download_btn"],
        data=output.getvalue(),
        file_name="optimal_bus_routes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
else:
    if app_mode == T["mode_viewer"]:
        st.info("👋 ברוכים הבאים! האפליקציה במצב צפייה. אם טרם חושבו מסלולים, המנהל צריך להעלות את קובץ הכתובות פעם אחת במצב מנהל.")