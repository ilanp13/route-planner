import io
import os
import time

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import maps_store as ms
import route_core as rc
from i18n import TRANSLATIONS

st.set_page_config(page_title="Bus Route Optimizer | מתכנן קווי איסוף", layout="wide")

ROUTE_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
                "#0891b2", "#be185d", "#4d7c0f", "#a16207", "#475569"]


# ------------------------------------------------------------------ config

def get_setting(name, default=""):
    """Env var first (Streamlit Cloud 'env vars'), then st.secrets."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def auth_configured():
    try:
        return "auth" in st.secrets
    except Exception:
        return False


GMAPS_KEY = str(get_setting("GOOGLE_MAPS_API_KEY")).strip()
EDITORS = [e for e in ms.parse_emails(get_setting("EDITOR_EMAILS")) if e != ms.PUBLIC]
DEV_MODE = os.environ.get("ROUTE_PLANNER_DEV") == "1"
store = ms.MapStore(get_setting("MAPS_DIR", "data/maps"))

lang_choice = st.sidebar.radio("Language / שפה", ["עברית", "English"], horizontal=True)
lang_code = "he" if lang_choice == "עברית" else "en"
T = TRANSLATIONS[lang_code]

st.markdown(f"""<style>
.stApp, div[data-testid="stSidebarUserContent"] {{ direction: {T['dir']}; text-align: {T['align']}; }}
th, td {{ text-align: {T['align']} !important; }}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ identity

HAS_AUTH = auth_configured()
if HAS_AUTH and st.user.is_logged_in:
    user_email = (st.user.get("email") or "").lower()
elif DEV_MODE and not HAS_AUTH:
    user_email = "dev@local"
else:
    user_email = None
is_editor = bool(user_email) and (ms.email_allowed(user_email, EDITORS) or (DEV_MODE and not HAS_AUTH))

with st.sidebar:
    if user_email and HAS_AUTH:
        st.caption(f"{T['logged_as']} {user_email}")
        if st.button(T["logout_btn"]):
            st.logout()
    elif HAS_AUTH:
        if st.button(T["login_btn"], type="primary"):
            st.login()
    elif not DEV_MODE:
        st.caption(T["auth_missing"])

# ------------------------------------------------------------------ cached network calls

@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def cached_city(city, key, lang):
    return rc.geocode_city(city, key, lang)


@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def cached_geocode(address, city, key, lang, city_info):
    return rc.geocode_address(address, city, key, lang, city_info)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def cached_stops(bbox, key, lang):
    frames = []
    if key:
        frames.append(rc.fetch_bus_stops_google(bbox, key, lang))
    frames.append(rc.fetch_bus_stops_osm(bbox, lang))
    sources = [n for n, f in zip(["Google", "OSM"] if key else ["OSM"], frames) if not f.empty]
    if not sources:
        raise LookupError("no stops")  # exceptions aren't cached, so the next run retries
    return rc.merge_stops(frames), sources


def get_stops(bbox, key, lang):
    try:
        return cached_stops(bbox, key, lang)
    except LookupError:
        return pd.DataFrame(columns=rc.STOP_COLS), []


@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def cached_walk(p_coords, s_coords, max_walk, key):
    return rc.walking_matrix(p_coords, s_coords, max_walk, key)


# ------------------------------------------------------------------ computation

def compute(df_input, address_col, name_col, city, dest_address, origin_address,
            num_routes, capacity, max_walk):
    with st.status(T["status_city"], expanded=True) as status:
        city_info = cached_city(city.strip(), GMAPS_KEY, lang_code)
        if not city_info:
            st.error(f"{T['err_city']} {city}")
            return None

        def locate(addr):
            # Destination/origin may lie just outside the city (e.g. a regional school).
            return (cached_geocode(addr, city, GMAPS_KEY, lang_code, city_info)
                    or cached_geocode(f"{addr}, {city}" if "," not in addr else addr, "", GMAPS_KEY, lang_code, None))

        dest = locate(dest_address)
        if not dest:
            st.error(f"{T['err_dest']} {dest_address}")
            return None
        origin = locate(origin_address) if origin_address.strip() else None

        status.update(label=T["status_geocoding"])
        rows = rc.build_passenger_rows(df_input, address_col, name_col)
        unique = list(dict.fromkeys(r["address"] for r in rows))
        coords, failed = {}, []
        prog = st.progress(0.0)
        for i, addr in enumerate(unique):
            started = time.time()
            hit = cached_geocode(addr, city, GMAPS_KEY, lang_code, city_info)
            if hit:
                coords[addr] = (hit["lat"], hit["lon"])
            else:
                failed.append(addr)
            prog.progress((i + 1) / len(unique))
            # Nominatim policy: max 1 request/second. Cached hits return instantly.
            if not GMAPS_KEY and time.time() - started > 0.05:
                time.sleep(1.0)
        passengers = pd.DataFrame([{**r, "lat": coords[r["address"]][0], "lon": coords[r["address"]][1]}
                                   for r in rows if r["address"] in coords])
        if passengers.empty:
            st.error(T["err_no_valid"])
            return None

        status.update(label=T["status_stops"])
        bbox = rc.pad_bbox((passengers.lat.min(), passengers.lon.min(),
                            passengers.lat.max(), passengers.lon.max()), max_walk + 300)
        stops, stop_sources = get_stops(tuple(round(x, 5) for x in bbox), GMAPS_KEY, lang_code)

        status.update(label=T["status_walk"])
        p_xy = passengers[["lat", "lon"]].to_numpy()
        walk = cached_walk(p_xy, stops[["lat", "lon"]].to_numpy(), max_walk, GMAPS_KEY) if not stops.empty else None

        status.update(label=T["status_clustering"])
        sel, passengers = rc.select_optimal_stops(passengers, stops, walk, max_walk)

        status.update(label=T["status_routing"])
        stop_xy = list(zip(sel.lat, sel.lon))
        dest_xy = (dest["lat"], dest["lon"])
        origin_xy = (origin["lat"], origin["lon"]) if origin else None
        routes_idx = rc.solve_vrp(stop_xy, dest_xy, origin_xy, num_routes,
                                  sel["passengers_count"].tolist(), capacity or None)
        if routes_idx is None:
            st.error(T["err_capacity"])
            return None

        sel["route_id"] = 0
        routes, geometry = {}, {}
        for r_id, idxs in enumerate(routes_idx, start=1):
            sel.loc[idxs, "route_id"] = r_id
            nodes = []
            if origin_xy:
                nodes.append({"kind": "origin", "name": origin_address, "lat": origin_xy[0], "lon": origin_xy[1], "passengers": 0})
            nodes += [{"kind": "stop", "name": sel.at[i, "name"], "lat": sel.at[i, "lat"], "lon": sel.at[i, "lon"],
                       "passengers": int(sel.at[i, "passengers_count"])} for i in idxs]
            nodes.append({"kind": "dest", "name": dest_address, "lat": dest_xy[0], "lon": dest_xy[1], "passengers": 0})
            routes[r_id] = nodes
            geometry[r_id] = rc.get_road_geometry([(n["lat"], n["lon"]) for n in nodes], GMAPS_KEY)
        passengers["route_id"] = sel["route_id"].values[passengers["assigned_stop_idx"]]

        status.update(label=T["done"], state="complete", expanded=False)
    return {"passengers_df": passengers, "selected_stops": sel, "routes": routes, "routes_geometry": geometry,
            "dest_coords": dest_xy, "origin_coords": origin_xy, "failed_addrs": failed,
            "stop_sources": stop_sources, "city": city, "used_google": bool(GMAPS_KEY)}


# ------------------------------------------------------------------ rendering

def route_table(nodes):
    def label(n):
        if n["kind"] == "origin":
            return f"{T['origin_label']}: {n['name']}"
        if n["kind"] == "dest":
            return f"{T['dest_label']}: {n['name']}"
        return n["name"]
    return pd.DataFrame([{T["col_stop_order"]: i + 1, T["col_stop_name"]: label(n),
                          T["col_passengers_count"]: n["passengers"]} for i, n in enumerate(nodes)])


def color_for(r_num):
    return ROUTE_COLORS[(int(r_num) - 1) % len(ROUTE_COLORS)]


def render_results(res):
    p_df, s_df, routes, r_geom = res["passengers_df"], res["selected_stops"], res["routes"], res["routes_geometry"]

    st.markdown("---")
    q = st.text_input(T["search_label"], value="")
    if q.strip():
        matched = rc.search_passengers(p_df, q)
        if matched.empty:
            st.warning(T["search_none"])
        else:
            st.success(T["search_results_title"])
            for _, row in matched.iterrows():
                st.info(f"👤 **{row['name']}** | 🏠 {row['address']}  \n"
                        f"🚌 **{T['my_route']}** {T['route_name']} {int(row['route_id'])} | "
                        f"🚏 **{T['my_stop']}** {row['assigned_stop_name']} | "
                        f"🚶 **{T['my_walk']}** {row['walk_distance_m']} {T['meters']}", icon="📍")
    st.markdown("---")

    if res["failed_addrs"]:
        st.warning(f"{T['unassigned_warn']} {', '.join(res['failed_addrs'])}")
    over = p_df[p_df["over_max_walk"]]
    if not over.empty:
        st.warning(f"{T['over_walk_warn']} " + ", ".join(f"{r['name']} ({r['walk_distance_m']}m)" for _, r in over.iterrows()))
    if res.get("stop_sources") is not None:
        if res["stop_sources"]:
            st.caption(f"{T['stops_source']} {', '.join(res['stop_sources'])}")
        else:
            st.warning(T["virtual_stops"])

    st.subheader(T["map_title"])
    pts = p_df[["lat", "lon"]].values.tolist() + [list(res["dest_coords"])]
    m = folium.Map(control_scale=True)
    m.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)], [max(p[0] for p in pts), max(p[1] for p in pts)]])

    for r_num, geom in r_geom.items():
        color = color_for(r_num)
        folium.PolyLine(geom, color=color, weight=5, opacity=0.85, tooltip=f"{T['route_name']} {r_num}").add_to(m)
        for i, n in enumerate(routes[r_num]):
            if n["kind"] != "stop":
                continue
            folium.CircleMarker([n["lat"], n["lon"]], radius=9, color=color, fill=True, fill_color="#ffffff",
                                fill_opacity=1.0,
                                popup=folium.Popup(f"<b>{T['route_name']} {r_num} (#{i + 1})</b><br>{n['name']}<br>👥 {n['passengers']}",
                                                   max_width=250)).add_to(m)

    for _, p in p_df.iterrows():
        stop = s_df.loc[p["assigned_stop_idx"]]
        color = color_for(p["route_id"])
        folium.PolyLine([(p["lat"], p["lon"]), (stop["lat"], stop["lon"])], color=color, weight=2,
                        dash_array="4, 4", opacity=0.8).add_to(m)
        folium.CircleMarker([p["lat"], p["lon"]], radius=4, color="#f59e0b" if p["over_max_walk"] else color,
                            fill=True, fill_color=color, fill_opacity=0.9,
                            tooltip=f"{p['name']} - {p['address']} ({T['route_name']} {p['route_id']}, "
                                    f"{T['walk_label']}: {p['walk_distance_m']}m)").add_to(m)

    folium.Marker(res["dest_coords"], tooltip=T["dest_label"], icon=folium.Icon(color="black", icon="flag")).add_to(m)
    if res["origin_coords"]:
        folium.Marker(res["origin_coords"], tooltip=T["origin_label"], icon=folium.Icon(color="green", icon="play")).add_to(m)
    st_folium(m, width=None, height=540, returned_objects=[])

    tab1, tab2, tab3 = st.tabs([T["tab_outbound"], T["tab_inbound"], T["tab_passengers"]])
    for tab, reverse in ((tab1, False), (tab2, True)):
        with tab:
            cols = st.columns(min(len(routes), 4) or 1)
            for i, (r_num, nodes) in enumerate(routes.items()):
                with cols[i % len(cols)]:
                    st.markdown(f"**{T['route_name']} {r_num}**")
                    st.dataframe(route_table(nodes[::-1] if reverse else nodes), use_container_width=True, hide_index=True)

    display_p = pd.DataFrame({
        T["col_name"]: p_df["name"], T["col_address"]: p_df["address"],
        T["col_route"]: p_df["route_id"].map(lambda x: f"{T['route_name']} {x}"),
        T["col_assigned_stop"]: p_df["assigned_stop_name"], T["col_walk_dist"]: p_df["walk_distance_m"],
    })
    with tab3:
        st.dataframe(display_p, use_container_width=True, hide_index=True)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        for r_num, nodes in routes.items():
            route_table(nodes).to_excel(w, sheet_name=f"Route {r_num} - Outbound", index=False)
            route_table(nodes[::-1]).to_excel(w, sheet_name=f"Route {r_num} - Inbound", index=False)
        display_p.to_excel(w, sheet_name="Passengers Assignment", index=False)
    st.download_button(T["download_btn"], out.getvalue(), file_name="optimal_bus_routes.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)


def share_link(map_id):
    base = get_setting("APP_URL") or getattr(st.context, "url", "") or ""
    return f"{base.split('?')[0].rstrip('/')}/?map={map_id}"


def render_share_panel(map_id, meta):
    with st.expander(T["share_title"], expanded=False):
        st.code(share_link(map_id), language=None)
        viewers = [v for v in meta.get("viewers", []) if v != ms.PUBLIC]
        public = st.checkbox(T["public_link"], value=ms.PUBLIC in meta.get("viewers", []), key=f"pub_{map_id}")
        txt = st.text_area(T["viewers_input"], value="\n".join(viewers), key=f"vw_{map_id}")
        c1, c2 = st.columns(2)
        if c1.button(T["save_sharing"], key=f"save_{map_id}"):
            store.update_meta(map_id, viewers=ms.parse_emails(txt) + ([ms.PUBLIC] if public else []))
            st.success(T["saved"])
        if c2.button(T["delete_map"], key=f"del_{map_id}"):
            store.delete(map_id)
            st.query_params.clear()
            st.rerun()


# ------------------------------------------------------------------ page

st.title(T["title"])
st.write(T["subtitle"])

map_id = st.query_params.get("map")
run_request = None

with st.sidebar:
    st.header(T["settings"])
    app_mode = st.radio(T["mode_select"], [T["mode_viewer"], T["mode_admin"]]) if is_editor else T["mode_viewer"]

    if is_editor:
        mine = [m for m in store.list() if ms.can_view(m, user_email, EDITORS)]
        if mine:
            st.subheader(T["my_maps"])
            labels = {m["id"]: f"{m['title']} · {time.strftime('%d/%m/%y', time.localtime(m['created']))}" for m in mine}
            choice = st.selectbox(T["open_map"], [""] + list(labels), format_func=lambda i: labels.get(i, "—"),
                                  index=([""] + list(labels)).index(map_id) if map_id in labels else 0)
            if choice and choice != map_id:
                st.query_params["map"] = choice
                st.rerun()

    if app_mode == T["mode_admin"]:
        st.divider()
        if GMAPS_KEY:
            st.success(T["api_active"])
        else:
            st.info(T["api_fallback"])
        uploaded = st.file_uploader(T["upload"], type=["xlsx", "xls", "csv"])
        df_input = address_col = name_col = None
        if uploaded is not None:
            df_input = rc.read_table(uploaded.getvalue(), uploaded.name)
            cols = df_input.columns.tolist()
            guess = next((i for i, c in enumerate(cols) if "כתובת" in str(c) or "address" in str(c).lower()), 0)
            address_col = st.selectbox(T["select_col"], cols, index=guess)
            name_opts = [T["none_opt"]] + cols
            name_guess = next((i for i, c in enumerate(name_opts) if str(c).strip() in ("שם", "name", "Name")), 0)
            nc = st.selectbox(T["select_name_col"], name_opts, index=name_guess)
            name_col = None if nc == T["none_opt"] else nc
        city = st.text_input(T["city_hint"], value="")
        dest_address = st.text_input(T["dest_input"], value="")
        origin_address = st.text_input(T["origin_input"], value="")
        num_routes = st.number_input(T["num_routes"], min_value=1, max_value=10, value=2, step=1)
        capacity = st.number_input(T["capacity"], min_value=0, max_value=100, value=0, step=1)
        max_walk = st.slider(T["max_walk"], min_value=100, max_value=1000, value=350, step=50)
        title = st.text_input(T["map_title_input"], value=uploaded.name.rsplit(".", 1)[0] if uploaded else "")
        viewers_txt = st.text_area(T["viewers_input"], value="")
        public = st.checkbox(T["public_link"], value=False)
        st.caption(T["storage_note"])

        if st.button(T["run_btn"], type="primary", use_container_width=True):
            if df_input is None or not dest_address.strip() or not city.strip():
                st.error(T["err_no_file"])
            else:
                run_request = dict(df_input=df_input, address_col=address_col, name_col=name_col, city=city,
                                   dest_address=dest_address, origin_address=origin_address,
                                   num_routes=int(num_routes), capacity=int(capacity), max_walk=int(max_walk))
                run_meta = dict(title=title or city,
                                viewers=ms.parse_emails(viewers_txt) + ([ms.PUBLIC] if public else []))

if run_request:  # runs in the main area so progress is visible
    res = compute(**run_request)
    if res:
        st.query_params["map"] = store.save(res, owner=user_email, **run_meta)
        st.rerun()
    st.stop()

if map_id:
    meta = store.meta(map_id)
    if meta is None:
        st.error(T["map_not_found"])
    elif ms.can_view(meta, user_email, EDITORS):
        st.subheader(meta["title"])
        if is_editor:
            render_share_panel(map_id, meta)
        render_results(store.load(map_id))
    elif not user_email and HAS_AUTH:
        st.info(T["login_prompt"])
        if st.button(T["login_btn"], key="login_main", type="primary"):
            st.login()
    else:
        st.error(T["no_access"])
else:
    st.info(T["welcome"])
