# route-planner

Streamlit app that picks bus stops for a list of home addresses and plans pickup routes.

## Free by design
No paid APIs and no keys: addresses via Nominatim, bus stops via OpenStreetMap
(Overpass), walking distances/paths via the FOSSGIS OSRM foot router, driving via
OSRM car routing (kerb-side approach, stop headings), base map via OSM/Carto tiles.
All are community servers — results are cached to keep load low.

## Settings (env vars or Streamlit secrets)
- `EDITOR_EMAILS` — comma-separated emails allowed to compute/edit maps.
- `[auth]` section in secrets — Google OIDC for `st.login` (required for editing when deployed).
- `MAPS_DIR` (default `data/maps`), `APP_URL` (base for share links). Local dev without auth: `ROUTE_PLANNER_DEV=1`.

Each computed map gets its own link (`?map=<id>`); its editor chooses viewer emails, `@domain`, or public.

## Tests
`pip install -r requirements.txt pytest && PYTHONPATH=. pytest`
