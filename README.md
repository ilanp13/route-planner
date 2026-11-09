# route-planner

Streamlit app that picks bus stops for a list of home addresses and plans pickup routes.

## Settings (env vars or Streamlit secrets)
- `GOOGLE_MAPS_API_KEY` — enables Google geocoding, Places (New) bus stops and Routes API walking distances. Without it: OSM/Nominatim/Overpass with estimated walking.
- `EDITOR_EMAILS` — comma-separated emails allowed to compute/edit maps.
- `[auth]` section in secrets — Google OIDC for `st.login` (required for editing when deployed).
- `MAPS_DIR` (default `data/maps`), `APP_URL` (base for share links). Local dev without auth: `ROUTE_PLANNER_DEV=1`.

Each computed map gets its own link (`?map=<id>`); its editor chooses viewer emails, `@domain`, or public.

## Tests
`pip install -r requirements.txt pytest && PYTHONPATH=. pytest`
