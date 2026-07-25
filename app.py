from flask import Flask, request, jsonify
import requests
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import base64
import time
import json
import os
import tempfile
import zipfile
from io import BytesIO, StringIO

# V2 dependencies
import geopandas as gpd
import pandas as pd
from shapely.geometry import (
    Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
)

app = Flask(__name__)


# ═════════════════════════════════════════════════════════════════════
# CORS — allow the browser chat UI (served from a file or Netlify) to call
# this backend directly (e.g. GET /handle/<id> to draw geometry on the map).
# Without these headers the browser blocks cross-origin requests.
# ═════════════════════════════════════════════════════════════════════
@app.after_request
def _add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp


@app.route('/<path:_any>', methods=['OPTIONS'])
def _cors_preflight(_any):
    # Respond to browser preflight checks for any route.
    return ('', 204)

# ═════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════
CATEGORY_MAP = {
    'hospital':     [
        ('amenity', 'hospital'),
        ('healthcare', 'hospital'),
        ('amenity', 'clinic'),
        ('healthcare', 'clinic'),
    ],
    'pharmacy':     [
        ('amenity', 'pharmacy'),
        ('healthcare', 'pharmacy'),
        ('shop', 'chemist'),
    ],
    'school':       [('amenity', 'school')],
    'university':   [('amenity', 'university')],
    'library':      [('amenity', 'library')],
    'restaurant':   [('amenity', 'restaurant')],
    'cafe':         [('amenity', 'cafe')],
    'bank':         [('amenity', 'bank')],
    'atm':          [('amenity', 'atm')],
    'police':       [('amenity', 'police')],
    'fire_station': [('amenity', 'fire_station')],
    'gas_station':  [('amenity', 'fuel')],
    'parking':      [('amenity', 'parking')],
    'supermarket':  [('shop', 'supermarket')],
    'bakery':       [('shop', 'bakery')],
    'park':         [('leisure', 'park')],
    'playground':   [('leisure', 'playground')],
    'hotel':        [('tourism', 'hotel')],
    'museum':       [('tourism', 'museum')],
}

STYLE_MAP = {
    'hospital':     '#e74c3c',
    'pharmacy':     '#8e44ad',
    'school':       '#2980b9',
    'university':   '#2c3e50',
    'library':      '#34495e',
    'restaurant':   '#e67e22',
    'cafe':         '#d35400',
    'bank':         '#27ae60',
    'atm':          '#16a085',
    'police':       '#2c3e50',
    'fire_station': '#c0392b',
    'gas_station':  '#f39c12',
    'parking':      '#7f8c8d',
    'supermarket':  '#16a085',
    'bakery':       '#d35400',
    'park':         '#27ae60',
    'playground':   '#2ecc71',
    'hotel':        '#2980b9',
    'museum':       '#9b59b6',
}
DEFAULT_COLOR = '#3498db'

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Mirrors tried in order. The public servers get overloaded and return 504s;
# falling back across mirrors + retrying makes fetches far more reliable.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# OSRM routing engine (drive-time / isochrone tool, V3.7c).
# Public demo server — fine for development; for production self-host OSRM
# (Docker) or use a paid routing provider. Profile is 'driving'.
OSRM_BASE = "https://router.project-osrm.org"

PDF_MAX_ROWS = 100

MAX_FILE_SIZE_MB = 5
USER_LAYER_COLOR = '#1f6feb'
USER_LAYER_FILL  = '#3b82f6'


# ═════════════════════════════════════════════════════════════════════
# HANDLE STORE (V3.7f — agent session layer)
#
# Lets the agent compose tools without passing big GeoJSON through the LLM.
# A tool can store its result here and return a short handle ("layer_3")
# plus a one-line summary; the next tool reads the geometry back by handle.
#
# In-memory with a TTL. Fine for development/demo. For production this is
# the ONE place to swap for Redis / durable storage — the rest of the code
# only touches HANDLES.put() / .get() / .summary(), so the swap is local.
# ═════════════════════════════════════════════════════════════════════
import threading as _threading


class _HandleStore:
    def __init__(self, ttl_seconds=3600, max_handles=300):
        self._store = {}
        self._ttl = ttl_seconds
        self._max = max_handles
        self._counter = 0
        self._lock = _threading.Lock()

    def _evict_expired(self):
        now = time.time()
        dead = [k for k, v in self._store.items() if now - v['ts'] > self._ttl]
        for k in dead:
            del self._store[k]

    def put(self, geojson, summary):
        with self._lock:
            self._evict_expired()
            if len(self._store) >= self._max:
                oldest = min(self._store, key=lambda k: self._store[k]['ts'])
                del self._store[oldest]
            self._counter += 1
            hid = f"layer_{self._counter}"
            self._store[hid] = {'geojson': geojson, 'summary': summary,
                                'ts': time.time()}
            return hid

    def get(self, hid):
        with self._lock:
            self._evict_expired()
            rec = self._store.get(hid)
            return rec['geojson'] if rec else None

    def summary(self, hid):
        with self._lock:
            rec = self._store.get(hid)
            return rec['summary'] if rec else None


HANDLES = _HandleStore()


def _is_handle(x):
    """True if x looks like a stored-layer handle rather than inline GeoJSON."""
    return isinstance(x, str) and x.startswith('layer_')


def _resolve_geojson_input(value):
    """Accept EITHER a handle string ('layer_3') OR an inline GeoJSON dict,
    and return the GeoJSON dict. Raises ValueError if a handle is unknown
    (e.g. expired). This is what lets every tool work in both modes."""
    if _is_handle(value):
        gj = HANDLES.get(value)
        if gj is None:
            raise ValueError(f"unknown or expired handle: {value}")
        return gj
    return value


def _summarize_geojson(gj):
    """One-line human summary of a FeatureCollection for the agent."""
    try:
        feats = gj.get('features', []) if isinstance(gj, dict) else []
        n = len(feats)
        kinds = {}
        for f in feats:
            g = (f.get('geometry') or {}).get('type', '?')
            kinds[g] = kinds.get(g, 0) + 1
        if not kinds:
            return "0 features"
        parts = ", ".join(f"{v} {k}" for k, v in kinds.items())
        return f"{n} feature(s): {parts}"
    except Exception:
        return "feature collection"


def _wants_lean():
    """True if the caller wants a LEAN response (handle + summary + stats,
    but NOT the heavy geojson coordinate dump). The agent should use this so
    it never pulls full geometry into the LLM context — it only needs the
    handle. Triggered by JSON body {"lean": true} or query ?lean=1.
    Defaults to TRUE for /tool/* calls unless explicitly disabled, because
    the agent is the primary caller and lean saves ~90% of tokens."""
    # explicit query param wins
    qp = request.args.get('lean')
    if qp is not None:
        return str(qp).lower() in ('1', 'true', 'yes')
    # then JSON body flag
    body = request.get_json(silent=True) or {}
    if 'lean' in body:
        return str(body.get('lean')).lower() in ('1', 'true', 'yes')
    # default: lean ON for tool endpoints (agent is the main caller)
    return True


def _properties_digest(gj, max_items=60):
    """A compact, coordinate-free digest of a FeatureCollection's properties,
    so the agent can still name/describe features in lean mode without the
    huge coordinate arrays. Returns a list of small dicts (name + key props)."""
    out = []
    try:
        feats = gj.get('features', []) if isinstance(gj, dict) else []
        for f in feats[:max_items]:
            props = f.get('properties') or {}
            item = {}
            # prefer a name-like field
            for k in ('name', 'Name', 'NAME', 'name_en'):
                if props.get(k):
                    item['name'] = props[k]
                    break
            if 'name' not in item:
                item['name'] = props.get('name', 'Unnamed')
            # carry small, useful status/category fields if present
            for k in ('join_status', 'polygon_name', 'polygon_index',
                      'type', 'within'):
                if k in props and props[k] is not None:
                    item[k] = props[k]
            out.append(item)
    except Exception as e:
        print(f"_properties_digest error: {e}")
    return out


def _attach_handle(resp_dict, summary_override=None):
    """Store the response's 'geojson' under a handle and add 'handle' +
    'summary'. In LEAN mode (the default for tool calls), the heavy 'geojson'
    is REMOVED and replaced with a compact coordinate-free 'features' digest
    (names + status), so the agent can still describe results without pulling
    megabytes of coordinates into the LLM context. Full geometry stays in the
    handle store, retrievable via /handle/<id> or by passing {"lean": false}."""
    gj = resp_dict.get('geojson')
    if gj is None:
        return resp_dict
    summary = summary_override or _summarize_geojson(gj)
    hid = HANDLES.put(gj, summary)
    resp_dict['handle'] = hid
    resp_dict['summary'] = summary
    if _wants_lean():
        digest = _properties_digest(gj)
        resp_dict.pop('geojson', None)
        resp_dict['features'] = digest          # compact, names + status only
        resp_dict['geojson_in_handle'] = hid    # full geometry under this handle
        n_total = len(gj.get('features', [])) if isinstance(gj, dict) else 0
        if n_total > len(digest):
            resp_dict['features_truncated'] = True
            resp_dict['features_shown'] = len(digest)
            resp_dict['features_total'] = n_total
    return resp_dict


def _store_subset_handle(gj, predicate, summary):
    """Store ONLY the features matching predicate(props) as a new handle, and
    return (handle_id, count). Used to give callers a clean 'inside-only' (or
    any subset) layer so the map can draw exactly those features. Returns
    (None, 0) if nothing matches."""
    try:
        feats = gj.get('features', []) if isinstance(gj, dict) else []
        kept = [f for f in feats if predicate(f.get('properties') or {})]
        if not kept:
            return None, 0
        sub = {'type': 'FeatureCollection', 'features': kept}
        hid = HANDLES.put(sub, summary)
        return hid, len(kept)
    except Exception as e:
        print(f"_store_subset_handle error: {e}")
        return None, 0


def _overpass_post(query, label="overpass", attempts_per_endpoint=2,
                   timeout=45):
    """POST an Overpass QL query with retries across multiple mirrors.

    Returns (data_dict, True) on success, (None, False) if every mirror and
    retry failed. Distinguishing these lets callers tell a real fetch failure
    apart from a genuinely empty result — so we never present a timeout as
    "0 features found".
    """
    headers = {
        'User-Agent': 'gis_agent_v3/1.0 (GIS analysis tool)',
        'Accept':     'application/json',
    }
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(attempts_per_endpoint):
            try:
                resp = requests.post(endpoint, data={'data': query},
                                     headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    try:
                        return resp.json(), True
                    except ValueError:
                        print(f"{label}: 200 but invalid JSON from {endpoint}")
                        break  # bad response from this mirror, try next
                elif resp.status_code in (429, 502, 503, 504):
                    wait = 2 * (attempt + 1)
                    print(f"{label}: HTTP {resp.status_code} from {endpoint} "
                          f"(attempt {attempt + 1}/{attempts_per_endpoint}); "
                          f"waiting {wait}s")
                    time.sleep(wait)
                    continue  # retry same mirror
                else:
                    print(f"{label}: HTTP {resp.status_code} from {endpoint}: "
                          f"{resp.text[:150]}")
                    break  # non-retryable; move to next mirror
            except requests.exceptions.Timeout:
                print(f"{label}: timeout from {endpoint} "
                      f"(attempt {attempt + 1}/{attempts_per_endpoint})")
                time.sleep(2)
                continue
            except Exception as e:
                print(f"{label}: error from {endpoint}: {e}")
                break  # move to next mirror
    print(f"{label}: ALL mirrors/retries failed")
    return None, False


# V2.5 — branding + cartography
LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'kl_geospatial_logo_pdf.png',
)
BRAND_NAME    = "K&L Geospatial"
BRAND_NAVY    = (26, 26, 46)
BRAND_ACCENT  = (31, 111, 235)
BRAND_GRAY    = (107, 114, 128)
STATIC_MAP_W  = 10        # inches at 150 dpi
STATIC_MAP_H  = 7
STATIC_MAP_DPI = 150


# ═════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════
def _safe_str(val, max_len=120):
    """Stringify safely for popups. None / NaN → '', truncate huge values."""
    if val is None:
        return ''
    try:
        if pd.isna(val):
            return ''
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() == 'nan':
        return ''
    # Strip any HTML to keep popups simple and safe
    if '<' in s and '>' in s:
        import re
        s = re.sub(r'<[^>]+>', '', s)
        s = s.strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip() + '…'
    return s


def _extract_kml_attrs_from_description(desc_html):
    """
    ArcGIS-exported KML encodes the original attribute table inside the
    <description> HTML as a two-column table (label, value). Parse it back
    into {field: value} pairs so popups can show real attributes instead of
    a wall of HTML. Returns {} if nothing recognizable is found.
    """
    if not desc_html or not isinstance(desc_html, str):
        return {}
    try:
        import re
        rows = re.findall(
            r'<tr[^>]*>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>\s*</tr>',
            desc_html, flags=re.IGNORECASE | re.DOTALL,
        )
        attrs = {}
        for raw_key, raw_val in rows:
            key = re.sub(r'<[^>]+>', '', raw_key).strip()
            val = re.sub(r'<[^>]+>', '', raw_val).strip()
            if not key or not val or key == val:
                continue
            if key not in attrs:
                attrs[key] = val
        return attrs
    except Exception as e:
        print(f"KML description parse error: {e}")
        return {}


# ═════════════════════════════════════════
# STEP 1 — Geocode (V3.5: country-biased + candidate listing)
# ═════════════════════════════════════════
def geocode_location(location_str, retries=2, country_code=None,
                     return_candidates=False):
    """Geocode a free-text place name.

    - country_code: ISO-2 code (e.g. 'sa', 'lb') to bias results to one
      country. Dramatically reduces wrong-region matches for ambiguous
      Saudi/Lebanese place names.
    - return_candidates: if True, returns a list of up to 5 candidate
      dicts [{lat, lon, address}] instead of a single tuple, so callers
      can disambiguate ("did you mean X or Y?").

    Default behavior (no kwargs) is unchanged: returns (lat, lon, address)
    or (None, None, None). Backward-compatible with all existing callers.
    """
    geolocator = Nominatim(user_agent="gis_agent_v3/1.0")
    for attempt in range(retries + 1):
        try:
            if return_candidates:
                locs = geolocator.geocode(
                    location_str, timeout=10, language='en',
                    exactly_one=False, limit=5, addressdetails=True,
                    country_codes=country_code,
                )
                if not locs:
                    return []
                return [
                    {'lat': l.latitude, 'lon': l.longitude, 'address': l.address}
                    for l in locs
                ]
            else:
                loc = geolocator.geocode(
                    location_str, timeout=10, language='en',
                    country_codes=country_code,
                )
                if loc:
                    return loc.latitude, loc.longitude, loc.address
                return None, None, None
        except (GeocoderTimedOut, GeocoderServiceError):
            if attempt < retries:
                time.sleep(1.5)
                continue
            return [] if return_candidates else (None, None, None)


# ═════════════════════════════════════════
# STEP 2 — OSM query
# ═════════════════════════════════════════
def query_osm(lat, lon, radius_meters, category):
    tag_pairs = CATEGORY_MAP.get(category, [('amenity', category)])

    query_parts = []
    for key, value in tag_pairs:
        query_parts.append(
            f'  node["{key}"="{value}"](around:{radius_meters},{lat},{lon});'
        )
        query_parts.append(
            f'  way["{key}"="{value}"](around:{radius_meters},{lat},{lon});'
        )

    query = f"""[out:json][timeout:30];
(
{chr(10).join(query_parts)}
);
out center;"""

    data, ok = _overpass_post(query, label="query_osm (points)")
    if not ok or data is None:
        print("query_osm: fetch failed across all mirrors")
        return []

    features = []
    for element in data.get('elements', []):
        tags = element.get('tags', {})
        if element['type'] == 'node':
            el_lat = element.get('lat')
            el_lon = element.get('lon')
        elif element['type'] == 'way':
            center = element.get('center', {})
            el_lat = center.get('lat')
            el_lon = center.get('lon')
        else:
            continue
        if el_lat is None or el_lon is None:
            continue
        features.append({
            'lat':           float(el_lat),
            'lon':           float(el_lon),
            'name':          tags.get('name:en') or tags.get('name', 'Unnamed'),
            'type':          category,
            'phone':         tags.get('phone', ''),
            'website':       tags.get('website', ''),
            'opening_hours': tags.get('opening_hours', ''),
        })
    return features


# ═════════════════════════════════════════
# STEP 2b (NEW in V3.5) — Fetch LINE features (e.g. roads) with real geometry
# The existing query_osm() collapses ways to centre points (`out center`),
# which is fine for POIs but useless for roads — you can't buffer a point and
# call it a road. This fetches actual line geometry (`out geom`) so we can
# buffer it accurately.
# ═════════════════════════════════════════

# "Major roads" = the OSM highway classes a planner would call arterial.
MAJOR_ROAD_VALUES = ['motorway', 'trunk', 'primary', 'secondary',
                     'motorway_link', 'trunk_link', 'primary_link']

# Named line presets the agent can request
LINE_CATEGORY_MAP = {
    'major_roads': [('highway', v) for v in MAJOR_ROAD_VALUES],
    'all_roads':   [('highway', v) for v in MAJOR_ROAD_VALUES +
                    ['tertiary', 'residential', 'unclassified', 'service']],
    'railways':    [('railway', 'rail'), ('railway', 'light_rail'),
                    ('railway', 'subway')],
    'rivers':      [('waterway', 'river'), ('waterway', 'stream'),
                    ('waterway', 'canal')],
}


def query_osm_lines(lat, lon, radius_meters, line_category):
    """Fetch line features (roads, rail, rivers) WITH geometry.
    Returns (lines, fetch_ok). fetch_ok is False ONLY when the fetch failed
    across all mirrors/retries — so the caller can tell a real failure apart
    from a genuinely empty area. lines = [{'coords':[(lon,lat)...],'name','kind'}]."""
    tag_pairs = LINE_CATEGORY_MAP.get(line_category, [('highway', 'primary')])

    query_parts = []
    for key, value in tag_pairs:
        query_parts.append(
            f'  way["{key}"="{value}"](around:{radius_meters},{lat},{lon});'
        )

    # `out geom;` returns the full vertex list for each way
    query = f"""[out:json][timeout:60];
(
{chr(10).join(query_parts)}
);
out geom;"""

    data, ok = _overpass_post(query, label="query_osm_lines")
    if not ok or data is None:
        print(f"query_osm_lines: FETCH FAILED for {line_category}")
        return [], False

    lines = []
    for element in data.get('elements', []):
        if element.get('type') != 'way':
            continue
        geom = element.get('geometry') or []
        if len(geom) < 2:
            continue
        coords = [(pt['lon'], pt['lat']) for pt in geom
                  if 'lon' in pt and 'lat' in pt]
        if len(coords) < 2:
            continue
        tags = element.get('tags', {})
        lines.append({
            'coords': coords,
            'name':   tags.get('name:en') or tags.get('name', 'Unnamed road'),
            'kind':   tags.get('highway') or tags.get('railway')
                      or tags.get('waterway') or 'line',
        })
    print(f"query_osm_lines: {len(lines)} {line_category} features fetched")
    return lines, True


# ═════════════════════════════════════════
# STEP 3 — File ingestion
# ═════════════════════════════════════════
def _looks_like_esri_json(obj):
    if not isinstance(obj, dict):
        return False
    if 'features' not in obj or not isinstance(obj['features'], list):
        return False
    if 'geometryType' in obj:
        return True
    if obj['features']:
        first = obj['features'][0]
        if isinstance(first, dict) and 'attributes' in first:
            return True
    return False


def _esri_geom_to_shapely(geom, geom_type):
    if not isinstance(geom, dict):
        return None
    try:
        if geom_type == 'esriGeometryPoint':
            return Point(geom['x'], geom['y'])
        if geom_type == 'esriGeometryMultipoint':
            return MultiPoint([(p[0], p[1]) for p in geom.get('points', [])])
        if geom_type == 'esriGeometryPolyline':
            paths = geom.get('paths', [])
            if not paths:
                return None
            lines = [LineString(p) for p in paths if len(p) >= 2]
            if not lines:
                return None
            return lines[0] if len(lines) == 1 else MultiLineString(lines)
        if geom_type == 'esriGeometryPolygon':
            rings = geom.get('rings', [])
            if not rings:
                return None
            polys = [Polygon(r) for r in rings if len(r) >= 4]
            if not polys:
                return None
            return polys[0] if len(polys) == 1 else MultiPolygon(polys)
    except Exception as e:
        print(f"Esri geom parse error: {e}")
        return None
    return None


def _esri_json_to_geodataframe(esri):
    geom_type = esri.get('geometryType', 'esriGeometryPoint')
    rows = []
    for feat in esri.get('features', []):
        attrs = dict(feat.get('attributes') or {})
        shape = _esri_geom_to_shapely(feat.get('geometry'), geom_type)
        if shape is None:
            continue
        attrs['geometry'] = shape
        rows.append(attrs)

    if not rows:
        raise ValueError("Esri JSON contains no parseable features")

    gdf = gpd.GeoDataFrame(rows, geometry='geometry')

    sr = esri.get('spatialReference') or {}
    if 'wkt' in sr and sr['wkt']:
        try:
            gdf.set_crs(sr['wkt'], inplace=True, allow_override=True)
        except Exception:
            wkid = sr.get('latestWkid') or sr.get('wkid')
            if wkid:
                gdf.set_crs(epsg=int(wkid), inplace=True, allow_override=True)
    elif sr.get('latestWkid') or sr.get('wkid'):
        wkid = sr.get('latestWkid') or sr.get('wkid')
        gdf.set_crs(epsg=int(wkid), inplace=True, allow_override=True)
    else:
        gdf.set_crs(epsg=4326, inplace=True, allow_override=True)

    return gdf


def _parse_geojson_or_esri(content_bytes):
    text = content_bytes.decode('utf-8-sig', errors='replace')
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Not valid JSON: {e}")

    if _looks_like_esri_json(obj):
        return _esri_json_to_geodataframe(obj)

    gdf = gpd.read_file(StringIO(text))
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
    return gdf


def _parse_shapefile_zip(content_bytes):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with zipfile.ZipFile(BytesIO(content_bytes)) as z:
                z.extractall(tmp)
        except zipfile.BadZipFile:
            raise ValueError("Uploaded zip is not a valid archive")

        shp_path = None
        for root, _, files in os.walk(tmp):
            for f in files:
                if f.lower().endswith('.shp'):
                    shp_path = os.path.join(root, f)
                    break
            if shp_path:
                break

        if shp_path is None:
            raise ValueError(
                "No .shp file found in the zip. Must include .shp/.shx/.dbf "
                "(and ideally .prj)."
            )

        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        return gdf


def _parse_kml_bytes(kml_bytes):
    """Read raw KML bytes. KMLs often contain multiple layers (e.g. an
    Esri-exported KMZ stores each shapefile as a separate layer) — read
    ALL of them and concatenate so nothing gets silently dropped."""
    import warnings
    with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as tmp:
        tmp.write(kml_bytes)
        tmp_path = tmp.name
    try:
        # Discover all layers in the KML
        try:
            import pyogrio
            layer_info = pyogrio.list_layers(tmp_path)
            layer_names = [row[0] for row in layer_info]
        except Exception:
            layer_names = [None]  # fallback: let geopandas pick default
        if not layer_names:
            layer_names = [None]

        all_gdfs = []
        for ln in layer_names:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g = gpd.read_file(tmp_path, layer=ln) if ln else gpd.read_file(tmp_path)
                if g is not None and len(g) > 0:
                    all_gdfs.append(g)
            except Exception as e:
                print(f"KML layer {ln!r} read error: {e}")

        if not all_gdfs:
            raise ValueError("Could not read any layer from the KML")

        if len(all_gdfs) == 1:
            gdf = all_gdfs[0]
        else:
            gdf = gpd.GeoDataFrame(
                pd.concat(all_gdfs, ignore_index=True, sort=False),
                crs=all_gdfs[0].crs or 'EPSG:4326',
            )

        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        print(f"KML: combined {len(all_gdfs)} layer(s), {len(gdf)} features total")
        return gdf
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _parse_kmz(content_bytes):
    try:
        with zipfile.ZipFile(BytesIO(content_bytes)) as z:
            kml_name = None
            for n in z.namelist():
                if n.lower().endswith('.kml'):
                    kml_name = n
                    break
            if kml_name is None:
                raise ValueError("No .kml file inside the KMZ archive")
            kml_bytes = z.read(kml_name)
    except zipfile.BadZipFile:
        raise ValueError("Uploaded KMZ is not a valid archive")
    return _parse_kml_bytes(kml_bytes)


def _parse_csv_with_coords(content_bytes):
    text = content_bytes.decode('utf-8-sig', errors='replace')
    try:
        df = pd.read_csv(StringIO(text))
    except Exception as e:
        raise ValueError(f"Could not read CSV: {e}")

    cols = {c.lower().strip(): c for c in df.columns}
    lat_col, lon_col = None, None
    for c in ('latitude', 'lat', 'y'):
        if c in cols:
            lat_col = cols[c]
            break
    for c in ('longitude', 'lon', 'lng', 'long', 'x'):
        if c in cols:
            lon_col = cols[c]
            break

    if not lat_col or not lon_col:
        raise ValueError(
            f"CSV needs latitude/longitude columns. Found: {list(df.columns)}. "
            f"Expected latitude/lat/y AND longitude/lon/lng/long/x."
        )

    df = df.dropna(subset=[lat_col, lon_col])
    geom = [Point(xy) for xy in zip(df[lon_col].astype(float),
                                    df[lat_col].astype(float))]
    return gpd.GeoDataFrame(df, geometry=geom, crs='EPSG:4326')


def _parse_gpkg(content_bytes):
    with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as tmp:
        tmp.write(content_bytes)
        tmp_path = tmp.name
    try:
        gdf = gpd.read_file(tmp_path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        return gdf
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def parse_uploaded_file(content_bytes, filename):
    """Main dispatcher. Returns a GeoDataFrame in EPSG:4326."""
    if not content_bytes:
        raise ValueError("Uploaded file is empty")
    size_mb = len(content_bytes) / (1024 * 1024)
    print(f"DEBUG upload: filename={filename!r}, bytes={len(content_bytes)}, "
          f"first_16_hex={content_bytes[:16].hex()}")
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File too large ({size_mb:.1f} MB). Limit is {MAX_FILE_SIZE_MB} MB."
        )

    name = (filename or '').lower().strip()
    if not name:
        raise ValueError("Uploaded file has no name; cannot detect format")

    if name.endswith('.csv'):
        gdf = _parse_csv_with_coords(content_bytes)
    elif name.endswith('.zip'):
        gdf = _parse_shapefile_zip(content_bytes)
    elif name.endswith('.kmz'):
        gdf = _parse_kmz(content_bytes)
    elif name.endswith('.kml'):
        gdf = _parse_kml_bytes(content_bytes)
    elif name.endswith('.geojson') or name.endswith('.json'):
        gdf = _parse_geojson_or_esri(content_bytes)
    elif name.endswith('.gpkg'):
        gdf = _parse_gpkg(content_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: {filename}. V2 supports: "
            ".geojson, .json (incl. Esri JSON), .zip (shapefile), "
            ".kml, .kmz, .csv (with lat/lon columns), .gpkg"
        )

    if gdf is None or len(gdf) == 0:
        raise ValueError("Parsed file but found no usable features")

    try:
        epsg = gdf.crs.to_epsg() if gdf.crs else None
    except Exception:
        epsg = None
    if epsg != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def summarize_gdf(gdf, filename, original_crs_str):
    geom_types = sorted(set(gdf.geometry.geom_type.dropna().tolist()))
    attr_cols = [c for c in gdf.columns if c != 'geometry']
    return {
        'filename':       filename,
        'feature_count':  int(len(gdf)),
        'geometry_types': geom_types,
        'original_crs':   original_crs_str,
        'display_crs':    'EPSG:4326',
        'attributes':     attr_cols[:20],
    }


# ═════════════════════════════════════════
# STEP 4 — Render user feature into folium layer
# Per-row, fully wrapped in try/except so one bad geometry can't 500
# the whole request. Polygons / lines / points each get their own style.
# ═════════════════════════════════════════
def _render_one_feature_to_layer(geom, popup, layer):
    if geom is None or geom.is_empty:
        return
    gt = geom.geom_type

    if gt == 'Point':
        folium.CircleMarker(
            [geom.y, geom.x],
            radius=7,
            color=USER_LAYER_COLOR,
            fill=True, fill_color=USER_LAYER_FILL, fill_opacity=0.7,
            popup=popup,
        ).add_to(layer)

    elif gt == 'MultiPoint':
        for p in geom.geoms:
            folium.CircleMarker(
                [p.y, p.x],
                radius=7,
                color=USER_LAYER_COLOR,
                fill=True, fill_color=USER_LAYER_FILL, fill_opacity=0.7,
                popup=popup,
            ).add_to(layer)

    elif gt == 'LineString':
        coords = [(c[1], c[0]) for c in geom.coords]
        folium.PolyLine(
            coords, color=USER_LAYER_COLOR, weight=4, opacity=0.85, popup=popup,
        ).add_to(layer)

    elif gt == 'MultiLineString':
        for ln in geom.geoms:
            coords = [(c[1], c[0]) for c in ln.coords]
            folium.PolyLine(
                coords, color=USER_LAYER_COLOR, weight=4, opacity=0.85,
                popup=popup,
            ).add_to(layer)

    elif gt == 'Polygon':
        _add_polygon(geom, popup, layer)

    elif gt == 'MultiPolygon':
        for poly in geom.geoms:
            _add_polygon(poly, popup, layer)

    else:
        print(f"User layer: unsupported geometry type {gt}, skipped")


def _add_polygon(poly, popup, layer):
    exterior = [(c[1], c[0]) for c in poly.exterior.coords]
    holes = [
        [(c[1], c[0]) for c in ring.coords]
        for ring in poly.interiors
    ]
    if holes:
        locations = [exterior] + holes
    else:
        locations = exterior
    folium.Polygon(
        locations=locations,
        color=USER_LAYER_COLOR,
        weight=2,
        fill=True,
        fill_color=USER_LAYER_FILL,
        fill_opacity=0.35,
        popup=popup,
    ).add_to(layer)


# ═════════════════════════════════════════
# STEP 5 — Build the interactive map
# ═════════════════════════════════════════
def generate_map(lat, lon, radius_meters, features, location_name, category,
                 user_gdf=None, user_filename=None, exposure=None, notice=None,
                 boundary=None):
    m = folium.Map(location=[lat, lon], zoom_start=15, tiles='CartoDB positron')
    color = STYLE_MAP.get(category, DEFAULT_COLOR)
    category_label = category.replace('_', ' ').title()
    count = len(features)

    badge_color = '#27ae60' if count > 0 else '#95a5a6'
    badge_icon  = '✅' if count > 0 else 'ℹ️'
    plural      = 's' if count != 1 else ''

    is_boundary = boundary is not None
    # ── Boundary mode: draw the district polygons first (outlined, light
    #    fill), then color points inside/outside. ──
    if is_boundary:
        b_gdf = boundary.get('_boundary_gdf')
        if b_gdf is not None and len(b_gdf) > 0:
            bnd_layer = folium.FeatureGroup(name="🗺️ District boundary", show=True)
            per_poly = {p['index']: p for p in boundary.get('per_polygon', [])}
            for bi in range(len(b_gdf)):
                geom = b_gdf.geometry.iloc[bi]
                if geom is None or geom.is_empty:
                    continue
                info = per_poly.get(bi, {})
                tip = (f"{info.get('name', 'Area ' + str(bi + 1))}: "
                       f"{info.get('count', 0)} {category_label.lower()}"
                       f"{'s' if info.get('count', 0) != 1 else ''} · "
                       f"{info.get('area', {}).get('km2', 0)} km² · "
                       f"{info.get('density', 0)}/km²")
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda _: {
                        'fillColor': '#1f6feb', 'color': '#1f6feb',
                        'weight': 2, 'fillOpacity': 0.07,
                    },
                    tooltip=tip,
                ).add_to(bnd_layer)
            bnd_layer.add_to(m)

    # ── Exposure mode: draw the comparison lines + buffer zone first
    #    (so points sit on top), then color points red/green by exposure. ──
    is_exposure = exposure is not None
    if is_exposure:
        line_label = exposure['line_category'].replace('_', ' ').title()
        within_m   = exposure['within_m']

        # Buffer zone (translucent blue fill)
        buf_gdf = exposure.get('_buffer_gdf')
        if buf_gdf is not None and len(buf_gdf) > 0:
            zone_layer = folium.FeatureGroup(
                name=f"🔵 {int(within_m)} m buffer", show=True)
            try:
                buf_geom = buf_gdf.geometry.union_all()
            except AttributeError:
                buf_geom = buf_gdf.geometry.unary_union
            folium.GeoJson(
                buf_geom.__geo_interface__,
                style_function=lambda _: {
                    'fillColor': '#1f6feb', 'color': '#1f6feb',
                    'weight': 1.2, 'fillOpacity': 0.08, 'dashArray': '6,4',
                },
            ).add_to(zone_layer)
            zone_layer.add_to(m)

        # Comparison lines (roads) — dark, drawn under points
        lines_gdf = exposure.get('_lines_gdf')
        if lines_gdf is not None and len(lines_gdf) > 0:
            line_layer = folium.FeatureGroup(
                name=f"➖ {line_label}", show=True)
            for _, lrow in lines_gdf.iterrows():
                geom = lrow.geometry
                if geom is None or geom.is_empty:
                    continue
                try:
                    if geom.geom_type == 'LineString':
                        coords = [(c[1], c[0]) for c in geom.coords]
                        folium.PolyLine(coords, color='#34404f', weight=2.5,
                                        opacity=0.8,
                                        tooltip=lrow.get('name', '')).add_to(line_layer)
                    elif geom.geom_type == 'MultiLineString':
                        for ln in geom.geoms:
                            coords = [(c[1], c[0]) for c in ln.coords]
                            folium.PolyLine(coords, color='#34404f', weight=2.5,
                                            opacity=0.8).add_to(line_layer)
                except Exception as e:
                    print(f"exposure line skip: {e}")
            line_layer.add_to(m)

    # === OSM features layer ===
    osm_layer = folium.FeatureGroup(name=f"{category_label} (OSM)", show=True)

    folium.Marker(
        [lat, lon],
        popup=folium.Popup(
            f"<b>📍 Center</b><br>{location_name[:80]}", max_width=250
        ),
        icon=folium.Icon(color='red', icon='map-marker', prefix='glyphicon'),
    ).add_to(osm_layer)

    # In exposure mode the radius circle adds clutter; only show it in
    # plain proximity mode.
    if not is_exposure:
        folium.Circle(
            [lat, lon],
            radius=radius_meters,
            color='#e74c3c', fill=True, fill_opacity=0.08,
            weight=2, dash_array='8',
            popup=f"Radius: {radius_meters / 1000:.1f} km",
        ).add_to(osm_layer)

    # Build a lookup of exposure tag per feature (by position) if available
    exposure_tags = None
    if is_exposure:
        tagged = exposure.get('_points_gdf')
        if tagged is not None and 'exposure' in tagged.columns \
                and len(tagged) == len(features):
            exposure_tags = list(tagged['exposure'])

    # Boundary mode: inside/outside tag per feature (by position)
    boundary_tags = None
    if is_boundary:
        bt = boundary.get('_points_gdf')
        if bt is not None and 'within' in bt.columns and len(bt) == len(features):
            boundary_tags = list(bt['within'])

    for i, f in enumerate(features):
        parts = [
            f"<b>{f['name']}</b><br>",
            f"<span style='color:#666'>{category_label}</span>",
        ]
        # Exposure tag in popup + color
        if exposure_tags is not None:
            tag = exposure_tags[i]
            pt_color = '#e74c3c' if tag == 'exposed' else '#27ae60'
            tag_label = ('🔴 Exposed (within %dm)' % exposure['within_m']
                         if tag == 'exposed' else '🟢 Shielded')
            parts.append(f"<br><b style='color:{pt_color}'>{tag_label}</b>")
        elif boundary_tags is not None:
            tag = boundary_tags[i]
            pt_color = '#1f6feb' if tag == 'inside' else '#b0b6c0'
            tag_label = ('🔵 Inside district' if tag == 'inside'
                         else '⚪ Outside district')
            parts.append(f"<br><b style='color:{pt_color}'>{tag_label}</b>")
        else:
            pt_color = color
        if f.get('phone'):
            parts.append(f"<br>📞 {f['phone']}")
        if f.get('opening_hours'):
            parts.append(f"<br>⏰ {f['opening_hours']}")
        if f.get('website'):
            parts.append(
                f"<br>🔗 <a href='{f['website']}' target='_blank'>website</a>"
            )
        popup_html = (
            "<div style='font-family:Arial;min-width:160px'>"
            + ''.join(parts) + "</div>"
        )
        # Points outside the district are drawn smaller + faded
        pt_radius = 9
        pt_opacity = 0.85
        if boundary_tags is not None and boundary_tags[i] == 'outside':
            pt_radius = 5
            pt_opacity = 0.5
        folium.CircleMarker(
            [f['lat'], f['lon']],
            radius=pt_radius,
            color=pt_color, fill=True, fill_color=pt_color, fill_opacity=pt_opacity,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f['name'],
        ).add_to(osm_layer)
    osm_layer.add_to(m)

    # === User-uploaded layer ===
    if user_gdf is not None and len(user_gdf) > 0:
        user_label = (user_filename or 'Your Data').rsplit('.', 1)[0][:40]
        user_layer = folium.FeatureGroup(name=f"📂 {user_label}", show=True)

        # Up to 5 attribute columns for the popup
        popup_cols = [c for c in user_gdf.columns if c != 'geometry'][:5]

        rendered = 0
        skipped  = 0
        # Detect KML-origin layers — those carry attributes inside Description HTML
        has_kml_description = 'Description' in user_gdf.columns

        for idx, row in user_gdf.iterrows():
            try:
                # Prefer real attributes parsed out of KML <description>;
                # fall back to whatever columns the GDF has otherwise.
                attrs = {}
                if has_kml_description:
                    attrs = _extract_kml_attrs_from_description(
                        row.get('Description', '')
                    )
                    # Include the KML Name field if present and not already there
                    name_val = _safe_str(row.get('Name', ''))
                    if name_val and 'Name' not in attrs:
                        attrs = {'Name': name_val, **attrs}

                if not attrs:
                    # Standard path for shapefile / geojson / esri-json / csv
                    for c in popup_cols:
                        v = _safe_str(row.get(c, ''))
                        if v:
                            attrs[c] = v

                # Build the popup as a clean ArcGIS-style attribute table
                popup = None
                if attrs:
                    rows_html = "".join(
                        f"<tr>"
                        f"<td style='padding:3px 8px 3px 0;font-weight:600;"
                        f"color:#1a1a2e;vertical-align:top;white-space:nowrap'>{k}</td>"
                        f"<td style='padding:3px 0;color:#333'>{v}</td>"
                        f"</tr>"
                        for k, v in attrs.items()
                    )
                    popup_html = (
                        "<div style='font-family:Arial;font-size:12px;"
                        "min-width:220px;max-width:320px'>"
                        "<div style='font-weight:700;font-size:13px;color:#1f6feb;"
                        "border-bottom:1px solid #e0e0e0;padding-bottom:4px;"
                        "margin-bottom:6px'>Feature Attributes</div>"
                        "<table style='border-collapse:collapse;width:100%'>"
                        + rows_html +
                        "</table></div>"
                    )
                    popup = folium.Popup(popup_html, max_width=340)

                _render_one_feature_to_layer(row.geometry, popup, user_layer)
                rendered += 1
            except Exception as e:
                skipped += 1
                print(f"User layer row {idx} skipped: {type(e).__name__}: {e}")

        print(f"User layer: rendered={rendered}, skipped={skipped}")
        user_layer.add_to(m)

        # Fit bounds to include both the user layer and the OSM radius
        try:
            minx, miny, maxx, maxy = user_gdf.total_bounds
            deg = radius_meters / 111000.0
            minx = min(minx, lon - deg)
            maxx = max(maxx, lon + deg)
            miny = min(miny, lat - deg)
            maxy = max(maxy, lat + deg)
            m.fit_bounds([[miny, minx], [maxy, maxx]])
        except Exception as e:
            print(f"fit_bounds error: {e}")

    folium.LayerControl(collapsed=False, position='topleft').add_to(m)

    # === Stats overlay (top-right) ===
    truncated = (
        location_name[:50] + '...' if len(location_name) > 50 else location_name
    )
    user_layer_html = ""
    if user_gdf is not None and len(user_gdf) > 0:
        user_layer_html = (
            f"<div style='font-size:13px;color:#1f6feb;margin-bottom:6px'>"
            f"📂 {len(user_gdf)} feature(s) from your file</div>"
        )

    # Exposure mode shows a breakdown box instead of the simple "found" badge
    if is_exposure:
        line_label = exposure['line_category'].replace('_', ' ').title()
        badge_html = f"""
        <div style='background:#1a1a2e;color:white;border-radius:8px;
                    padding:10px 12px;font-weight:600;font-size:13px'>
            <div style='margin-bottom:6px'>
                {exposure['total_points']} {category_label}{plural} analysed
            </div>
            <div style='display:flex;justify-content:space-between;
                        color:#ff8a80;margin-bottom:3px'>
                <span>🔴 Exposed (&le;{int(exposure['within_m'])}m)</span>
                <span>{exposure['exposed']} · {exposure['pct_exposed']}%</span>
            </div>
            <div style='display:flex;justify-content:space-between;color:#a5e8b8'>
                <span>🟢 Shielded</span>
                <span>{exposure['shielded']}</span>
            </div>
            <div style='border-top:1px solid #3a4156;margin-top:7px;
                        padding-top:6px;font-weight:400;font-size:11px;color:#aab'>
                vs {line_label} · buffer {exposure['buffer_area']['km2']} km²
            </div>
        </div>
        """
        accent = '#1a1a2e'
    elif is_boundary:
        # Per-polygon rows (cap at 6 for the overlay; full list in PDF)
        rows = ""
        for p in boundary.get('per_polygon', [])[:6]:
            rows += (
                f"<div style='display:flex;justify-content:space-between;"
                f"margin-bottom:2px'><span style='max-width:150px;"
                f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
                f"🔵 {p['name']}</span>"
                f"<span>{p['count']} · {p['density']}/km²</span></div>"
            )
        more = boundary['polygon_count'] - 6
        if more > 0:
            rows += (f"<div style='color:#aab;font-size:11px'>"
                     f"+ {more} more area(s)</div>")
        badge_html = f"""
        <div style='background:#1a1a2e;color:white;border-radius:8px;
                    padding:10px 12px;font-weight:600;font-size:12.5px'>
            <div style='margin-bottom:6px'>
                {boundary['total_inside']} of {boundary['total_points']}
                {category_label}{plural} inside
            </div>
            {rows}
            <div style='border-top:1px solid #3a4156;margin-top:7px;
                        padding-top:6px;font-weight:400;font-size:11px;color:#aab'>
                {boundary['polygon_count']} area(s) · {boundary['total_area']['km2']} km²
                · {boundary['overall_density']}/km² overall
            </div>
        </div>
        """
        accent = '#1a1a2e'
    else:
        badge_html = f"""
        <div style='background:{badge_color};color:white;border-radius:8px;
                    padding:8px;text-align:center;font-weight:700'>
            {badge_icon} {count} {category_label}{plural} Found
        </div>
        """
        accent = color

    if is_exposure:
        search_line = (f"📏 within {int(exposure['within_m'])} m of "
                       f"{exposure['line_category'].replace('_', ' ')}")
    elif is_boundary:
        search_line = f"🗺️ within district boundary"
    else:
        search_line = f"📏 {radius_meters / 1000:.1f} km radius"

    notice_html = ""
    if notice:
        notice_html = (
            "<div style='background:#fff4e5;border:1px solid #f0c277;"
            "color:#8a5300;border-radius:6px;padding:8px 10px;"
            "font-size:11.5px;margin-bottom:10px;line-height:1.4'>"
            f"⚠️ {notice}</div>"
        )

    stats_html = f"""
    <div style='position:fixed;top:15px;right:15px;background:white;
                padding:16px 20px;border-radius:12px;
                box-shadow:0 4px 20px rgba(0,0,0,0.15);z-index:1000;
                min-width:250px;max-width:300px;font-family:Arial;
                border-left:4px solid {accent}'>
        <div style='font-size:17px;font-weight:700;margin-bottom:2px'>
            K&amp;L Geospatial
        </div>
        <div style='font-size:10px;letter-spacing:2px;color:#888;
                    margin-bottom:10px'>
            GEOAI GOVERNMENT ASSISTANT
        </div>
        {notice_html}
        <div style='font-size:13px;color:#555;margin-bottom:6px'>
            📍 {truncated}
        </div>
        <div style='font-size:13px;color:#555;margin-bottom:6px'>
            🔍 {category_label}
        </div>
        <div style='font-size:13px;color:#555;margin-bottom:12px'>
            {search_line}
        </div>
        {user_layer_html}
        {badge_html}
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    return m.get_root().render()


# ═════════════════════════════════════════
# GEOPROCESSING ENGINE (NEW in V3.5)
# CRS-safe spatial operations. Every measurement reprojects to an
# appropriate metre-based CRS first, so areas/distances are real-world
# accurate — never computed in degrees.
# ═════════════════════════════════════════
def pick_metric_crs(lat, lon):
    """Return an EPSG code for an appropriate metre-based CRS at this location.
    Correct UTM zone for the point; World Mercator near the poles.
    Verified against Riyadh (32638), Beirut (32636), and others."""
    if lat is None or lon is None:
        return 3857
    if lat > 84 or lat < -80:
        return 3857
    zone = int((lon + 180) / 6) + 1
    zone = max(1, min(zone, 60))
    return (32600 if lat >= 0 else 32700) + zone


def _lines_to_gdf(lines):
    """Convert query_osm_lines() output → GeoDataFrame (WGS84)."""
    from shapely.geometry import LineString
    rows = []
    for ln in lines:
        try:
            rows.append({
                'name': ln.get('name', 'Unnamed'),
                'kind': ln.get('kind', 'line'),
                'geometry': LineString(ln['coords']),
            })
        except Exception as e:
            print(f"_lines_to_gdf skip: {e}")
    if not rows:
        return gpd.GeoDataFrame(
            {'name': [], 'kind': [], 'geometry': []}, crs='EPSG:4326')
    return gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')


def _points_to_gdf(features):
    """Convert query_osm() point output → GeoDataFrame (WGS84)."""
    from shapely.geometry import Point
    rows = []
    for f in features:
        try:
            rows.append({
                'name': f.get('name', 'Unnamed'),
                'geometry': Point(f['lon'], f['lat']),
            })
        except Exception as e:
            print(f"_points_to_gdf skip: {e}")
    if not rows:
        return gpd.GeoDataFrame(
            {'name': [], 'geometry': []}, crs='EPSG:4326')
    return gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')


def buffer_features(gdf, distance_m, lat, lon, dissolve=True):
    """Buffer features by distance_m metres. Reprojects to a metric CRS,
    buffers, optionally dissolves to one polygon, returns WGS84 GeoDataFrame.
    Also returns the metric-CRS buffer for accurate area calc."""
    if gdf is None or len(gdf) == 0:
        return None, None, 0
    metric_epsg = pick_metric_crs(lat, lon)
    g_metric = gdf.to_crs(epsg=metric_epsg)
    buffered = g_metric.geometry.buffer(distance_m)

    if dissolve:
        try:
            merged = buffered.union_all()       # shapely 2.0 / geopandas 1.0
        except AttributeError:
            merged = buffered.unary_union        # older fallback
        buf_metric = gpd.GeoDataFrame(
            {'geometry': [merged]}, geometry='geometry', crs=f'EPSG:{metric_epsg}')
    else:
        buf_metric = gpd.GeoDataFrame(
            {'geometry': buffered}, geometry='geometry', crs=f'EPSG:{metric_epsg}')

    area_m2 = float(buf_metric.geometry.area.sum())
    buf_wgs = buf_metric.to_crs(epsg=4326)
    return buf_wgs, buf_metric, area_m2


def tag_within_buffer(points_gdf, buffer_wgs):
    """Tag each point as inside ('exposed') or outside ('shielded') the buffer.
    Returns (tagged_gdf, n_inside, n_outside)."""
    if points_gdf is None or len(points_gdf) == 0:
        return points_gdf, 0, 0
    if buffer_wgs is None or len(buffer_wgs) == 0:
        out = points_gdf.copy()
        out['exposure'] = 'shielded'
        return out, 0, len(out)

    # Single dissolved buffer geometry
    try:
        buf_geom = buffer_wgs.geometry.union_all()
    except AttributeError:
        buf_geom = buffer_wgs.geometry.unary_union

    out = points_gdf.copy()
    inside_mask = out.geometry.within(buf_geom)
    out['exposure'] = inside_mask.map({True: 'exposed', False: 'shielded'})
    n_in = int(inside_mask.sum())
    n_out = int(len(out) - n_in)
    return out, n_in, n_out


def compute_area_units(area_m2):
    """Return area in m², hectares, and km² as a dict."""
    return {
        'm2':  round(area_m2, 1),
        'ha':  round(area_m2 / 10_000.0, 3),
        'km2': round(area_m2 / 1_000_000.0, 4),
    }


def compute_length_units(gdf, lat, lon):
    """Total length of line features in metres and km."""
    if gdf is None or len(gdf) == 0:
        return {'m': 0.0, 'km': 0.0}
    metric_epsg = pick_metric_crs(lat, lon)
    g = gdf.to_crs(epsg=metric_epsg)
    total_m = float(g.geometry.length.sum())
    return {'m': round(total_m, 1), 'km': round(total_m / 1000.0, 3)}


def proximity_exposure_analysis(lat, lon, radius_m, point_category,
                                line_category, within_m, prefetched_points=None):
    """The mockup's hero query, as a reusable function.
    'Which <point_category> are within <within_m> of <line_category>?'
    Returns a dict of results + the GeoDataFrames for rendering.

    prefetched_points: optional list of point feature dicts already fetched
    by the caller (avoids querying OSM twice when /analyze already has them).
    """
    # 1. Fetch both layers (reuse points if the caller already has them)
    point_feats = (prefetched_points if prefetched_points is not None
                   else query_osm(lat, lon, radius_m, point_category))
    line_feats, lines_ok = query_osm_lines(lat, lon, radius_m, line_category)

    points_gdf = _points_to_gdf(point_feats)
    lines_gdf  = _lines_to_gdf(line_feats)

    # 2. Buffer the lines
    buf_wgs, buf_metric, area_m2 = buffer_features(
        lines_gdf, within_m, lat, lon, dissolve=True)

    # 3. Tag points inside/outside the buffer
    tagged, n_in, n_out = tag_within_buffer(points_gdf, buf_wgs)

    total = n_in + n_out
    pct_in = round(100.0 * n_in / total, 1) if total else 0.0

    return {
        'point_category': point_category,
        'line_category':  line_category,
        'within_m':       within_m,
        'total_points':   total,
        'exposed':        n_in,
        'shielded':       n_out,
        'pct_exposed':    pct_in,
        'road_length':    compute_length_units(lines_gdf, lat, lon),
        'buffer_area':    compute_area_units(area_m2),
        'road_fetch_failed': (not lines_ok),   # True only on real fetch failure
        'road_count':     int(len(lines_gdf)),
        # GeoDataFrames for the renderer (used in V3.5b):
        '_points_gdf':    tagged,
        '_lines_gdf':     lines_gdf,
        '_buffer_gdf':    buf_wgs,
    }


def within_boundary_analysis(lat, lon, radius_m, point_category,
                             boundary_gdf, prefetched_points=None):
    """V3.6 — count features of `point_category` that fall INSIDE each polygon
    of `boundary_gdf` (a user-uploaded district/area boundary).

    Per-polygon counts + area + density, plus a combined total. Points are
    tagged 'inside'/'outside'. Returns a dict + GeoDataFrames for rendering.
    """
    # 1. Points (reuse if caller already fetched them)
    point_feats = (prefetched_points if prefetched_points is not None
                   else query_osm(lat, lon, radius_m, point_category))
    points_gdf = _points_to_gdf(point_feats)

    # 2. Keep only polygon geometries from the uploaded boundary
    if boundary_gdf is None or len(boundary_gdf) == 0:
        return None
    try:
        poly_mask = boundary_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        polys = boundary_gdf[poly_mask].copy().reset_index(drop=True)
    except Exception as e:
        print(f"within_boundary: polygon filter failed: {e}")
        return None
    if len(polys) == 0:
        return {'boundary_has_no_polygons': True}

    # 3. Metric CRS for accurate areas
    metric_epsg = pick_metric_crs(lat, lon)
    polys_metric = polys.to_crs(epsg=metric_epsg)

    # Try to find a human-readable name column for each polygon
    name_col = None
    for cand in ('name', 'Name', 'NAME', 'district', 'District', 'DISTRICT',
                 'NAME_EN', 'name_en', 'ADM3_EN', 'label', 'Label'):
        if cand in polys.columns:
            name_col = cand
            break

    # 4. Per-polygon: count points inside, compute area + density
    per_polygon = []
    if len(points_gdf) > 0:
        pts_for_test = points_gdf.geometry
    else:
        pts_for_test = None

    # Tag each point with the index of the polygon it falls in (-1 = outside)
    point_tags = [-1] * len(points_gdf)

    for pi in range(len(polys)):
        poly_geom = polys.geometry.iloc[pi]
        area_m2 = float(polys_metric.geometry.iloc[pi].area)
        inside_names = []
        count_in = 0
        if pts_for_test is not None:
            try:
                mask = pts_for_test.within(poly_geom)
                for idx_pos, is_in in enumerate(mask.tolist()):
                    if is_in and point_tags[idx_pos] == -1:
                        point_tags[idx_pos] = pi
                        count_in += 1
                        nm = points_gdf.iloc[idx_pos].get('name', 'Unnamed')
                        inside_names.append(nm)
            except Exception as e:
                print(f"within_boundary: point test failed for poly {pi}: {e}")
        area_units = compute_area_units(area_m2)
        density = round(count_in / area_units['km2'], 2) if area_units['km2'] > 0 else 0.0
        label = None
        if name_col is not None:
            try:
                label = str(polys.iloc[pi][name_col])
            except Exception:
                label = None
        per_polygon.append({
            'index':    pi,
            'name':     label or f"Area {pi + 1}",
            'count':    count_in,
            'area':     area_units,
            'density':  density,    # features per km²
        })

    total_inside = sum(p['count'] for p in per_polygon)
    total_outside = len(points_gdf) - total_inside
    total_area_m2 = float(polys_metric.geometry.area.sum())

    # Build a tagged points gdf for rendering (inside/outside)
    tagged_points = points_gdf.copy()
    if len(tagged_points) > 0:
        tagged_points['within'] = ['inside' if t >= 0 else 'outside'
                                   for t in point_tags]

    return {
        'point_category':  point_category,
        'polygon_count':   int(len(polys)),
        'total_points':    int(len(points_gdf)),
        'total_inside':    int(total_inside),
        'total_outside':   int(total_outside),
        'total_area':      compute_area_units(total_area_m2),
        'overall_density': round(total_inside / compute_area_units(total_area_m2)['km2'], 2)
                           if compute_area_units(total_area_m2)['km2'] > 0 else 0.0,
        'per_polygon':     per_polygon,
        # GeoDataFrames for rendering:
        '_points_gdf':     tagged_points,
        '_boundary_gdf':   polys,   # WGS84 polygons
    }


# ═════════════════════════════════════════
# STEP 6 (NEW in V2.5) — Static cartographic map
# Print-quality PNG with basemap, OSM features, user layer,
# radius, north arrow, scale bar, legend, attribution.
# Returns a path to a temp PNG (caller must delete after use).
# ═════════════════════════════════════════
def generate_static_map(lat, lon, radius_meters, features, location_name,
                        category, user_gdf=None, exposure=None, boundary=None):
    """Render a print-quality static map. Returns a temp PNG path."""
    import math
    import matplotlib
    matplotlib.use('Agg')                          # headless backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.lines import Line2D
    import contextily as cx
    from pyproj import Transformer, CRS
    from shapely.geometry import Point as ShapelyPoint
    from shapely.ops import transform as shapely_transform

    is_exposure = exposure is not None
    is_boundary = boundary is not None

    # WGS84 → Web Mercator (for plotting alongside basemap tiles)
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857",
                                   always_xy=True).transform
    cx_x, cx_y = to_3857(lon, lat)

    # Build a geographically accurate search-radius circle using AEQD
    # projection centered on the query (azimuthal equidistant — distances
    # from the center are real-world meters), then reproject the buffer
    # polygon to Web Mercator for plotting.
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +ellps=WGS84"
    )
    to_aeqd      = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True).transform
    aeqd_to_3857 = Transformer.from_crs(aeqd, "EPSG:3857", always_xy=True).transform
    buffer_aeqd  = shapely_transform(to_aeqd, ShapelyPoint(lon, lat)).buffer(radius_meters)
    buffer_3857  = shapely_transform(aeqd_to_3857, buffer_aeqd)

    bxmin, bymin, bxmax, bymax = buffer_3857.bounds

    # Expand bounds to include uploaded layer (Option B from V2.5 plan)
    if user_gdf is not None and len(user_gdf) > 0:
        try:
            user_3857 = user_gdf.to_crs(epsg=3857)
            uxmin, uymin, uxmax, uymax = user_3857.total_bounds
            bxmin = min(bxmin, uxmin); bymin = min(bymin, uymin)
            bxmax = max(bxmax, uxmax); bymax = max(bymax, uymax)
        except Exception as e:
            print(f"Static map: user layer reprojection failed: {e}")
            user_gdf = None

    # Pre-project exposure layers (lines + buffer) and expand bounds to fit
    exp_lines_3857 = None
    exp_buffer_3857 = None
    if is_exposure:
        try:
            lg = exposure.get('_lines_gdf')
            if lg is not None and len(lg) > 0:
                exp_lines_3857 = lg.to_crs(epsg=3857)
            bg = exposure.get('_buffer_gdf')
            if bg is not None and len(bg) > 0:
                exp_buffer_3857 = bg.to_crs(epsg=3857)
                lxmin, lymin, lxmax, lymax = exp_buffer_3857.total_bounds
                bxmin = min(bxmin, lxmin); bymin = min(bymin, lymin)
                bxmax = max(bxmax, lxmax); bymax = max(bymax, lymax)
        except Exception as e:
            print(f"Static map: exposure layer reprojection failed: {e}")

    # Pre-project boundary polygons (V3.6) and expand bounds to fit them
    bnd_3857 = None
    if is_boundary:
        try:
            bg = boundary.get('_boundary_gdf')
            if bg is not None and len(bg) > 0:
                bnd_3857 = bg.to_crs(epsg=3857)
                pxmin, pymin, pxmax, pymax = bnd_3857.total_bounds
                bxmin = min(bxmin, pxmin); bymin = min(bymin, pymin)
                bxmax = max(bxmax, pxmax); bymax = max(bymax, pymax)
        except Exception as e:
            print(f"Static map: boundary reprojection failed: {e}")

    # Padding + maintain a reasonable aspect (close to STATIC_MAP_W/H)
    width  = bxmax - bxmin
    height = bymax - bymin
    target_aspect = STATIC_MAP_W / STATIC_MAP_H
    if width / height > target_aspect:
        extra = (width / target_aspect - height) / 2
        bymin -= extra; bymax += extra
    elif height / width > 1.0 / target_aspect:
        extra = (height * target_aspect - width) / 2
        bxmin -= extra; bxmax += extra
    width  = bxmax - bxmin
    height = bymax - bymin
    bxmin -= width * 0.05; bxmax += width * 0.05
    bymin -= height * 0.05; bymax += height * 0.05

    fig, ax = plt.subplots(figsize=(STATIC_MAP_W, STATIC_MAP_H),
                           dpi=STATIC_MAP_DPI)
    ax.set_xlim(bxmin, bxmax)
    ax.set_ylim(bymin, bymax)

    # Basemap (best-effort — keep going if tile fetch fails)
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                       attribution=False, zoom='auto')
    except Exception as e:
        print(f"Static map: basemap fetch failed ({e}); continuing without it")

    # Search-radius polygon (red dashed) — only in plain proximity mode
    if not is_exposure and buffer_3857.geom_type == 'Polygon':
        ax.add_patch(MplPolygon(
            list(buffer_3857.exterior.coords),
            facecolor='#e74c3c', alpha=0.08,
            edgecolor='#e74c3c', linestyle='--', linewidth=1.8, zorder=2,
        ))

    # Exposure buffer zone (translucent blue) + comparison lines
    if is_exposure:
        if exp_buffer_3857 is not None:
            try:
                exp_buffer_3857.plot(ax=ax, facecolor='#1f6feb', alpha=0.08,
                                     edgecolor='#1f6feb', linewidth=1.2,
                                     linestyle=(0, (6, 4)), zorder=2)
            except Exception as e:
                print(f"Static map: buffer plot failed: {e}")
        if exp_lines_3857 is not None:
            try:
                exp_lines_3857.plot(ax=ax, color='#34404f', linewidth=1.8,
                                    alpha=0.8, zorder=3)
            except Exception as e:
                print(f"Static map: lines plot failed: {e}")

    # Boundary polygons (V3.6) — outlined, light fill
    if is_boundary and bnd_3857 is not None:
        try:
            bnd_3857.plot(ax=ax, facecolor='#1f6feb', alpha=0.07,
                          edgecolor='#1f6feb', linewidth=1.8, zorder=2)
        except Exception as e:
            print(f"Static map: boundary plot failed: {e}")

    # User uploaded layer (per geometry type — same color scheme as web map)
    if user_gdf is not None and len(user_gdf) > 0:
        try:
            user_3857 = user_gdf.to_crs(epsg=3857)
            gt = user_3857.geometry.geom_type
            polys  = user_3857[gt.isin(['Polygon', 'MultiPolygon'])]
            lines  = user_3857[gt.isin(['LineString', 'MultiLineString'])]
            points = user_3857[gt.isin(['Point', 'MultiPoint'])]
            if not polys.empty:
                polys.plot(ax=ax, facecolor=USER_LAYER_FILL,
                           edgecolor=USER_LAYER_COLOR, alpha=0.5,
                           linewidth=1.2, zorder=4)
            if not lines.empty:
                lines.plot(ax=ax, color=USER_LAYER_COLOR, linewidth=2.5,
                           alpha=0.9, zorder=4)
            if not points.empty:
                points.plot(ax=ax, color=USER_LAYER_FILL,
                            edgecolor=USER_LAYER_COLOR, markersize=50,
                            alpha=0.85, linewidth=1, zorder=5)
        except Exception as e:
            print(f"Static map: user layer plot failed: {e}")

    # OSM features
    category_color = STYLE_MAP.get(category, DEFAULT_COLOR)
    category_label = category.replace('_', ' ').title()

    # Exposure-tag lookup by position
    exposure_tags = None
    if is_exposure:
        tagged = exposure.get('_points_gdf')
        if tagged is not None and 'exposure' in tagged.columns \
                and len(tagged) == len(features):
            exposure_tags = list(tagged['exposure'])

    # Boundary inside/outside lookup by position
    boundary_tags = None
    if is_boundary:
        bt = boundary.get('_points_gdf')
        if bt is not None and 'within' in bt.columns and len(bt) == len(features):
            boundary_tags = list(bt['within'])

    if features:
        if exposure_tags is not None:
            # Two scatter calls: exposed (red) and shielded (green)
            ex_x, ex_y, sh_x, sh_y = [], [], [], []
            for f, tag in zip(features, exposure_tags):
                x, y = to_3857(f['lon'], f['lat'])
                if tag == 'exposed':
                    ex_x.append(x); ex_y.append(y)
                else:
                    sh_x.append(x); sh_y.append(y)
            if sh_x:
                ax.scatter(sh_x, sh_y, c='#27ae60', s=60, alpha=0.9,
                           edgecolor='white', linewidth=1.2, zorder=6)
            if ex_x:
                ax.scatter(ex_x, ex_y, c='#e74c3c', s=60, alpha=0.9,
                           edgecolor='white', linewidth=1.2, zorder=7)
        elif boundary_tags is not None:
            # Inside (blue, prominent) vs outside (grey, faded/small)
            in_x, in_y, out_x, out_y = [], [], [], []
            for f, tag in zip(features, boundary_tags):
                x, y = to_3857(f['lon'], f['lat'])
                if tag == 'inside':
                    in_x.append(x); in_y.append(y)
                else:
                    out_x.append(x); out_y.append(y)
            if out_x:
                ax.scatter(out_x, out_y, c='#b0b6c0', s=28, alpha=0.55,
                           edgecolor='white', linewidth=0.8, zorder=6)
            if in_x:
                ax.scatter(in_x, in_y, c='#1f6feb', s=62, alpha=0.92,
                           edgecolor='white', linewidth=1.2, zorder=7)
        else:
            fx, fy = [], []
            for f in features:
                x, y = to_3857(f['lon'], f['lat'])
                fx.append(x); fy.append(y)
            ax.scatter(fx, fy, c=category_color, s=65, alpha=0.9,
                       edgecolor='white', linewidth=1.2, zorder=6)

    # Center marker (red triangle)
    ax.scatter([cx_x], [cx_y], c='#e74c3c', s=180, marker='v',
               edgecolor='white', linewidth=1.8, zorder=8)

    # Clean up axes
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xr = x1 - x0; yr = y1 - y0

    # North arrow (top right)
    nx = x0 + xr * 0.95
    ax.annotate('', xy=(nx, y0 + yr * 0.94),
                xytext=(nx, y0 + yr * 0.85),
                arrowprops=dict(arrowstyle='->', color='#1a1a2e', lw=2.2),
                zorder=10)
    ax.text(nx, y0 + yr * 0.955, 'N', fontsize=11, fontweight='bold',
            color='#1a1a2e', ha='center', va='bottom', zorder=10)

    # Scale bar (bottom left). Mercator scale factor ≈ 1/cos(lat).
    lat_scale          = math.cos(math.radians(lat))
    real_width_m       = xr * lat_scale
    target_bar_m       = real_width_m * 0.20
    nice               = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    bar_real_m         = min(nice, key=lambda v: abs(v - target_bar_m))
    bar_in_plot_units  = bar_real_m / lat_scale
    bx0 = x0 + xr * 0.04
    by  = y0 + yr * 0.05
    ax.plot([bx0, bx0 + bar_in_plot_units], [by, by],
            color='#1a1a2e', linewidth=3.5, solid_capstyle='butt', zorder=10)
    tick_h = yr * 0.008
    ax.plot([bx0, bx0], [by - tick_h, by + tick_h],
            color='#1a1a2e', linewidth=2, zorder=10)
    ax.plot([bx0 + bar_in_plot_units, bx0 + bar_in_plot_units],
            [by - tick_h, by + tick_h],
            color='#1a1a2e', linewidth=2, zorder=10)
    bar_label = (f"{bar_real_m / 1000:.0f} km"
                 if bar_real_m >= 1000 else f"{bar_real_m} m")
    ax.text(bx0 + bar_in_plot_units / 2, by + yr * 0.018, bar_label,
            fontsize=9, color='#1a1a2e', ha='center',
            fontweight='bold', zorder=10)

    # Legend (bottom right)
    if is_exposure:
        line_label = exposure['line_category'].replace('_', ' ').title()
        legend_items = [
            Line2D([0], [0], marker='v', color='w', label='Query center',
                   markerfacecolor='#e74c3c', markeredgecolor='white',
                   markersize=11, markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w',
                   label=f'{category_label} — exposed',
                   markerfacecolor='#e74c3c', markeredgecolor='white',
                   markersize=10, markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w',
                   label=f'{category_label} — shielded',
                   markerfacecolor='#27ae60', markeredgecolor='white',
                   markersize=10, markeredgewidth=1.5),
            Line2D([0], [0], color='#34404f', lw=2.2, label=line_label),
        ]
    elif is_boundary:
        legend_items = [
            Line2D([0], [0], marker='v', color='w', label='Query center',
                   markerfacecolor='#e74c3c', markeredgecolor='white',
                   markersize=11, markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w',
                   label=f'{category_label} — inside',
                   markerfacecolor='#1f6feb', markeredgecolor='white',
                   markersize=10, markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w',
                   label=f'{category_label} — outside',
                   markerfacecolor='#b0b6c0', markeredgecolor='white',
                   markersize=8, markeredgewidth=1.0),
            Line2D([0], [0], marker='s', color='w', label='District boundary',
                   markerfacecolor='none', markeredgecolor='#1f6feb',
                   markersize=11, markeredgewidth=1.8),
        ]
    else:
        legend_items = [
            Line2D([0], [0], marker='v', color='w', label='Query center',
                   markerfacecolor='#e74c3c', markeredgecolor='white',
                   markersize=11, markeredgewidth=1.5),
        ]
        if features:
            legend_items.append(Line2D(
                [0], [0], marker='o', color='w', label=category_label,
                markerfacecolor=category_color, markeredgecolor='white',
                markersize=10, markeredgewidth=1.5,
            ))
        if user_gdf is not None and len(user_gdf) > 0:
            legend_items.append(Line2D(
                [0], [0], marker='s', color='w', label='Uploaded data',
                markerfacecolor=USER_LAYER_FILL, markeredgecolor=USER_LAYER_COLOR,
                markersize=11, markeredgewidth=1.5,
            ))
    ax.legend(handles=legend_items, loc='lower right', framealpha=0.95,
              fontsize=9, facecolor='white', edgecolor='#cccccc', frameon=True)

    # Title above the plot
    if is_exposure:
        line_label = exposure['line_category'].replace('_', ' ')
        title_text = (
            f"{category_label} within {int(exposure['within_m'])} m of "
            f"{line_label} — {location_name[:60]}"
        )
    elif is_boundary:
        title_text = (
            f"{category_label} within district boundary "
            f"({boundary['polygon_count']} area(s)) — {location_name[:55]}"
        )
    else:
        title_text = (
            f"{category_label} within {radius_meters / 1000:.1f} km of "
            f"{location_name[:75]}"
        )
    fig.suptitle(title_text, fontsize=11, color='#1a1a2e',
                 x=0.05, y=0.97, ha='left')

    # Attribution
    fig.text(0.5, 0.02,
             f"© OpenStreetMap contributors  ·  Basemap © CartoDB  ·  Generated by {BRAND_NAME}",
             fontsize=7, color='#888', ha='center')

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])

    fd, png_path = tempfile.mkstemp(suffix='.png', prefix='gis_static_map_')
    os.close(fd)
    plt.savefig(png_path, dpi=STATIC_MAP_DPI, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    return png_path


# ═════════════════════════════════════════
# STEP 6 — PDF report
# ═════════════════════════════════════════
def _latin1(s):
    if s is None:
        return ""
    return str(s).encode("latin-1", "replace").decode("latin-1")


# ── Arabic support (graceful: if libs/font missing, falls back silently) ──
ARABIC_FONT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'NotoNaskhArabic-Regular.ttf',
)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _ARABIC_LIBS = True
except Exception:
    _ARABIC_LIBS = False


def _has_arabic(text):
    """True if the string contains any Arabic-script characters."""
    if not text:
        return False
    for c in str(text):
        if ('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
                or '\u08A0' <= c <= '\u08FF' or '\uFB50' <= c <= '\uFDFF'
                or '\uFE70' <= c <= '\uFEFF'):
            return True
    return False


def _shape_arabic(text):
    """Reshape + reorder Arabic for correct RTL display in the PDF.
    No-op on non-Arabic text or if the libraries aren't available."""
    if _ARABIC_LIBS and _has_arabic(text):
        try:
            return get_display(arabic_reshaper.reshape(str(text)))
        except Exception as e:
            print(f"Arabic shaping failed: {e}")
            return str(text)
    return str(text)


def generate_pdf_report(location_name, category, radius_km, features,
                        user_summary=None, static_map_path=None, exposure=None,
                        notice=None, boundary=None):
    category_label = category.replace('_', ' ').title()
    count = len(features)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # Register the Arabic font if both the file and the reshaping libs exist.
    # If anything's missing, arabic_ok stays False and we fall back to latin-1
    # (so the PDF still renders fine, just with Arabic names as before).
    arabic_ok = False
    if _ARABIC_LIBS and os.path.exists(ARABIC_FONT_PATH):
        try:
            pdf.add_font('NotoArabic', '', ARABIC_FONT_PATH)
            arabic_ok = True
        except Exception as e:
            print(f"Arabic font registration failed: {e}")

    pdf.add_page()

    # ── Branded header: logo (left) + title block (right) ──
    header_y = pdf.get_y()
    logo_drawn_width = 0
    if os.path.exists(LOGO_PATH):
        try:
            # Logo is 550×256 px = aspect ≈ 2.15
            logo_w = 42
            logo_h = logo_w / (550 / 256)  # ≈ 19.5mm
            pdf.image(LOGO_PATH, x=pdf.l_margin, y=header_y, w=logo_w)
            logo_drawn_width = logo_w + 6   # gap after logo
        except Exception as e:
            print(f"PDF logo embed failed: {e}")

    # Title to the right of (or instead of) the logo
    title_x = pdf.l_margin + logo_drawn_width
    pdf.set_xy(title_x, header_y + 2)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*BRAND_NAVY)
    pdf.cell(0, 9, _latin1(f"{BRAND_NAME} - Analysis Report"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(title_x, header_y + 12)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*BRAND_GRAY)
    pdf.cell(0, 5, _latin1("Proximity & overlay analysis"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Drop below whichever is taller — logo block or title block
    pdf.set_y(max(header_y + 22, pdf.get_y() + 2))

    # Thin divider rule
    pdf.set_draw_color(220, 224, 230)
    pdf.set_line_width(0.3)
    y_div = pdf.get_y()
    pdf.line(pdf.l_margin, y_div, pdf.w - pdf.r_margin, y_div)
    pdf.ln(3)

    # ── Notice banner (data caveats — e.g. road fetch failed) ──
    if notice:
        pdf.set_fill_color(255, 244, 229)
        pdf.set_draw_color(240, 194, 119)
        pdf.set_text_color(138, 83, 0)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, _latin1("Note: " + notice),
                       border=1, fill=True,
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(40, 40, 40)
        pdf.ln(3)

    # ── Embedded static map (NEW in V2.5) ──
    if static_map_path and os.path.exists(static_map_path):
        try:
            avail_w = pdf.w - pdf.l_margin - pdf.r_margin
            map_w = avail_w
            map_h = map_w * (STATIC_MAP_H / STATIC_MAP_W) * 0.78  # slight squeeze
            pdf.image(static_map_path, x=pdf.l_margin, y=pdf.get_y(),
                      w=map_w, h=map_h)
            pdf.set_y(pdf.get_y() + map_h + 4)
        except Exception as e:
            print(f"PDF static-map embed failed: {e}")

    # ── Summary block ──
    pdf.set_text_color(*BRAND_NAVY)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, _latin1("Analysis Summary"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    pdf.set_text_color(40, 40, 40)
    summary = [
        ("Location",      location_name),
        ("Category",      category_label),
        ("Search radius", f"{radius_km} km"),
        ("Results found", str(count)),
    ]
    for label, value in summary:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(35, 6, _latin1(label + ":"))
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _latin1(value),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Exposure analysis block (V3.5b) ──
    if exposure is not None:
        line_label = exposure['line_category'].replace('_', ' ').title()
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*BRAND_ACCENT)
        pdf.cell(0, 7, _latin1(
            f"Proximity Exposure  -  vs {line_label}"),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(40, 40, 40)
        exp_rows = [
            ("Buffer distance", f"{int(exposure['within_m'])} m"),
            ("Total analysed",  f"{exposure['total_points']} {category_label.lower()}"),
            ("Exposed (within)", f"{exposure['exposed']}  ({exposure['pct_exposed']}%)"),
            ("Shielded (beyond)", f"{exposure['shielded']}"),
            (f"{line_label} length", f"{exposure['road_length']['km']} km"),
            ("Buffer area", f"{exposure['buffer_area']['km2']} km²  "
                            f"({exposure['buffer_area']['ha']} ha)"),
        ]
        for label, value in exp_rows:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(45, 6, _latin1(label + ":"))
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, _latin1(value),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Within-boundary analysis block (V3.6) ──
    if boundary is not None and not boundary.get('boundary_has_no_polygons'):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*BRAND_ACCENT)
        pdf.cell(0, 7, _latin1("Within-Boundary Analysis"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(40, 40, 40)
        bnd_rows = [
            ("Areas analysed",  f"{boundary['polygon_count']}"),
            ("Total inside",    f"{boundary['total_inside']} of "
                                f"{boundary['total_points']} {category_label.lower()}"),
            ("Outside areas",   f"{boundary['total_outside']}"),
            ("Combined area",   f"{boundary['total_area']['km2']} km²  "
                                f"({boundary['total_area']['ha']} ha)"),
            ("Overall density", f"{boundary['overall_density']} per km²"),
        ]
        for label, value in bnd_rows:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(45, 6, _latin1(label + ":"))
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, _latin1(value),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Per-polygon breakdown table
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(*BRAND_NAVY)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(75, 7, _latin1("Area"), fill=True)
        pdf.cell(30, 7, _latin1("Count"), fill=True)
        pdf.cell(35, 7, _latin1("Area (km2)"), fill=True)
        pdf.cell(0, 7, _latin1("Density /km2"), fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(40, 40, 40)
        bfill = False
        for p in boundary.get('per_polygon', []):
            pdf.set_fill_color(245, 247, 250) if bfill else pdf.set_fill_color(255, 255, 255)
            pname = str(p['name'])[:38]
            if arabic_ok and _has_arabic(pname):
                pdf.set_font("NotoArabic", "", 9)
                pdf.cell(75, 6, _shape_arabic(pname), fill=True)
            else:
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(75, 6, _latin1(pname), fill=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(30, 6, str(p['count']), fill=True)
            pdf.cell(35, 6, f"{p['area']['km2']}", fill=True)
            pdf.cell(0, 6, f"{p['density']}", fill=True,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            bfill = not bfill
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*BRAND_ACCENT)
        pdf.cell(0, 7, _latin1("Uploaded Data Layer"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(40, 40, 40)
        for label, key in [
            ("File",       "filename"),
            ("Features",   "feature_count"),
            ("Geometry",   "geometry_types"),
            ("Source CRS", "original_crs"),
        ]:
            val = user_summary.get(key, '')
            if isinstance(val, list):
                val = ", ".join(map(str, val))
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(35, 6, _latin1(label + ":"))
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, _latin1(str(val)),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    elif user_summary and 'error' in user_summary:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(192, 57, 43)
        pdf.multi_cell(0, 5, _latin1(
            f"Uploaded file '{user_summary.get('filename','')}' could not be "
            f"processed: {user_summary['error']}"
        ))
        pdf.set_text_color(40, 40, 40)

    pdf.ln(3)

    # ── Results table ──
    if count == 0:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*BRAND_GRAY)
        pdf.multi_cell(0, 6, _latin1(
            "No features were found in this area for the selected category."))
        return base64.b64encode(bytes(pdf.output())).decode("utf-8")

    # Build a per-feature status-tag list (aligned by position) if available
    table_tags = None
    status_mode = None   # 'exposure' or 'boundary'
    if exposure is not None:
        tg = exposure.get('_points_gdf')
        if tg is not None and 'exposure' in tg.columns and len(tg) == len(features):
            table_tags = list(tg['exposure'])
            status_mode = 'exposure'
    elif boundary is not None:
        bt = boundary.get('_points_gdf')
        if bt is not None and 'within' in bt.columns and len(bt) == len(features):
            table_tags = list(bt['within'])
            status_mode = 'boundary'

    show_status = table_tags is not None

    # Table header — Status column replaces Phone in exposure/boundary mode
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*BRAND_NAVY)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(10, 7, "#", fill=True)
    pdf.cell(85, 7, _latin1("Name"), fill=True)
    if show_status:
        pdf.cell(45, 7, _latin1("Status"), fill=True)
    else:
        pdf.cell(45, 7, _latin1("Phone"), fill=True)
    pdf.cell(0, 7, _latin1("Coordinates"), fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_text_color(40, 40, 40)
    fill = False
    for i, f in enumerate(features[:PDF_MAX_ROWS], start=1):
        pdf.set_fill_color(245, 247, 250) if fill else pdf.set_fill_color(255, 255, 255)
        coords = f"{f['lat']:.5f}, {f['lon']:.5f}"
        name  = (f.get('name', 'Unnamed') or 'Unnamed')[:42]

        pdf.set_font("Helvetica", "", 8)
        pdf.cell(10, 6, str(i), fill=True)

        # Name cell — use the Arabic font + shaping when the name is Arabic
        if arabic_ok and _has_arabic(name):
            pdf.set_font("NotoArabic", "", 9)
            pdf.cell(85, 6, _shape_arabic(name), fill=True)
            pdf.set_font("Helvetica", "", 8)
        else:
            pdf.cell(85, 6, _latin1(name), fill=True)

        # Status (exposure / boundary) or Phone
        if show_status:
            tag = table_tags[i - 1]
            if status_mode == 'exposure':
                if tag == 'exposed':
                    pdf.set_text_color(192, 57, 43)
                    pdf.cell(45, 6, _latin1("Exposed"), fill=True)
                else:
                    pdf.set_text_color(39, 174, 96)
                    pdf.cell(45, 6, _latin1("Shielded"), fill=True)
            else:  # boundary
                if tag == 'inside':
                    pdf.set_text_color(31, 111, 235)
                    pdf.cell(45, 6, _latin1("Inside"), fill=True)
                else:
                    pdf.set_text_color(140, 146, 160)
                    pdf.cell(45, 6, _latin1("Outside"), fill=True)
            pdf.set_text_color(40, 40, 40)
        else:
            phone = f.get('phone', '') or '-'
            pdf.cell(45, 6, _latin1(phone[:22]), fill=True)

        pdf.cell(0, 6, _latin1(coords), fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        fill = not fill

    if count > PDF_MAX_ROWS:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*BRAND_GRAY)
        pdf.multi_cell(0, 5, _latin1(
            f"... and {count - PDF_MAX_ROWS} more results not listed here."))

    return base64.b64encode(bytes(pdf.output())).decode("utf-8")


# ═════════════════════════════════════════
# Endpoint
# ═════════════════════════════════════════
@app.route('/analyze', methods=['POST'])
def analyze():
    is_multipart = request.content_type and request.content_type.startswith('multipart/form-data')

    if is_multipart:
        location    = (request.form.get('location') or '').strip()
        category    = request.form.get('category') or request.form.get('amenity_type') or 'hospital'
        include_pdf = str(request.form.get('include_pdf', 'false')).lower() in ('true', '1', 'yes')
        try:
            radius_km = float(request.form.get('radius_km', 2))
        except (TypeError, ValueError):
            radius_km = 2.0
        compare_against = (request.form.get('compare_against') or '').strip().lstrip('=').strip()
        try:
            within_m = float(request.form.get('within_m', 500))
        except (TypeError, ValueError):
            within_m = 500.0
        country = (request.form.get('country') or '').strip() or None
        analysis_type = (request.form.get('analysis_type') or '').strip().lstrip('=').strip().lower()
        uploaded = request.files.get('file')
        file_bytes = uploaded.read() if uploaded else b''
        file_name  = uploaded.filename if uploaded else ''
    else:
        data = request.get_json(silent=True) or {}
        location    = (data.get('location') or '').strip()
        category    = data.get('category') or data.get('amenity_type') or 'hospital'
        include_pdf = bool(data.get('include_pdf', False))
        try:
            radius_km = float(data.get('radius_km', 2))
        except (TypeError, ValueError):
            radius_km = 2.0
        compare_against = (data.get('compare_against') or '').strip().lstrip('=').strip()
        try:
            within_m = float(data.get('within_m', 500))
        except (TypeError, ValueError):
            within_m = 500.0
        country = (data.get('country') or '').strip() or None
        analysis_type = (data.get('analysis_type') or '').strip().lower()
        file_b64  = data.get('file_b64') or ''
        file_name = (data.get('file_name') or '').strip()
        file_bytes = base64.b64decode(file_b64) if file_b64 else b''

    radius_km = max(0.1, min(radius_km, 20.0))
    radius_m  = radius_km * 1000

    if not location:
        return jsonify({'success': False, 'error': 'No location provided'}), 400

    lat, lon, full_address = geocode_location(location, country_code=country)
    if lat is None:
        return jsonify({
            'success': False,
            'error': f'Could not find location: {location}',
        }), 404

    features = query_osm(lat, lon, radius_m, category)

    # Parse uploaded file early — it may be the boundary for within-boundary mode
    user_gdf     = None
    user_summary = None
    if file_bytes and file_name:
        try:
            user_gdf_native = parse_uploaded_file(file_bytes, file_name)
            original_crs_str = str(user_gdf_native.crs) if user_gdf_native.crs else 'unknown'
            user_gdf = user_gdf_native
            user_summary = summarize_gdf(user_gdf, file_name, original_crs_str)
        except Exception as e:
            print(f"File ingestion error ({file_name}): {e}")
            user_summary = {'filename': file_name, 'error': str(e)}

    # Decide which analysis to run. Explicit analysis_type wins; otherwise infer.
    exposure = None
    boundary = None
    notice = None

    want_boundary = (analysis_type in ('within-boundary', 'within_boundary',
                                       'boundary', 'within boundary'))
    want_exposure = (analysis_type in ('exposure', 'proximity-exposure')
                     or (not analysis_type and bool(compare_against)))

    # ── Exposure analysis (V3.5b) ──
    if want_exposure and compare_against:
        within_m = max(10.0, min(within_m, 5000.0))   # sane bounds
        try:
            exposure = proximity_exposure_analysis(
                lat, lon, radius_m, category, compare_against, within_m,
                prefetched_points=features,
            )
        except Exception as e:
            import traceback
            print(f"Exposure analysis failed: {e}")
            traceback.print_exc()
            exposure = None
            notice = ("Exposure analysis could not be completed due to an "
                      "internal error. Showing locations only — please retry.")

        if exposure is not None and exposure.get('road_fetch_failed'):
            line_label = compare_against.replace('_', ' ')
            notice = (f"Road data for '{line_label}' could not be retrieved "
                      f"(mapping server timed out). Showing locations only — "
                      f"please retry in a moment for the exposure analysis.")
            exposure = None
        elif exposure is not None and exposure.get('road_count', 0) == 0:
            line_label = compare_against.replace('_', ' ')
            notice = (f"No {line_label} found in this area in OpenStreetMap. "
                      f"All locations are shown as shielded; real exposure may "
                      f"be higher if road data is incomplete here.")

    # ── Within-boundary analysis (V3.6) ──
    elif want_boundary:
        if user_gdf is None or len(user_gdf) == 0:
            notice = ("Within-boundary analysis needs a polygon file. Upload a "
                      "district/area boundary (GeoJSON, Shapefile .zip, or KML) "
                      "and try again. Showing locations only.")
        else:
            try:
                boundary = within_boundary_analysis(
                    lat, lon, radius_m, category, user_gdf,
                    prefetched_points=features,
                )
            except Exception as e:
                import traceback
                print(f"Boundary analysis failed: {e}")
                traceback.print_exc()
                boundary = None
                notice = ("Within-boundary analysis could not be completed due "
                          "to an internal error. Showing locations only.")
            if boundary is not None and boundary.get('boundary_has_no_polygons'):
                notice = ("The uploaded file has no polygon (area) geometry — "
                          "within-boundary analysis needs polygons, not points "
                          "or lines. Showing locations only.")
                boundary = None
            # When boundary mode is active, don't also overlay the raw user_gdf
            # (the boundary IS that file, rendered as the district outline).
            if boundary is not None:
                user_gdf = None

    try:
        map_html = generate_map(
            lat, lon, radius_m, features, location, category,
            user_gdf=user_gdf, user_filename=file_name or None,
            exposure=exposure, notice=notice, boundary=boundary,
        )
    except Exception as e:
        # Last-resort fallback: render the map WITHOUT the user layer rather than 500
        import traceback
        print("Map generation crashed; falling back to OSM-only map.")
        traceback.print_exc()
        map_html = generate_map(
            lat, lon, radius_m, features, location, category,
            user_gdf=None, user_filename=None, exposure=exposure,
            notice=notice, boundary=boundary,
        )
        if user_summary and 'error' not in user_summary:
            user_summary['error'] = f"Could not render layer: {type(e).__name__}"

    result = {
        'success':     True,
        'location':    full_address,
        'category':    category,
        'radius_km':   radius_km,
        'count':       len(features),
        'features':    features,
        'map_html':    map_html,
        'file_summary': user_summary,
    }
    if notice:
        result['notice'] = notice

    # Surface exposure stats in the JSON response (strip internal GDFs)
    if exposure is not None:
        result['exposure'] = {
            k: v for k, v in exposure.items() if not k.startswith('_')
        }

    # Surface boundary stats in the JSON response (strip internal GDFs)
    if boundary is not None:
        result['boundary'] = {
            k: v for k, v in boundary.items() if not k.startswith('_')
        }

    if include_pdf:
        # NEW in V2.5: generate the static cartographic map first,
        # embed it in the PDF, then clean up the temp file.
        static_map_path = None
        try:
            static_map_path = generate_static_map(
                lat, lon, radius_m, features, full_address, category,
                user_gdf=user_gdf, exposure=exposure, boundary=boundary,
            )
        except Exception as e:
            import traceback
            print(f"Static map generation failed: {e}")
            traceback.print_exc()

        try:
            result['pdf_base64'] = generate_pdf_report(
                full_address, category, radius_km, features,
                user_summary=user_summary,
                static_map_path=static_map_path,
                exposure=exposure, notice=notice, boundary=boundary,
            )
        finally:
            if static_map_path and os.path.exists(static_map_path):
                try:
                    os.unlink(static_map_path)
                except OSError:
                    pass

    return jsonify(result)


@app.route('/geocode_check', methods=['GET'])
def geocode_check():
    """V3.5: show what a place name resolves to, with alternatives.
    Prevents silently analyzing the wrong location.

      /geocode_check?location=Al Olaya&country=sa

    Returns the ranked candidates so ambiguity is visible.
    """
    q = (request.args.get('location') or '').strip()
    country = request.args.get('country')  # ISO-2, e.g. 'sa', 'lb'
    if not q:
        return jsonify({'success': False, 'error': 'No location provided'}), 400

    candidates = geocode_location(q, country_code=country, return_candidates=True)
    return jsonify({
        'success': True,
        'query': q,
        'country_bias': country or 'none',
        'candidate_count': len(candidates),
        'candidates': candidates,
        'note': ('More than one match — the analysis would use the first. '
                 'Refine the query or pass &country= to disambiguate.'
                 if len(candidates) > 1 else
                 'Single confident match.' if candidates else
                 'No match found.'),
    })


@app.route('/geoprocess_test', methods=['GET'])
def geoprocess_test():
    """V3.5a self-test — proves the geoprocessing engine works WITHOUT touching
    n8n. Hit this directly in a browser:

      /geoprocess_test?location=Al Olaya, Riyadh&point=school&line=major_roads&within=500&radius=4

    Returns the proximity-exposure analysis as JSON.
    """
    location = (request.args.get('location') or 'Al Olaya, Riyadh, Saudi Arabia').strip()
    point_category = request.args.get('point', 'school').strip()
    line_category  = request.args.get('line', 'major_roads').strip()
    try:
        within_m = float(request.args.get('within', 500))
    except (TypeError, ValueError):
        within_m = 500.0
    try:
        radius_km = float(request.args.get('radius', 4))
    except (TypeError, ValueError):
        radius_km = 4.0
    radius_m = max(0.1, min(radius_km, 20.0)) * 1000

    country = request.args.get('country')  # optional ISO-2 bias, e.g. 'sa'
    lat, lon, full_address = geocode_location(location, country_code=country)
    if lat is None:
        return jsonify({'success': False,
                        'error': f'Could not geocode: {location}'}), 404

    try:
        result = proximity_exposure_analysis(
            lat, lon, radius_m, point_category, line_category, within_m)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'{type(e).__name__}: {e}'}), 500

    # Strip the internal GeoDataFrames before returning JSON
    clean = {k: v for k, v in result.items() if not k.startswith('_')}
    clean['success'] = True
    clean['location'] = full_address
    clean['center'] = {'lat': lat, 'lon': lon}
    clean['metric_crs'] = f"EPSG:{pick_metric_crs(lat, lon)}"

    # Human-readable one-liner — the demo headline
    clean['summary'] = (
        f"{result['exposed']} of {result['total_points']} "
        f"{point_category}(s) ({result['pct_exposed']}%) are within "
        f"{int(within_m)} m of {line_category.replace('_', ' ')}; "
        f"{result['shielded']} are beyond."
    )
    return jsonify(clean)


# ═════════════════════════════════════════════════════════════════════
# AGENT TOOLS (V3.7 — pass 1 of refactor)
#
# Each /tool/* endpoint exposes one geoprocessing primitive as a clean,
# stateless HTTP tool. Inputs and outputs are JSON; geometry is passed
# as GeoJSON FeatureCollection. The agent (n8n + Claude) can compose
# them by passing one tool's output as the next tool's input.
#
# These are ADDITIVE — the existing /analyze endpoint is untouched.
# ═════════════════════════════════════════════════════════════════════

def _gdf_to_geojson(gdf):
    """GeoDataFrame → GeoJSON FeatureCollection dict (WGS84). Empty-safe."""
    if gdf is None or len(gdf) == 0:
        return {"type": "FeatureCollection", "features": []}
    if gdf.crs is not None and str(gdf.crs).upper() != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)
    return json.loads(gdf.to_json())


def _geojson_to_gdf(geojson_obj):
    """GeoJSON FeatureCollection (dict) → GeoDataFrame (WGS84). Tolerates
    a single Feature or a bare geometry too. Raises ValueError on bad input."""
    from shapely.geometry import shape
    if not isinstance(geojson_obj, dict):
        raise ValueError("geojson must be a JSON object")
    t = geojson_obj.get('type')
    if t == 'FeatureCollection':
        feats = geojson_obj.get('features') or []
    elif t == 'Feature':
        feats = [geojson_obj]
    elif t in ('Point', 'MultiPoint', 'LineString', 'MultiLineString',
               'Polygon', 'MultiPolygon', 'GeometryCollection'):
        feats = [{'type': 'Feature', 'geometry': geojson_obj, 'properties': {}}]
    else:
        raise ValueError(f"unrecognized GeoJSON type: {t}")

    rows = []
    for f in feats:
        geom = f.get('geometry')
        if not geom:
            continue
        try:
            shp = shape(geom)
        except Exception as e:
            print(f"_geojson_to_gdf: skipping bad geometry: {e}")
            continue
        props = dict(f.get('properties') or {})
        props['geometry'] = shp
        rows.append(props)
    if not rows:
        return gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
    return gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')


def _centroid_of(gdf):
    """Return (lat, lon) of the centroid of a GeoDataFrame's combined
    geometry — used to pick the right metric CRS for buffering. WGS84 in,
    WGS84 out. Returns (None, None) if empty."""
    if gdf is None or len(gdf) == 0:
        return None, None
    try:
        u = gdf.geometry.union_all()
    except AttributeError:
        u = gdf.geometry.unary_union
    c = u.centroid
    return float(c.y), float(c.x)


# ─── TOOL 1: geocode ────────────────────────────────────────────────
@app.route('/tool/geocode', methods=['POST'])
def tool_geocode():
    """Resolve a free-text place name to coordinates.

    Request JSON:
      { "location": "Al Malaz, Riyadh",
        "country":  "sa",          # optional ISO-2 bias
        "candidates": false }      # if true, return up to 5 candidates

    Response:
      { "success": true,
        "lat": 24.65, "lon": 46.74, "address": "...",
        "handle": "layer_N",       # a drawable 1-point layer for this place
        "candidates": [...]        # only if requested
      }

    The `handle` holds a single Point feature at the resolved location, so a
    caller can draw the place on the map (role "points") for "show me / where
    is X" questions — geocode alone otherwise returns only coordinates.
    """
    data = request.get_json(silent=True) or {}
    location = (data.get('location') or '').strip()
    country  = (data.get('country') or '').strip() or None
    want_cands = bool(data.get('candidates', False))

    if not location:
        return jsonify({'success': False, 'error': 'location is required'}), 400

    def _point_handle(lat, lon, label):
        """Store a single-point FeatureCollection and return its handle."""
        fc = {'type': 'FeatureCollection', 'features': [{
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            'properties': {'name': label, 'type': 'geocoded_place'}
        }]}
        return HANDLES.put(fc, f"location: {label}")

    if want_cands:
        cands = geocode_location(location, country_code=country,
                                 return_candidates=True)
        if not cands:
            return jsonify({'success': False,
                            'error': f"no match for: {location}"}), 404
        top = cands[0]
        hid = _point_handle(top['lat'], top['lon'], top['address'] or location)
        return jsonify({
            'success':    True,
            'lat':        top['lat'],
            'lon':        top['lon'],
            'address':    top['address'],
            'handle':     hid,
            'candidates': cands,
        })

    lat, lon, addr = geocode_location(location, country_code=country)
    if lat is None:
        return jsonify({'success': False,
                        'error': f"no match for: {location}"}), 404
    hid = _point_handle(lat, lon, addr or location)
    return jsonify({'success': True, 'lat': lat, 'lon': lon, 'address': addr,
                    'handle': hid})


# ─── TOOL 2: fetch_osm ──────────────────────────────────────────────
@app.route('/tool/fetch_osm', methods=['POST'])
def tool_fetch_osm():
    """Fetch features from OpenStreetMap within a radius of a coordinate.

    Request JSON:
      { "lat": 24.65, "lon": 46.74,
        "radius_m": 4000,
        "category": "school",
        "kind": "points" | "lines" }    # default: "points"

    For kind='points', `category` is a POINT_CATEGORY_MAP key (school,
    hospital, mosque, pharmacy, restaurant, ...).
    For kind='lines',  `category` is a LINE_CATEGORY_MAP key (major_roads,
    all_roads, railways, rivers).

    Response:
      { "success": true,
        "geojson": { FeatureCollection },
        "count":    N,
        "fetch_ok": true,             // false only on real fetch failure
        "kind":     "points"|"lines"
      }
    """
    data = request.get_json(silent=True) or {}
    try:
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        radius_m = float(data.get('radius_m', 2000))
    except (TypeError, ValueError):
        return jsonify({'success': False,
                        'error': 'lat, lon, radius_m must be numeric'}), 400
    radius_m = max(50.0, min(radius_m, 30000.0))
    category = (data.get('category') or '').strip()
    kind     = (data.get('kind') or 'points').strip().lower()
    if not category:
        return jsonify({'success': False, 'error': 'category is required'}), 400

    if kind == 'points':
        feats = query_osm(lat, lon, radius_m, category)
        gdf = _points_to_gdf(feats)
        # Add other point attributes back in (phone, website, etc.)
        if len(gdf) > 0:
            for i, src in enumerate(feats):
                for key in ('phone', 'website', 'opening_hours', 'type'):
                    if src.get(key):
                        gdf.at[i, key] = src[key]
        return jsonify(_attach_handle({
            'success':  True,
            'kind':     'points',
            'count':    len(gdf),
            'fetch_ok': True,    # query_osm returns [] on both empty + fail
            'geojson':  _gdf_to_geojson(gdf),
        }, summary_override=f"{len(gdf)} {category} point(s)"))
    elif kind == 'lines':
        feats, ok = query_osm_lines(lat, lon, radius_m, category)
        gdf = _lines_to_gdf(feats)
        return jsonify(_attach_handle({
            'success':  True,
            'kind':     'lines',
            'count':    len(gdf),
            'fetch_ok': bool(ok),
            'geojson':  _gdf_to_geojson(gdf),
            'notice':   (None if ok else
                         "OSM data source timed out — retry; the result "
                         "below is empty, not a real zero."),
        }, summary_override=f"{len(gdf)} {category} line(s)"))
    else:
        return jsonify({'success': False,
                        'error': f"kind must be 'points' or 'lines', got '{kind}'"
                        }), 400


# ─── TOOL 3: buffer ─────────────────────────────────────────────────
@app.route('/tool/buffer', methods=['POST'])
def tool_buffer():
    """Buffer a GeoJSON FeatureCollection by N metres in a proper metric CRS.

    Request JSON:
      { "geojson":    { FeatureCollection },
        "distance_m": 500,
        "dissolve":   true }     # union into one polygon (default true)

    The right UTM zone is auto-picked from the input geometry's centroid,
    so buffer distances are real-world metres regardless of input location.

    Response:
      { "success": true,
        "distance_m": 500,
        "dissolve":   true,
        "feature_count": N,
        "area": { "m2":..., "ha":..., "km2":... },
        "metric_crs": "EPSG:32638",
        "geojson": { FeatureCollection of buffered polygons }
      }
    """
    data = request.get_json(silent=True) or {}
    try:
        gj = data.get('geojson')
        distance_m = float(data.get('distance_m'))
    except (TypeError, ValueError):
        return jsonify({'success': False,
                        'error': 'distance_m must be numeric'}), 400
    distance_m = max(1.0, min(distance_m, 50000.0))
    dissolve = bool(data.get('dissolve', True))
    if gj is None:
        return jsonify({'success': False,
                        'error': 'geojson is required'}), 400

    try:
        gj = _resolve_geojson_input(gj)      # accept handle OR inline GeoJSON
        gdf = _geojson_to_gdf(gj)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    if len(gdf) == 0:
        return jsonify({'success': False,
                        'error': 'input geojson has no features'}), 400

    c_lat, c_lon = _centroid_of(gdf)
    buf_wgs, _, area_m2 = buffer_features(gdf, distance_m, c_lat, c_lon,
                                          dissolve=dissolve)
    if buf_wgs is None:
        return jsonify({'success': False,
                        'error': 'buffer operation produced no geometry'}), 500
    area = compute_area_units(area_m2)
    return jsonify(_attach_handle({
        'success':       True,
        'distance_m':    distance_m,
        'dissolve':      dissolve,
        'feature_count': len(buf_wgs),
        'area':          area,
        'metric_crs':    f"EPSG:{pick_metric_crs(c_lat, c_lon)}",
        'geojson':       _gdf_to_geojson(buf_wgs),
    }, summary_override=f"buffer {int(distance_m)}m, {area['km2']} km2"))


# ─── TOOL 4: spatial_join ───────────────────────────────────────────
@app.route('/tool/spatial_join', methods=['POST'])
def tool_spatial_join():
    """For each Point in `points`, tag whether it falls inside any geometry
    of `target_polygons` (typically a buffer, or a district boundary).

    Request JSON:
      { "points":          { FeatureCollection of Point features },
        "target_polygons": { FeatureCollection of Polygon features },
        "inside_label":    "exposed",   // default "inside"
        "outside_label":   "shielded"   // default "outside"
      }

    Response:
      { "success": true,
        "total":   N,
        "inside":  X,        "outside": Y,
        "pct_inside": 53.8,
        "geojson": { FeatureCollection — same as input points, with each
                     feature's properties gaining a "join_status" field
                     equal to inside_label or outside_label }
      }
    """
    data = request.get_json(silent=True) or {}
    pts_gj = data.get('points')
    tgt_gj = data.get('target_polygons')
    inside_label  = (data.get('inside_label') or 'inside').strip() or 'inside'
    outside_label = (data.get('outside_label') or 'outside').strip() or 'outside'

    if pts_gj is None or tgt_gj is None:
        return jsonify({'success': False,
                        'error': 'points and target_polygons are required'
                        }), 400
    try:
        pts_gj = _resolve_geojson_input(pts_gj)
        tgt_gj = _resolve_geojson_input(tgt_gj)
        pts_gdf = _geojson_to_gdf(pts_gj)
        tgt_gdf = _geojson_to_gdf(tgt_gj)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400

    if len(tgt_gdf) == 0:
        return jsonify({'success': False,
                        'error': 'target_polygons has no features'}), 400

    # Union the target polygons into one shape for a single .within() test
    try:
        tgt_union = tgt_gdf.geometry.union_all()
    except AttributeError:
        tgt_union = tgt_gdf.geometry.unary_union

    n_in = 0
    n_out = 0
    if len(pts_gdf) > 0:
        mask = pts_gdf.geometry.within(tgt_union).tolist()
        pts_gdf = pts_gdf.copy()
        pts_gdf['join_status'] = [inside_label if m else outside_label
                                  for m in mask]
        n_in  = int(sum(1 for m in mask if m))
        n_out = int(sum(1 for m in mask if not m))

    total = n_in + n_out
    pct_in = round(100.0 * n_in / total, 1) if total else 0.0

    # Build a clean "inside-only" handle so the UI can draw exactly the inside
    # features when the user asked only for those.
    full_gj = _gdf_to_geojson(pts_gdf)
    inside_hid, inside_n = _store_subset_handle(
        full_gj,
        lambda p: str(p.get('join_status', '')).lower() == str(inside_label).lower(),
        f"{n_in} {inside_label} (inside only)")
    outside_hid, _ = _store_subset_handle(
        full_gj,
        lambda p: str(p.get('join_status', '')).lower() == str(outside_label).lower(),
        f"{n_out} {outside_label} (outside only)")

    resp = _attach_handle({
        'success':    True,
        'total':      total,
        'inside':     n_in,
        'outside':    n_out,
        'pct_inside': pct_in,
        'geojson':    full_gj,
    }, summary_override=f"{n_in} {inside_label}, {n_out} {outside_label} "
                        f"of {total}")
    # expose the subset handles to the agent
    if inside_hid:  resp['inside_handle'] = inside_hid
    if outside_hid: resp['outside_handle'] = outside_hid
    return jsonify(resp)


# ─── TOOL 5: parse_file (Pass 2) ────────────────────────────────────
@app.route('/tool/parse_file', methods=['POST'])
def tool_parse_file():
    """Parse an uploaded geospatial file into GeoJSON.

    Accepts GeoJSON, Esri JSON, Shapefile (.zip), KML/KMZ, CSV with lat/lon
    columns, and GeoPackage. Reprojects to WGS84 (EPSG:4326).

    Two ways to send the file:
      A) Multipart form: field `file` = the file upload
      B) JSON: { "file_b64": "<base64 bytes>", "file_name": "boundary.kmz" }

    Response:
      { "success": true,
        "summary": { filename, feature_count, geometry_types, original_crs, attributes },
        "geojson": { FeatureCollection in WGS84 }
      }
    """
    is_multipart = (request.content_type
                    and request.content_type.startswith('multipart/form-data'))

    if is_multipart:
        uploaded = request.files.get('file')
        if not uploaded:
            return jsonify({'success': False,
                            'error': "no file field in multipart body"}), 400
        file_bytes = uploaded.read()
        file_name  = uploaded.filename or 'upload'
    else:
        data = request.get_json(silent=True) or {}
        b64 = data.get('file_b64') or ''
        file_name = (data.get('file_name') or '').strip() or 'upload'
        if not b64:
            return jsonify({'success': False,
                            'error': "either multipart 'file' or JSON 'file_b64'"
                                     " is required"}), 400
        try:
            file_bytes = base64.b64decode(b64)
        except Exception as e:
            return jsonify({'success': False,
                            'error': f"file_b64 decode failed: {e}"}), 400

    if not file_bytes:
        return jsonify({'success': False, 'error': "empty file"}), 400

    try:
        gdf = parse_uploaded_file(file_bytes, file_name)
    except Exception as e:
        return jsonify({'success': False,
                        'error': f"could not parse file: {type(e).__name__}: {e}"
                        }), 400

    original_crs = str(gdf.crs) if gdf.crs is not None else 'unknown'
    file_sum = summarize_gdf(gdf, file_name, original_crs)
    geomtypes = file_sum.get('geometry_types', []) if isinstance(file_sum, dict) else []
    return jsonify(_attach_handle({
        'success': True,
        'file_summary': file_sum,
        'geojson': _gdf_to_geojson(gdf),
    }, summary_override=f"{file_sum.get('feature_count', len(gdf))} feature(s) "
                        f"from {file_name} ({', '.join(geomtypes) or 'geometry'})"))


# ─── TOOL 6: aggregate_by_polygon (Pass 2) ──────────────────────────
@app.route('/tool/aggregate_by_polygon', methods=['POST'])
def tool_aggregate_by_polygon():
    """Count points inside each polygon, with per-polygon area + density.
    Generalizes V3.6's within-boundary analysis as a standalone tool.

    Request JSON:
      { "points":   { FeatureCollection of Points },
        "polygons": { FeatureCollection of Polygons / MultiPolygons },
        "name_field": "name"   // optional — property to label each polygon by
      }

    For each polygon: counts how many points fall inside, computes its area
    in proper metric units (km²/ha/m²), and density (points per km²).
    A point that's inside multiple overlapping polygons is counted ONLY in
    the first matching polygon (first-match-wins, by input order).

    Response:
      { "success": true,
        "polygon_count": N,
        "total_points":  M,
        "total_inside":  M_in,
        "total_outside": M_out,
        "total_area":    { m2, ha, km2 },
        "overall_density": features-per-km2,
        "per_polygon": [
          { "index", "name", "count", "area": {m2,ha,km2}, "density" }, ...
        ],
        "geojson": { FeatureCollection — same as input points + each gets
                     a "polygon_index" property (-1 if outside any polygon)
                     and a "polygon_name" property }
      }
    """
    data = request.get_json(silent=True) or {}
    pts_gj = data.get('points')
    pol_gj = data.get('polygons')
    name_field = (data.get('name_field') or '').strip() or None
    if pts_gj is None or pol_gj is None:
        return jsonify({'success': False,
                        'error': "'points' and 'polygons' are required"
                        }), 400
    try:
        pts_gj = _resolve_geojson_input(pts_gj)
        pol_gj = _resolve_geojson_input(pol_gj)
        pts_gdf = _geojson_to_gdf(pts_gj)
        pol_gdf = _geojson_to_gdf(pol_gj)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400

    # Keep only polygon geometries
    try:
        keep = pol_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        pol_gdf = pol_gdf[keep].copy().reset_index(drop=True)
    except Exception as e:
        return jsonify({'success': False,
                        'error': f"polygon filter failed: {e}"}), 400
    if len(pol_gdf) == 0:
        return jsonify({'success': False,
                        'error': "no Polygon/MultiPolygon features in 'polygons'"
                        }), 400

    # Pick the best metric CRS based on the polygons' centroid
    c_lat, c_lon = _centroid_of(pol_gdf)
    metric_epsg = pick_metric_crs(c_lat, c_lon)
    pol_metric = pol_gdf.to_crs(epsg=metric_epsg)

    # Find the best label column (explicit name_field wins, else auto-detect)
    label_col = None
    if name_field and name_field in pol_gdf.columns:
        label_col = name_field
    else:
        for c in ('name', 'Name', 'NAME', 'district', 'District', 'DISTRICT',
                  'NAME_EN', 'name_en', 'ADM3_EN', 'label', 'Label'):
            if c in pol_gdf.columns:
                label_col = c
                break

    # Tag each point with its polygon index (-1 = outside all)
    point_tags = [-1] * len(pts_gdf)
    per_poly = []
    for pi in range(len(pol_gdf)):
        poly_geom = pol_gdf.geometry.iloc[pi]
        area_m2 = float(pol_metric.geometry.iloc[pi].area)
        count_in = 0
        if len(pts_gdf) > 0:
            try:
                mask = pts_gdf.geometry.within(poly_geom).tolist()
                for idx, is_in in enumerate(mask):
                    if is_in and point_tags[idx] == -1:
                        point_tags[idx] = pi
                        count_in += 1
            except Exception as e:
                print(f"aggregate_by_polygon: within test failed for #{pi}: {e}")
        area = compute_area_units(area_m2)
        density = round(count_in / area['km2'], 2) if area['km2'] > 0 else 0.0
        label = None
        if label_col is not None:
            try:
                label = str(pol_gdf.iloc[pi][label_col])
            except Exception:
                label = None
        per_poly.append({
            'index':   pi,
            'name':    label or f"Area {pi + 1}",
            'count':   int(count_in),
            'area':    area,
            'density': density,
        })

    total_inside  = sum(p['count'] for p in per_poly)
    total_outside = len(pts_gdf) - total_inside
    total_area_m2 = float(pol_metric.geometry.area.sum())
    total_area    = compute_area_units(total_area_m2)
    overall_density = (round(total_inside / total_area['km2'], 2)
                       if total_area['km2'] > 0 else 0.0)

    # Tag the points GDF with their polygon index + name for the response
    if len(pts_gdf) > 0:
        pts_gdf = pts_gdf.copy()
        pts_gdf['polygon_index'] = point_tags
        pts_gdf['polygon_name']  = [
            per_poly[t]['name'] if t >= 0 else 'outside'
            for t in point_tags
        ]

    full_gj = _gdf_to_geojson(pts_gdf)
    # inside-only handle: points that landed in some polygon (polygon_index >= 0)
    inside_hid, _ = _store_subset_handle(
        full_gj,
        lambda p: isinstance(p.get('polygon_index'), (int, float)) and p.get('polygon_index', -1) >= 0,
        f"{total_inside} inside (boundary only)")

    resp = _attach_handle({
        'success':         True,
        'polygon_count':   int(len(pol_gdf)),
        'total_points':    int(len(pts_gdf)),
        'total_inside':    int(total_inside),
        'total_outside':   int(total_outside),
        'total_area':      total_area,
        'overall_density': overall_density,
        'metric_crs':      f"EPSG:{metric_epsg}",
        'per_polygon':     per_poly,
        'geojson':         full_gj,
    }, summary_override=f"{total_inside} inside / {total_outside} outside across "
                        f"{len(pol_gdf)} area(s)")
    if inside_hid: resp['inside_handle'] = inside_hid
    return jsonify(resp)


# ─── TOOL 7: compute_metrics (Pass 2) ───────────────────────────────
@app.route('/tool/compute_metrics', methods=['POST'])
def tool_compute_metrics():
    """Compute area (for polygons) and/or length (for lines) of any GeoJSON,
    in proper metric units. The metric CRS is auto-picked from the geometry's
    centroid so values are real-world accurate.

    Request JSON:
      { "geojson": { FeatureCollection } }

    Response:
      { "success": true,
        "feature_count": N,
        "geometry_types": ["Polygon","LineString",...],
        "metric_crs": "EPSG:32638",
        "area":   { "m2", "ha", "km2" },        // 0 if no polygons
        "length": { "m",  "km" }                 // 0 if no lines
      }
    """
    data = request.get_json(silent=True) or {}
    gj = data.get('geojson')
    if gj is None:
        return jsonify({'success': False, 'error': "'geojson' is required"}), 400
    try:
        gj = _resolve_geojson_input(gj)
        gdf = _geojson_to_gdf(gj)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400

    if len(gdf) == 0:
        return jsonify({
            'success':       True,
            'feature_count': 0,
            'geometry_types': [],
            'area':          {'m2': 0.0, 'ha': 0.0, 'km2': 0.0},
            'length':        {'m': 0.0, 'km': 0.0},
        })

    c_lat, c_lon = _centroid_of(gdf)
    metric_epsg = pick_metric_crs(c_lat, c_lon)
    geom_types = sorted(set(gdf.geometry.geom_type.dropna().tolist()))

    # Area: sum polygon areas only
    poly_mask = gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
    if poly_mask.any():
        poly_metric = gdf[poly_mask].to_crs(epsg=metric_epsg)
        area_m2 = float(poly_metric.geometry.area.sum())
    else:
        area_m2 = 0.0

    # Length: sum line lengths only
    line_mask = gdf.geometry.geom_type.isin(['LineString', 'MultiLineString'])
    if line_mask.any():
        lines_gdf = gdf[line_mask].copy()
        length = compute_length_units(lines_gdf, c_lat, c_lon)
    else:
        length = {'m': 0.0, 'km': 0.0}

    return jsonify({
        'success':       True,
        'feature_count': int(len(gdf)),
        'geometry_types': geom_types,
        'metric_crs':    f"EPSG:{metric_epsg}",
        'area':          compute_area_units(area_m2),
        'length':        length,
    })


# ═════════════════════════════════════════════════════════════════════
# DRIVE-TIME / ISOCHRONE (V3.7c — first NEW primitive)
#
# OSRM doesn't return isochrones directly, so we:
#   1. sample destination points in rings around the origin
#   2. ask OSRM /table for the drive-time origin→each sample (one call)
#   3. keep samples reachable within the time threshold
#   4. wrap them in a concave hull → the isochrone polygon
# ═════════════════════════════════════════════════════════════════════
import math as _math


def _destination_point(lat, lon, bearing_deg, distance_m):
    """Forward geodesy: point at a bearing+distance from origin (haversine)."""
    R = 6371000.0
    br = _math.radians(bearing_deg)
    lat1 = _math.radians(lat)
    lon1 = _math.radians(lon)
    dr = distance_m / R
    lat2 = _math.asin(_math.sin(lat1) * _math.cos(dr) +
                      _math.cos(lat1) * _math.sin(dr) * _math.cos(br))
    lon2 = lon1 + _math.atan2(
        _math.sin(br) * _math.sin(dr) * _math.cos(lat1),
        _math.cos(dr) - _math.sin(lat1) * _math.sin(lat2))
    return _math.degrees(lat2), _math.degrees(lon2)


def _isochrone_polygon(reachable_lonlat):
    """Build a polygon from reachable (lon,lat) points. Concave hull for a
    realistic shape; convex hull fallback; None if too few points."""
    from shapely.geometry import MultiPoint
    if len(reachable_lonlat) < 3:
        return None
    mp = MultiPoint(reachable_lonlat)
    try:
        poly = mp.concave_hull(ratio=0.4)   # 0=tight, 1=convex
        if poly is None or poly.is_empty or poly.geom_type not in (
                'Polygon', 'MultiPolygon'):
            poly = mp.convex_hull
    except Exception as e:
        print(f"isochrone: concave_hull failed ({e}); using convex hull")
        poly = mp.convex_hull
    if poly.geom_type not in ('Polygon', 'MultiPolygon'):
        return None
    return poly


@app.route('/tool/drive_time_area', methods=['POST'])
def tool_drive_time_area():
    """Compute a drive-time isochrone: the area reachable from a point within
    N minutes by car. Returns the reachable area as a GeoJSON polygon.

    Request JSON:
      { "lat": 24.6877, "lon": 46.7219,
        "minutes": 15,
        "max_reach_km": 12 }     # optional; how far to sample (auto if omitted)

    Method: samples points in rings around the origin, asks OSRM for the
    real driving time to each, keeps those within `minutes`, and wraps them
    in a polygon.

    Response:
      { "success": true,
        "minutes": 15,
        "origin": { "lat":..., "lon":... },
        "reachable_count": K,        // sample points within the threshold
        "sampled_count":   N,
        "area": { "m2","ha","km2" }, // size of the reachable area
        "geojson": { FeatureCollection with one isochrone Polygon },
        "notice": "..."              // present if degraded/failed
      }

    On routing-service failure, falls back to a circular estimate (straight-
    line, using an average urban speed) and says so in `notice` — never
    silently returns a wrong shape as if it were real routing.
    """
    data = request.get_json(silent=True) or {}
    try:
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        minutes = float(data.get('minutes', 15))
    except (TypeError, ValueError):
        return jsonify({'success': False,
                        'error': 'lat, lon, minutes must be numeric'}), 400
    minutes = max(1.0, min(minutes, 60.0))

    # How far to sample. Assume up to ~80 km/h urban arterial as the ceiling,
    # so max straight-line reach ≈ minutes/60 * 80 km, capped for sanity.
    auto_reach_km = min((minutes / 60.0) * 80.0, 40.0)
    try:
        max_reach_km = float(data.get('max_reach_km', auto_reach_km))
    except (TypeError, ValueError):
        max_reach_km = auto_reach_km
    max_reach_m = max(500.0, min(max_reach_km * 1000.0, 50000.0))

    # 1. Build sample ring (16 bearings × 6 rings = 96 pts; +origin = 97 ≤ 100)
    bearings = [b * (360.0 / 16) for b in range(16)]
    ring_fracs = [0.18, 0.36, 0.55, 0.72, 0.86, 1.0]
    samples = []  # (lat, lon)
    for fr in ring_fracs:
        d = max_reach_m * fr
        for b in bearings:
            samples.append(_destination_point(lat, lon, b, d))

    # 2. OSRM /table: durations from origin (index 0) to all samples
    #    coords are lon,lat;lon,lat;...  sources=0  destinations=1..N
    coord_str = f"{lon},{lat};" + ";".join(f"{s[1]},{s[0]}" for s in samples)
    dest_idx = ";".join(str(i) for i in range(1, len(samples) + 1))
    url = (f"{OSRM_BASE}/table/v1/driving/{coord_str}"
           f"?sources=0&destinations={dest_idx}&annotations=duration")

    durations = None
    try:
        resp = requests.get(url, timeout=30,
                            headers={'User-Agent': 'gis_agent_v3/1.0'})
        if resp.status_code == 200:
            body = resp.json()
            if body.get('code') == 'Ok' and body.get('durations'):
                durations = body['durations'][0]  # row 0 = from origin
            else:
                print(f"OSRM table: unexpected body code={body.get('code')}")
        else:
            print(f"OSRM table HTTP {resp.status_code}: {resp.text[:150]}")
    except Exception as e:
        print(f"OSRM table error: {e}")

    threshold_s = minutes * 60.0

    # 3a. Routing worked → keep reachable samples, build concave hull
    if durations is not None:
        reachable = []
        for (slat, slon), dur in zip(samples, durations):
            if dur is not None and dur <= threshold_s:
                reachable.append((slon, slat))
        # Always include the origin itself
        reachable.append((lon, lat))
        poly = _isochrone_polygon(reachable)
        notice = None
        method = 'osrm_driving'
        if poly is None:
            # Too few reachable points to form an area
            return jsonify({
                'success': True,
                'minutes': minutes,
                'origin': {'lat': lat, 'lon': lon},
                'reachable_count': len(reachable),
                'sampled_count': len(samples),
                'area': {'m2': 0.0, 'ha': 0.0, 'km2': 0.0},
                'geojson': {'type': 'FeatureCollection', 'features': []},
                'notice': ('Very few points reachable within the time limit — '
                           'the area may be too small to map, or routing data '
                           'is sparse here.'),
                'method': method,
            })
    # 3b. Routing failed → honest circular fallback (straight-line estimate)
    else:
        avg_kmh = 35.0   # conservative urban average incl. stops/turns
        radius_m = (minutes / 60.0) * avg_kmh * 1000.0
        circle_pts = []
        for b in range(0, 360, 10):
            p = _destination_point(lat, lon, b, radius_m)
            circle_pts.append((p[1], p[0]))
        poly = _isochrone_polygon(circle_pts)
        reachable = circle_pts
        notice = (f"Routing service unavailable — showing a rough circular "
                  f"estimate assuming ~{int(avg_kmh)} km/h average, NOT real "
                  f"road-network drive time. Retry for an accurate isochrone.")
        method = 'circular_estimate'

    # 4. Area of the polygon in proper metric CRS
    poly_gdf = gpd.GeoDataFrame({'geometry': [poly]}, crs='EPSG:4326')
    metric_epsg = pick_metric_crs(lat, lon)
    area_m2 = float(poly_gdf.to_crs(epsg=metric_epsg).geometry.area.iloc[0])

    area = compute_area_units(area_m2)
    return jsonify(_attach_handle({
        'success':         True,
        'minutes':         minutes,
        'origin':          {'lat': lat, 'lon': lon},
        'reachable_count': len(reachable),
        'sampled_count':   len(samples),
        'area':            area,
        'metric_crs':      f"EPSG:{metric_epsg}",
        'method':          method,
        'geojson':         _gdf_to_geojson(poly_gdf),
        'notice':          notice,
    }, summary_override=f"{int(minutes)}-min drive area, {area['km2']} km2"))


# ═════════════════════════════════════════════════════════════════════
# ATTRIBUTE FILTER (V3.7d — second NEW primitive)
#
# Keep only the features whose PROPERTIES match one or more conditions.
# This is what turns "schools" into "schools with capacity > 500" or
# "parcels" into "vacant parcels larger than 20,000 m²". Pure data op,
# no geometry math — works on any GeoJSON regardless of source.
# ═════════════════════════════════════════════════════════════════════
def _coerce_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _eval_condition(props, cond):
    """Evaluate one condition dict against a feature's properties.
    cond = { "field": "...", "op": "...", "value": ... }"""
    field = cond.get('field')
    op    = (cond.get('op') or '=').lower().strip()
    val   = cond.get('value')
    actual = props.get(field)

    if op in ('exists', 'has'):
        return actual is not None and actual != ''
    if op in ('missing', 'is_empty', 'empty'):
        return actual is None or actual == ''

    # Numeric-or-string comparison operators
    if op in ('>', '>=', '<', '<=', '=', '==', '!=',
              'eq', 'ne', 'gt', 'lt', 'gte', 'lte'):
        a = _coerce_num(actual)
        b = _coerce_num(val)
        if a is not None and b is not None:           # numeric path
            if op in ('>', 'gt'):    return a > b
            if op in ('>=', 'gte'):  return a >= b
            if op in ('<', 'lt'):    return a < b
            if op in ('<=', 'lte'):  return a <= b
            if op in ('=', '==', 'eq'): return a == b
            if op in ('!=', 'ne'):   return a != b
        # string fallback for equality ops
        if op in ('=', '==', 'eq'):  return str(actual) == str(val)
        if op in ('!=', 'ne'):       return str(actual) != str(val)
        return False

    if op == 'contains':
        return actual is not None and str(val).lower() in str(actual).lower()
    if op in ('not_contains', 'excludes'):
        return actual is None or str(val).lower() not in str(actual).lower()
    if op == 'equals':      return str(actual) == str(val)
    if op == 'not_equals':  return str(actual) != str(val)
    if op == 'in':
        vals = val if isinstance(val, list) else [val]
        return actual in vals or str(actual) in [str(x) for x in vals]
    if op == 'not_in':
        vals = val if isinstance(val, list) else [val]
        return actual not in vals and str(actual) not in [str(x) for x in vals]
    return False


@app.route('/tool/filter_features', methods=['POST'])
def tool_filter_features():
    """Keep only features whose properties satisfy the given condition(s).

    Request JSON:
      { "geojson": { FeatureCollection },
        "conditions": [
          { "field": "landuse",  "op": "equals", "value": "vacant" },
          { "field": "area_m2",  "op": ">",      "value": 20000 }
        ],
        "combine": "and"   // "and" (all must pass) | "or" (any). default "and"
      }

    Supported ops:
      Numeric/compare:  >  >=  <  <=  =  !=
      Text:             equals  not_equals  contains  not_contains  in  not_in
      Existence:        exists  missing

    Numeric ops auto-coerce strings like "20000" → 20000 so they work on
    real-world data where numbers are often stored as text.

    Response:
      { "success": true,
        "input_count":  N,
        "kept_count":   K,
        "removed_count": N-K,
        "geojson": { FeatureCollection of the kept features }
      }
    """
    data = request.get_json(silent=True) or {}
    gj = data.get('geojson')
    conditions = data.get('conditions') or []
    combine = (data.get('combine') or 'and').lower().strip()
    if combine not in ('and', 'or'):
        combine = 'and'
    if gj is None:
        return jsonify({'success': False, 'error': "'geojson' is required"}), 400
    # Accept a handle string OR inline GeoJSON
    try:
        gj = _resolve_geojson_input(gj)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    if not isinstance(conditions, list) or not conditions:
        return jsonify({'success': False,
                        'error': "'conditions' must be a non-empty list"}), 400

    # Validate condition shape
    for c in conditions:
        if not isinstance(c, dict) or 'field' not in c:
            return jsonify({'success': False,
                            'error': "each condition needs at least a 'field'"
                            }), 400

    # Work directly on the GeoJSON features (no geopandas needed — pure
    # property filtering, preserves geometry untouched).
    if not isinstance(gj, dict):
        return jsonify({'success': False,
                        'error': 'geojson must be a JSON object'}), 400
    t = gj.get('type')
    if t == 'FeatureCollection':
        in_feats = gj.get('features') or []
    elif t == 'Feature':
        in_feats = [gj]
    else:
        return jsonify({'success': False,
                        'error': 'geojson must be a Feature or FeatureCollection'
                        }), 400

    kept = []
    for f in in_feats:
        props = f.get('properties') or {}
        results = [_eval_condition(props, c) for c in conditions]
        passed = all(results) if combine == 'and' else any(results)
        if passed:
            kept.append(f)

    return jsonify(_attach_handle({
        'success':       True,
        'input_count':   len(in_feats),
        'kept_count':    len(kept),
        'removed_count': len(in_feats) - len(kept),
        'combine':       combine,
        'geojson':       {'type': 'FeatureCollection', 'features': kept},
    }, summary_override=f"{len(kept)} of {len(in_feats)} feature(s) kept"))


# ═════════════════════════════════════════════════════════════════════
# COMBINE AREAS (V3.7e — third NEW primitive: multi-criteria geometry)
#
# Set operations on polygon areas — the geometry behind site selection:
#   intersect  (AND)     overlap of all inputs       "near road AND in zone"
#   union      (OR)      everything covered          "near school A OR B"
#   difference (AND NOT) area_a minus the others     "near road but NOT in flood"
#
# Lets the agent build complex suitability areas, then feed the result to
# spatial_join / aggregate_by_polygon to count what's inside.
# ═════════════════════════════════════════════════════════════════════
@app.route('/tool/combine_areas', methods=['POST'])
def tool_combine_areas():
    """Combine polygon areas with a set operation.

    Request JSON:
      { "op": "intersect" | "union" | "difference",
        "area_a": { FeatureCollection of polygons },
        "area_b": { FeatureCollection of polygons },   // not needed for union-of-one
        "areas":  [ {FC}, {FC}, ... ]   // alternative: 2+ areas for intersect/union
      }

    - intersect: the region common to ALL inputs (area_a ∩ area_b ∩ ...).
    - union:     the region covered by ANY input (area_a ∪ area_b ∪ ...).
    - difference: area_a with area_b (and any further areas) removed
                  (area_a − area_b − ...). Order matters: area_a is the base.

    Provide either area_a + area_b, OR an `areas` list of 2+ FeatureCollections.

    Response:
      { "success": true,
        "op": "...",
        "is_empty": false,           // true if the result is nothing (e.g. no overlap)
        "area": { "m2","ha","km2" },
        "metric_crs": "EPSG:32638",
        "geojson": { FeatureCollection with the resulting polygon(s) }
      }
    """
    data = request.get_json(silent=True) or {}
    op = (data.get('op') or '').lower().strip()
    if op not in ('intersect', 'union', 'difference'):
        return jsonify({'success': False,
                        'error': "op must be 'intersect', 'union', or 'difference'"
                        }), 400

    # Gather input areas as a list of GeoDataFrames
    raw_areas = data.get('areas')
    if not raw_areas:
        a = data.get('area_a')
        b = data.get('area_b')
        raw_areas = [x for x in (a, b) if x is not None]
    if not raw_areas or len(raw_areas) < 1:
        return jsonify({'success': False,
                        'error': "provide area_a (+area_b) or an 'areas' list"
                        }), 400
    if op in ('intersect', 'difference') and len(raw_areas) < 2:
        return jsonify({'success': False,
                        'error': f"'{op}' needs at least 2 areas"}), 400

    # Parse each to a single (unioned) polygon geometry
    geoms = []
    for i, fc in enumerate(raw_areas):
        try:
            fc = _resolve_geojson_input(fc)   # handle OR inline GeoJSON
            gdf = _geojson_to_gdf(fc)
        except ValueError as e:
            return jsonify({'success': False,
                            'error': f"area #{i + 1}: {e}"}), 400
        # keep only polygonal geometry
        try:
            poly_mask = gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
            gdf = gdf[poly_mask]
        except Exception:
            gdf = gdf
        if len(gdf) == 0:
            return jsonify({'success': False,
                            'error': f"area #{i + 1} has no polygon geometry"
                            }), 400
        try:
            merged = gdf.geometry.union_all()
        except AttributeError:
            merged = gdf.geometry.unary_union
        geoms.append(merged)

    # Apply the set operation
    from functools import reduce
    try:
        if op == 'union':
            result = reduce(lambda x, y: x.union(y), geoms)
        elif op == 'intersect':
            result = reduce(lambda x, y: x.intersection(y), geoms)
        else:  # difference: first minus the rest
            result = geoms[0]
            for g in geoms[1:]:
                result = result.difference(g)
    except Exception as e:
        import traceback
        print(f"combine_areas {op} failed: {e}")
        traceback.print_exc()
        return jsonify({'success': False,
                        'error': f"geometry operation failed: {type(e).__name__}"
                        }), 500

    is_empty = (result is None or result.is_empty)

    # Build response GeoDataFrame + area
    if is_empty:
        return jsonify({
            'success':   True,
            'op':        op,
            'is_empty':  True,
            'area':      {'m2': 0.0, 'ha': 0.0, 'km2': 0.0},
            'geojson':   {'type': 'FeatureCollection', 'features': []},
            'notice':    ('The operation produced an empty area '
                          + ('(the inputs do not overlap).'
                             if op == 'intersect'
                             else '(nothing remained after subtraction).')),
        })

    result_gdf = gpd.GeoDataFrame({'geometry': [result]}, crs='EPSG:4326')
    c_lat, c_lon = _centroid_of(result_gdf)
    metric_epsg = pick_metric_crs(c_lat, c_lon)
    area_m2 = float(result_gdf.to_crs(epsg=metric_epsg).geometry.area.iloc[0])

    area = compute_area_units(area_m2)
    return jsonify(_attach_handle({
        'success':    True,
        'op':         op,
        'is_empty':   False,
        'area':       area,
        'metric_crs': f"EPSG:{metric_epsg}",
        'geojson':    _gdf_to_geojson(result_gdf),
    }, summary_override=f"{op} result, {area['km2']} km2"))


@app.route('/handle/<hid>', methods=['GET'])
def get_handle(hid):
    """Retrieve the GeoJSON stored under a handle. Used by the final render
    step (n8n / the agent) to pull a layer's geometry by its handle.
    Returns 404 if the handle is unknown or expired."""
    gj = HANDLES.get(hid)
    if gj is None:
        return jsonify({'success': False,
                        'error': f'unknown or expired handle: {hid}'}), 404
    return jsonify({
        'success': True,
        'handle':  hid,
        'summary': HANDLES.summary(hid),
        'geojson': gj,
    })


# ═════════════════════════════════════════════════════════════════════
# SAUDI DISTRICT CONNECTOR (V3.8 — first authoritative-data connector)
#
# Resolves a Riyadh district name (Arabic or English) to its authoritative
# boundary polygon, from a compact bundled dataset derived from Saudi
# National Address data (maps.address.gov.sa). Loads the boundary into the
# handle store so the agent can analyse against the REAL district — no upload.
#
# Data: riyadh_districts.geojson (189 Riyadh city districts), bundled in repo.
# ═════════════════════════════════════════════════════════════════════
import re as _re

_DISTRICTS_CACHE = None   # lazy-loaded list of district features


def _load_districts():
    global _DISTRICTS_CACHE
    if _DISTRICTS_CACHE is not None:
        return _DISTRICTS_CACHE
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'riyadh_districts.geojson')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _DISTRICTS_CACHE = data.get('features', [])
    except Exception as e:
        print(f"saudi_district: could not load dataset: {e}")
        _DISTRICTS_CACHE = []
    return _DISTRICTS_CACHE


def _normalize_district_name(s):
    """Match the normalization used when the dataset was built."""
    s = (s or '').lower().strip()
    s = _re.sub(r'\bdistrict\b', '', s)
    s = _re.sub(r'\bdist\b', '', s)
    s = _re.sub(r'^حي\s+', '', s)
    s = _re.sub(r'[.\u060c,]', ' ', s)
    return _re.sub(r'\s+', ' ', s).strip()


@app.route('/tool/saudi_district', methods=['POST'])
def tool_saudi_district():
    """Look up a Riyadh district's authoritative boundary by name (Arabic or
    English) and load it as a layer. Use this INSTEAD of asking the user to
    upload a boundary when they name a Riyadh district.

    Request JSON:
      { "name": "Al Malaz" }        # Arabic or English, with/without "District"

    Response (match found):
      { "success": true,
        "matched": "Al Malaz Dist.",
        "name_ar": "الملز", "name_en": "Al Malaz Dist.",
        "handle": "layer_N",          # the district boundary polygon
        "area": { m2, ha, km2 },
        "source": "Saudi National Address (maps.address.gov.sa)"
      }

    Response (no exact match): returns success:false with up to 8 suggestions.
    """
    data = request.get_json(silent=True) or {}
    query = (data.get('name') or '').strip()
    if not query:
        return jsonify({'success': False, 'error': "'name' is required"}), 400

    feats = _load_districts()
    if not feats:
        return jsonify({'success': False,
                        'error': 'district dataset unavailable'}), 500

    q = _normalize_district_name(query)

    # 1. exact match on english or arabic normalized key
    match = None
    for f in feats:
        p = f['properties']
        if p.get('match_en') == q or p.get('match_ar') == q:
            match = f
            break

    # 2. fallback: contains match
    if match is None:
        for f in feats:
            p = f['properties']
            if q and (q in p.get('match_en', '') or q in p.get('match_ar', '')):
                match = f
                break

    if match is None:
        # suggestions: districts whose name shares any word with the query
        qwords = set(q.split())
        sugg = []
        for f in feats:
            p = f['properties']
            mwords = set(p.get('match_en', '').split())
            if qwords & mwords:
                sugg.append(p.get('name_en'))
            if len(sugg) >= 8:
                break
        return jsonify({
            'success': False,
            'error': f"no Riyadh district matched '{query}'",
            'suggestions': sugg,
        }), 404

    # Build the boundary layer + handle
    p = match['properties']
    label = p.get('name_en') or p.get('name_ar') or query
    fc = {'type': 'FeatureCollection', 'features': [{
        'type': 'Feature',
        'geometry': match['geometry'],
        'properties': {'name': label, 'name_ar': p.get('name_ar'),
                       'type': 'saudi_district'}
    }]}
    hid = HANDLES.put(fc, f"district boundary: {label}")

    # area
    try:
        gdf = _geojson_to_gdf(fc)
        c_lat, c_lon = _centroid_of(gdf)
        area_m2 = float(gdf.to_crs(epsg=pick_metric_crs(c_lat, c_lon))
                        .geometry.area.iloc[0])
        area = compute_area_units(area_m2)
    except Exception as e:
        print(f"saudi_district area calc failed: {e}")
        area = None

    return jsonify({
        'success':  True,
        'matched':  label,
        'name_ar':  p.get('name_ar'),
        'name_en':  p.get('name_en'),
        'handle':   hid,
        'area':     area,
        'source':   'Saudi National Address (maps.address.gov.sa)',
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'GIS Agent Backend running ✅ (V3.8 — Saudi district connector: authoritative Riyadh boundaries)'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
