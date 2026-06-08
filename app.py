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
PDF_MAX_ROWS = 100

MAX_FILE_SIZE_MB = 5
USER_LAYER_COLOR = '#1f6feb'
USER_LAYER_FILL  = '#3b82f6'


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
                 user_gdf=None, user_filename=None, exposure=None, notice=None):
    m = folium.Map(location=[lat, lon], zoom_start=15, tiles='CartoDB positron')
    color = STYLE_MAP.get(category, DEFAULT_COLOR)
    category_label = category.replace('_', ' ').title()
    count = len(features)

    badge_color = '#27ae60' if count > 0 else '#95a5a6'
    badge_icon  = '✅' if count > 0 else 'ℹ️'
    plural      = 's' if count != 1 else ''

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
        folium.CircleMarker(
            [f['lat'], f['lon']],
            radius=9,
            color=pt_color, fill=True, fill_color=pt_color, fill_opacity=0.85,
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
    else:
        badge_html = f"""
        <div style='background:{badge_color};color:white;border-radius:8px;
                    padding:8px;text-align:center;font-weight:700'>
            {badge_icon} {count} {category_label}{plural} Found
        </div>
        """
        accent = color

    search_line = (
        f"📏 within {int(exposure['within_m'])} m of "
        f"{exposure['line_category'].replace('_', ' ')}"
        if is_exposure else
        f"📏 {radius_meters / 1000:.1f} km radius"
    )

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


# ═════════════════════════════════════════
# STEP 6 (NEW in V2.5) — Static cartographic map
# Print-quality PNG with basemap, OSM features, user layer,
# radius, north arrow, scale bar, legend, attribution.
# Returns a path to a temp PNG (caller must delete after use).
# ═════════════════════════════════════════
def generate_static_map(lat, lon, radius_meters, features, location_name,
                        category, user_gdf=None, exposure=None):
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
                        notice=None):
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

    if user_summary and 'error' not in user_summary:
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

    # Build a per-feature exposure-tag list (aligned by position) if available
    table_tags = None
    if exposure is not None:
        tg = exposure.get('_points_gdf')
        if tg is not None and 'exposure' in tg.columns and len(tg) == len(features):
            table_tags = list(tg['exposure'])

    show_status = table_tags is not None

    # Table header — Status column replaces Phone in exposure mode
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

        # Status (exposure) or Phone
        if show_status:
            tag = table_tags[i - 1]
            if tag == 'exposed':
                pdf.set_text_color(192, 57, 43)
                pdf.cell(45, 6, _latin1("Exposed"), fill=True)
            else:
                pdf.set_text_color(39, 174, 96)
                pdf.cell(45, 6, _latin1("Shielded"), fill=True)
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

    # ── Exposure analysis (V3.5b) — only when a comparison layer is given ──
    exposure = None
    notice = None
    if compare_against:
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

        # Road fetch failed across all mirrors → don't present a misleading
        # "0% exposed". Fall back to a plain proximity map + honest notice.
        if exposure is not None and exposure.get('road_fetch_failed'):
            line_label = compare_against.replace('_', ' ')
            notice = (f"Road data for '{line_label}' could not be retrieved "
                      f"(mapping server timed out). Showing locations only — "
                      f"please retry in a moment for the exposure analysis.")
            exposure = None
        # Genuine data gap: fetch worked but the area has no such roads in OSM
        elif exposure is not None and exposure.get('road_count', 0) == 0:
            line_label = compare_against.replace('_', ' ')
            notice = (f"No {line_label} found in this area in OpenStreetMap. "
                      f"All locations are shown as shielded; real exposure may "
                      f"be higher if road data is incomplete here.")

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

    try:
        map_html = generate_map(
            lat, lon, radius_m, features, location, category,
            user_gdf=user_gdf, user_filename=file_name or None,
            exposure=exposure, notice=notice,
        )
    except Exception as e:
        # Last-resort fallback: render the map WITHOUT the user layer rather than 500
        import traceback
        print("Map generation crashed; falling back to OSM-only map.")
        traceback.print_exc()
        map_html = generate_map(
            lat, lon, radius_m, features, location, category,
            user_gdf=None, user_filename=None, exposure=exposure, notice=notice,
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

    if include_pdf:
        # NEW in V2.5: generate the static cartographic map first,
        # embed it in the PDF, then clean up the temp file.
        static_map_path = None
        try:
            static_map_path = generate_static_map(
                lat, lon, radius_m, features, full_address, category,
                user_gdf=user_gdf, exposure=exposure,
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
                exposure=exposure, notice=notice,
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


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'GIS Agent Backend running ✅ (V3.5d — Status column, Arabic-ready PDF, rebrand)'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
