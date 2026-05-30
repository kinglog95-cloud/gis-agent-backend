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
PDF_MAX_ROWS = 100

MAX_FILE_SIZE_MB = 5
USER_LAYER_COLOR = '#1f6feb'
USER_LAYER_FILL  = '#3b82f6'

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
# STEP 1 — Geocode
# ═════════════════════════════════════════
def geocode_location(location_str, retries=2):
    geolocator = Nominatim(user_agent="gis_agent_v2/1.0")
    for attempt in range(retries + 1):
        try:
            loc = geolocator.geocode(location_str, timeout=10, language='en')
            if loc:
                return loc.latitude, loc.longitude, loc.address
            return None, None, None
        except (GeocoderTimedOut, GeocoderServiceError):
            if attempt < retries:
                time.sleep(1.5)
                continue
            return None, None, None


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

    headers = {
        'User-Agent': 'gis_agent_v2/1.0 (GIS analysis tool)',
        'Accept':     'application/json',
    }

    try:
        response = requests.post(
            OVERPASS_URL, data={'data': query}, headers=headers, timeout=60,
        )
        if response.status_code != 200:
            print(f"Overpass HTTP {response.status_code}: {response.text[:200]}")
            return []

        data = response.json()
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

    except requests.exceptions.Timeout:
        print("Overpass error: request timed out")
        return []
    except Exception as e:
        print(f"Overpass error: {e}")
        return []


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
                 user_gdf=None, user_filename=None):
    m = folium.Map(location=[lat, lon], zoom_start=15, tiles='CartoDB positron')
    color = STYLE_MAP.get(category, DEFAULT_COLOR)
    category_label = category.replace('_', ' ').title()
    count = len(features)

    badge_color = '#27ae60' if count > 0 else '#95a5a6'
    badge_icon  = '✅' if count > 0 else 'ℹ️'
    plural      = 's' if count != 1 else ''

    # === OSM features layer ===
    osm_layer = folium.FeatureGroup(name=f"{category_label} (OSM)", show=True)

    folium.Marker(
        [lat, lon],
        popup=folium.Popup(
            f"<b>📍 Center</b><br>{location_name[:80]}", max_width=250
        ),
        icon=folium.Icon(color='red', icon='map-marker', prefix='glyphicon'),
    ).add_to(osm_layer)

    folium.Circle(
        [lat, lon],
        radius=radius_meters,
        color='#e74c3c', fill=True, fill_opacity=0.08,
        weight=2, dash_array='8',
        popup=f"Radius: {radius_meters / 1000:.1f} km",
    ).add_to(osm_layer)

    for f in features:
        parts = [
            f"<b>{f['name']}</b><br>",
            f"<span style='color:#666'>{category_label}</span>",
        ]
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
            color=color, fill=True, fill_color=color, fill_opacity=0.8,
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

    stats_html = f"""
    <div style='position:fixed;top:15px;right:15px;background:white;
                padding:16px 20px;border-radius:12px;
                box-shadow:0 4px 20px rgba(0,0,0,0.15);z-index:1000;
                min-width:240px;font-family:Arial;
                border-left:4px solid {color}'>
        <div style='font-size:18px;font-weight:700;margin-bottom:10px'>
            🗺️ GIS Agent
        </div>
        <div style='font-size:13px;color:#555;margin-bottom:6px'>
            📍 {truncated}
        </div>
        <div style='font-size:13px;color:#555;margin-bottom:6px'>
            🔍 {category_label}
        </div>
        <div style='font-size:13px;color:#555;margin-bottom:12px'>
            📏 {radius_meters / 1000:.1f} km radius
        </div>
        {user_layer_html}
        <div style='background:{badge_color};color:white;border-radius:8px;
                    padding:8px;text-align:center;font-weight:700'>
            {badge_icon} {count} {category_label}{plural} Found
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    return m.get_root().render()


# ═════════════════════════════════════════
# STEP 6 (NEW in V2.5) — Static cartographic map
# Print-quality PNG with basemap, OSM features, user layer,
# radius, north arrow, scale bar, legend, attribution.
# Returns a path to a temp PNG (caller must delete after use).
# ═════════════════════════════════════════
def generate_static_map(lat, lon, radius_meters, features, location_name,
                        category, user_gdf=None):
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

    # Search-radius polygon (red dashed)
    if buffer_3857.geom_type == 'Polygon':
        ax.add_patch(MplPolygon(
            list(buffer_3857.exterior.coords),
            facecolor='#e74c3c', alpha=0.08,
            edgecolor='#e74c3c', linestyle='--', linewidth=1.8, zorder=2,
        ))

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
                           linewidth=1.2, zorder=3)
            if not lines.empty:
                lines.plot(ax=ax, color=USER_LAYER_COLOR, linewidth=2.5,
                           alpha=0.9, zorder=4)
            if not points.empty:
                points.plot(ax=ax, color=USER_LAYER_FILL,
                            edgecolor=USER_LAYER_COLOR, markersize=50,
                            alpha=0.85, linewidth=1, zorder=5)
        except Exception as e:
            print(f"Static map: user layer plot failed: {e}")

    # OSM features (colored per category, matching web-map STYLE_MAP)
    category_color = STYLE_MAP.get(category, DEFAULT_COLOR)
    category_label = category.replace('_', ' ').title()
    if features:
        fx, fy = [], []
        for f in features:
            x, y = to_3857(f['lon'], f['lat'])
            fx.append(x); fy.append(y)
        ax.scatter(fx, fy, c=category_color, s=65, alpha=0.9,
                   edgecolor='white', linewidth=1.2, zorder=6)

    # Center marker (red triangle)
    ax.scatter([cx_x], [cx_y], c='#e74c3c', s=180, marker='v',
               edgecolor='white', linewidth=1.8, zorder=7)

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
    fig.suptitle(
        f"{category_label} within {radius_meters / 1000:.1f} km of "
        f"{location_name[:75]}",
        fontsize=11, color='#1a1a2e', x=0.05, y=0.97, ha='left',
    )

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


def generate_pdf_report(location_name, category, radius_km, features,
                        user_summary=None, static_map_path=None):
    category_label = category.replace('_', ' ').title()
    count = len(features)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
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

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*BRAND_NAVY)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(10, 7, "#", fill=True)
    pdf.cell(85, 7, _latin1("Name"), fill=True)
    pdf.cell(45, 7, _latin1("Phone"), fill=True)
    pdf.cell(0, 7, _latin1("Coordinates"), fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "", 8)
    fill = False
    for i, f in enumerate(features[:PDF_MAX_ROWS], start=1):
        pdf.set_fill_color(245, 247, 250) if fill else pdf.set_fill_color(255, 255, 255)
        coords = f"{f['lat']:.5f}, {f['lon']:.5f}"
        name  = f.get('name', 'Unnamed') or 'Unnamed'
        phone = f.get('phone', '') or '-'
        pdf.cell(10, 6, str(i), fill=True)
        pdf.cell(85, 6, _latin1(name[:42]), fill=True)
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
        file_b64  = data.get('file_b64') or ''
        file_name = (data.get('file_name') or '').strip()
        file_bytes = base64.b64decode(file_b64) if file_b64 else b''

    radius_km = max(0.1, min(radius_km, 20.0))
    radius_m  = radius_km * 1000

    if not location:
        return jsonify({'success': False, 'error': 'No location provided'}), 400

    lat, lon, full_address = geocode_location(location)
    if lat is None:
        return jsonify({
            'success': False,
            'error': f'Could not find location: {location}',
        }), 404

    features = query_osm(lat, lon, radius_m, category)

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
        )
    except Exception as e:
        # Last-resort fallback: render the map WITHOUT the user layer rather than 500
        import traceback
        print("Map generation crashed; falling back to OSM-only map.")
        traceback.print_exc()
        map_html = generate_map(
            lat, lon, radius_m, features, location, category,
            user_gdf=None, user_filename=None,
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

    if include_pdf:
        # NEW in V2.5: generate the static cartographic map first,
        # embed it in the PDF, then clean up the temp file.
        static_map_path = None
        try:
            static_map_path = generate_static_map(
                lat, lon, radius_m, features, full_address, category,
                user_gdf=user_gdf,
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
            )
        finally:
            if static_map_path and os.path.exists(static_map_path):
                try:
                    os.unlink(static_map_path)
                except OSError:
                    pass

    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'GIS Agent Backend running ✅ (V2.5)'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
