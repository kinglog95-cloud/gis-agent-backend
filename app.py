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

# V2 additions
import geopandas as gpd
import pandas as pd
from shapely.geometry import (
    Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
)

app = Flask(__name__)

# ─────────────────────────────────────────
# CATEGORY → OSM TAG MAPPING (unchanged from V1.5)
# ─────────────────────────────────────────
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

# V2: file upload limits
MAX_FILE_SIZE_MB = 5
USER_LAYER_COLOR = '#1f6feb'   # distinct blue for user-uploaded data
USER_LAYER_FILL  = '#3b82f6'


# ═════════════════════════════════════════
# STEP 1 — Geocode (unchanged from V1.5)
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
# STEP 2 — OSM query (unchanged from V1.5)
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
            OVERPASS_URL,
            data={'data': query},
            headers=headers,
            timeout=60,
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
# STEP 3 (NEW in V2) — File ingestion
# Parses geojson / esri-json / shapefile zip / kml / kmz / csv / gpkg
# and returns a GeoDataFrame reprojected to WGS84 (EPSG:4326).
# ═════════════════════════════════════════

def _looks_like_esri_json(obj):
    """Detect ArcGIS REST JSON (NOT standard GeoJSON)."""
    if not isinstance(obj, dict):
        return False
    if 'features' not in obj or not isinstance(obj['features'], list):
        return False
    # Strongest signals: geometryType + features[].attributes
    if 'geometryType' in obj:
        return True
    if obj['features']:
        first = obj['features'][0]
        if isinstance(first, dict) and 'attributes' in first:
            return True
    return False


def _esri_geom_to_shapely(geom, geom_type):
    """Convert one Esri JSON geometry into a Shapely geometry."""
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
            # Naive but works for the common case: each ring = one polygon.
            # Esri JSON uses ring winding for holes (CW exterior, CCW holes),
            # but properly resolving multi-ring polygons with holes is
            # complex — deferred to V3.5 when geoprocessing matures.
            polys = [Polygon(r) for r in rings if len(r) >= 4]
            if not polys:
                return None
            return polys[0] if len(polys) == 1 else MultiPolygon(polys)
    except Exception as e:
        print(f"Esri geom parse error: {e}")
        return None
    return None


def _esri_json_to_geodataframe(esri):
    """Convert ArcGIS REST JSON object → GeoDataFrame in its native CRS."""
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

    # Resolve CRS from spatialReference
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
        # No CRS info — assume WGS84 (best effort, may be wrong)
        gdf.set_crs(epsg=4326, inplace=True, allow_override=True)

    return gdf


def _parse_geojson_or_esri(content_bytes):
    """Standard GeoJSON or Esri ArcGIS REST JSON — auto-detect."""
    text = content_bytes.decode('utf-8-sig', errors='replace')
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Not valid JSON: {e}")

    if _looks_like_esri_json(obj):
        return _esri_json_to_geodataframe(obj)

    # Standard GeoJSON — let geopandas handle it
    gdf = gpd.read_file(StringIO(text))
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
    return gdf


def _parse_shapefile_zip(content_bytes):
    """Extract a zipped shapefile and read it. Handles nested folders."""
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
                "No .shp file found in the zip. A shapefile upload must include "
                ".shp, .shx, .dbf (and ideally .prj) together."
            )

        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            # No .prj — assume WGS84 (best effort)
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        return gdf


def _parse_kml_bytes(kml_bytes):
    """Read raw KML bytes via a temp file (GDAL needs a path)."""
    with tempfile.NamedTemporaryFile(suffix='.kml', delete=False) as tmp:
        tmp.write(kml_bytes)
        tmp_path = tmp.name
    try:
        gdf = gpd.read_file(tmp_path)
        if gdf.crs is None:
            # KML is always WGS84 by spec
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        return gdf
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _parse_kmz(content_bytes):
    """KMZ = zipped KML. Extract the .kml and read it."""
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
    """CSV with latitude/longitude columns → Point GeoDataFrame in WGS84."""
    text = content_bytes.decode('utf-8-sig', errors='replace')
    try:
        df = pd.read_csv(StringIO(text))
    except Exception as e:
        raise ValueError(f"Could not read CSV: {e}")

    # Case-insensitive column lookup with common synonyms
    cols = {c.lower().strip(): c for c in df.columns}
    lat_col = None
    lon_col = None
    for candidate in ('latitude', 'lat', 'y'):
        if candidate in cols:
            lat_col = cols[candidate]
            break
    for candidate in ('longitude', 'lon', 'lng', 'long', 'x'):
        if candidate in cols:
            lon_col = cols[candidate]
            break

    if not lat_col or not lon_col:
        raise ValueError(
            f"CSV needs latitude/longitude columns. Found columns: "
            f"{list(df.columns)}. Expected one of "
            f"latitude/lat/y and longitude/lon/lng/long/x."
        )

    df = df.dropna(subset=[lat_col, lon_col])
    geom = [Point(xy) for xy in zip(df[lon_col].astype(float),
                                    df[lat_col].astype(float))]
    return gpd.GeoDataFrame(df, geometry=geom, crs='EPSG:4326')


def _parse_gpkg(content_bytes):
    """GeoPackage — write to temp and read with geopandas."""
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
    # DEBUG: log what we received so we can verify base64 decoded correctly
    print(f"DEBUG upload: filename={filename!r}, bytes={len(content_bytes)}, "
          f"first_16_hex={content_bytes[:16].hex()}")
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File too large ({size_mb:.1f} MB). "
            f"Limit is {MAX_FILE_SIZE_MB} MB for V2."
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

    # Always reproject to WGS84 for web display
    try:
        epsg = gdf.crs.to_epsg() if gdf.crs else None
    except Exception:
        epsg = None
    if epsg != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def summarize_gdf(gdf, filename, original_crs_str):
    """Build a small JSON-safe summary of the uploaded layer."""
    geom_types = sorted(set(gdf.geometry.geom_type.dropna().tolist()))
    attr_cols = [c for c in gdf.columns if c != 'geometry']
    return {
        'filename':       filename,
        'feature_count':  int(len(gdf)),
        'geometry_types': geom_types,
        'original_crs':   original_crs_str,
        'display_crs':    'EPSG:4326',
        'attributes':     attr_cols[:20],   # cap for sanity
    }


# ═════════════════════════════════════════
# STEP 4 — Map generation (UPDATED in V2)
# Now optionally overlays an uploaded user layer with its own style + toggle.
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
            + ''.join(parts)
            + "</div>"
        )
        folium.CircleMarker(
            [f['lat'], f['lon']],
            radius=9,
            color=color, fill=True, fill_color=color, fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f['name'],
        ).add_to(osm_layer)
    osm_layer.add_to(m)

    # === User-uploaded layer (NEW in V2) ===
    # === User-uploaded layer (V2 + line-rendering fix) ===
    if user_gdf is not None and len(user_gdf) > 0:
        user_label = (user_filename or 'Your Data').rsplit('.', 1)[0][:40]
        user_layer = folium.FeatureGroup(name=f"📂 {user_label}", show=True)

        tooltip_fields = [c for c in user_gdf.columns if c != 'geometry'][:5]

        def _build_popup(row):
            lines = []
            for c in tooltip_fields:
                val = row.get(c, '')
                if val is not None and str(val).strip() and str(val) != 'nan':
                    lines.append(f"<b>{c}:</b> {val}")
            return ("<div style='font-family:Arial;min-width:140px'>"
                    + "<br>".join(lines) + "</div>") if lines else None

        # Split geometries by type — each renders with its own style
        for _, row in user_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            gt = geom.geom_type
            popup_html = _build_popup(row)
            popup = folium.Popup(popup_html, max_width=260) if popup_html else None

            # POINTS — small filled circles
            if gt in ('Point', 'MultiPoint'):
                pts = [geom] if gt == 'Point' else list(geom.geoms)
                for p in pts:
                    folium.CircleMarker(
                        [p.y, p.x],
                        radius=7,
                        color=USER_LAYER_COLOR,
                        fill=True, fill_color=USER_LAYER_FILL, fill_opacity=0.7,
                        popup=popup,
                    ).add_to(user_layer)

            # LINES — visible strokes, no fill
            elif gt in ('LineString', 'MultiLineString'):
                lines = [geom] if gt == 'LineString' else list(geom.geoms)
                for ln in lines:
                    coords = [(y, x) for x, y in ln.coords]  # folium wants [lat, lon]
                    folium.PolyLine(
                        coords,
                        color=USER_LAYER_COLOR,
                        weight=4,
                        opacity=0.85,
                        popup=popup,
                    ).add_to(user_layer)

            # POLYGONS — outlined, semi-transparent fill
            elif gt in ('Polygon', 'MultiPolygon'):
                polys = [geom] if gt == 'Polygon' else list(geom.geoms)
                for poly in polys:
                    exterior = [(y, x) for x, y in poly.exterior.coords]
                    holes = [
                        [(y, x) for x, y in ring.coords]
                        for ring in poly.interiors
                    ]
                    folium.Polygon(
                        locations=[exterior] + holes if holes else exterior,
                        color=USER_LAYER_COLOR,
                        weight=2,
                        fill=True,
                        fill_color=USER_LAYER_FILL,
                        fill_opacity=0.35,
                        popup=popup,
                    ).add_to(user_layer)

            # Anything else (GeometryCollection, etc.) — log and skip
            else:
                print(f"User layer: unsupported geometry type {gt}, skipped")

        user_layer.add_to(m)

        # Fit map bounds to include user layer + radius circle
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

    # Layer toggle (only useful when there's >1 layer, but harmless otherwise)
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
# STEP 5 — PDF report (extended in V2 to mention uploaded layer)
# ═════════════════════════════════════════
def _latin1(s):
    if s is None:
        return ""
    return str(s).encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(location_name, category, radius_km, features,
                        user_summary=None):
    category_label = category.replace('_', ' ').title()
    count = len(features)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(26, 26, 46)
    pdf.cell(0, 12, _latin1("GIS Agent - Analysis Report"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, _latin1("Powered by OpenStreetMap + Claude AI"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_text_color(40, 40, 40)
    summary = [
        ("Location", location_name),
        ("Category", category_label),
        ("Search radius", f"{radius_km} km"),
        ("Results found", str(count)),
    ]
    for label, value in summary:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(35, 8, _latin1(label + ":"))
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 8, _latin1(value),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # NEW in V2: include uploaded layer summary if present
    if user_summary and 'error' not in user_summary:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(31, 111, 235)
        pdf.cell(0, 8, _latin1("Uploaded Data Layer"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "", 11)
        for label, key in [
            ("File", "filename"),
            ("Features", "feature_count"),
            ("Geometry", "geometry_types"),
            ("Source CRS", "original_crs"),
        ]:
            val = user_summary.get(key, '')
            if isinstance(val, list):
                val = ", ".join(map(str, val))
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(35, 7, _latin1(label + ":"))
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 7, _latin1(str(val)),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    elif user_summary and 'error' in user_summary:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(192, 57, 43)
        pdf.multi_cell(0, 6, _latin1(
            f"Uploaded file '{user_summary.get('filename','')}' could not be "
            f"processed: {user_summary['error']}"
        ))
        pdf.set_text_color(40, 40, 40)

    pdf.ln(4)

    if count == 0:
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(0, 8, _latin1(
            "No features were found in this area for the selected category."))
        return base64.b64encode(bytes(pdf.output())).decode("utf-8")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(26, 26, 46)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(10, 8, "#", fill=True)
    pdf.cell(85, 8, _latin1("Name"), fill=True)
    pdf.cell(45, 8, _latin1("Phone"), fill=True)
    pdf.cell(0, 8, _latin1("Coordinates"), fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "", 9)
    fill = False
    for i, f in enumerate(features[:PDF_MAX_ROWS], start=1):
        pdf.set_fill_color(245, 247, 250) if fill else pdf.set_fill_color(255, 255, 255)
        coords = f"{f['lat']:.5f}, {f['lon']:.5f}"
        name = f.get('name', 'Unnamed') or 'Unnamed'
        phone = f.get('phone', '') or '-'
        pdf.cell(10, 7, str(i), fill=True)
        pdf.cell(85, 7, _latin1(name[:42]), fill=True)
        pdf.cell(45, 7, _latin1(phone[:22]), fill=True)
        pdf.cell(0, 7, _latin1(coords), fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        fill = not fill

    if count > PDF_MAX_ROWS:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(0, 6, _latin1(
            f"... and {count - PDF_MAX_ROWS} more results not listed here."))

    return base64.b64encode(bytes(pdf.output())).decode("utf-8")


# ═════════════════════════════════════════
# Endpoint
# ═════════════════════════════════════════
@app.route('/analyze', methods=['POST'])
def analyze():
    # Detect whether this is a multipart upload or pure JSON
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
        # Legacy base64 path (kept for backward compat)
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

    map_html = generate_map(
        lat, lon, radius_m, features, location, category,
        user_gdf=user_gdf, user_filename=file_name or None,
    )

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
        result['pdf_base64'] = generate_pdf_report(
            full_address, category, radius_km, features,
            user_summary=user_summary,
        )

    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'GIS Agent Backend running ✅ (V2)'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
