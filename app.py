from flask import Flask, request, jsonify
import requests
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

app = Flask(__name__)

# ─────────────────────────────────────────
# CATEGORY → OSM TAG MAPPING
# Not every POI lives under the "amenity" tag.
# Supermarkets are shop=supermarket, parks are leisure=park,
# hotels are tourism=hotel, etc. This map routes each
# category to the correct OSM tag so we don't return 0 results.
# Hospital/pharmacy also check the newer `healthcare=` schema
# because many Lebanese facilities use that tagging.
# ─────────────────────────────────────────
CATEGORY_MAP = {
    # health
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
    # amenity tag
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
    # shop tag
    'supermarket':  [('shop', 'supermarket')],
    'bakery':       [('shop', 'bakery')],
    # leisure tag
    'park':         [('leisure', 'park')],
    'playground':   [('leisure', 'playground')],
    # tourism tag
    'hotel':        [('tourism', 'hotel')],
    'museum':       [('tourism', 'museum')],
}

# Marker color per category
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

# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# ─────────────────────────────────────────
# STEP 1 — Geocode place name to coordinates
# ─────────────────────────────────────────
def geocode_location(location_str, retries=2):
    """Return (lat, lon, full_address) or (None, None, None)."""
    geolocator = Nominatim(user_agent="gis_agent_v1/1.0")
    for attempt in range(retries + 1):
        try:
            # language='en' forces Nominatim to return English place names
            # instead of the local script (Arabic, Cyrillic, etc.).
            loc = geolocator.geocode(location_str, timeout=10, language='en')
            if loc:
                return loc.latitude, loc.longitude, loc.address
            return None, None, None
        except (GeocoderTimedOut, GeocoderServiceError):
            if attempt < retries:
                time.sleep(1.5)
                continue
            return None, None, None


# ─────────────────────────────────────────
# STEP 2 — Pull features from OpenStreetMap
# Uses `requests` directly instead of the outdated `overpy` library,
# which sends headers that newer Overpass servers reject (HTTP 406).
# ─────────────────────────────────────────
def query_osm(lat, lon, radius_meters, category):
    """Query Overpass for features of `category` within radius."""
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
        'User-Agent': 'gis_agent_v1/1.0 (GIS analysis tool)',
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

            # Pick coordinates: nodes have lat/lon, ways use the center
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


# ─────────────────────────────────────────
# STEP 3 — Generate the interactive map
# ─────────────────────────────────────────
def generate_map(lat, lon, radius_meters, features, location_name, category):
    """Render an interactive Folium map and return its HTML."""
    m = folium.Map(location=[lat, lon], zoom_start=15, tiles='CartoDB positron')
    color = STYLE_MAP.get(category, DEFAULT_COLOR)
    category_label = category.replace('_', ' ').title()
    count = len(features)

    # Smart badge: green check for hits, neutral gray info for zero
    badge_color = '#27ae60' if count > 0 else '#95a5a6'
    badge_icon  = '✅' if count > 0 else 'ℹ️'
    plural      = 's' if count != 1 else ''

    # Center marker
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(
            f"<b>📍 Center</b><br>{location_name[:80]}", max_width=250
        ),
        icon=folium.Icon(color='red', icon='map-marker', prefix='glyphicon'),
    ).add_to(m)

    # Search-radius circle
    folium.Circle(
        [lat, lon],
        radius=radius_meters,
        color='#e74c3c', fill=True, fill_opacity=0.08,
        weight=2, dash_array='8',
        popup=f"Radius: {radius_meters / 1000:.1f} km",
    ).add_to(m)

    # Feature markers
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
        ).add_to(m)

    # Stats overlay (top-right corner)
    truncated = (
        location_name[:50] + '...' if len(location_name) > 50 else location_name
    )
    stats_html = f"""
    <div style='position:fixed;top:15px;right:15px;background:white;
                padding:16px 20px;border-radius:12px;
                box-shadow:0 4px 20px rgba(0,0,0,0.15);z-index:1000;
                min-width:220px;font-family:Arial;
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
        <div style='background:{badge_color};color:white;border-radius:8px;
                    padding:8px;text-align:center;font-weight:700'>
            {badge_icon} {count} {category_label}{plural} Found
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    return m.get_root().render()


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}

    location = (data.get('location') or '').strip()
    # Accept both `category` (new) and `amenity_type` (legacy) so existing
    # n8n workflows keep working during the switch.
    category = data.get('category') or data.get('amenity_type') or 'hospital'

    try:
        radius_km = float(data.get('radius_km', 2))
    except (TypeError, ValueError):
        radius_km = 2.0
    radius_km = max(0.1, min(radius_km, 20.0))  # clamp to sane range
    radius_m = radius_km * 1000

    if not location:
        return jsonify({'success': False, 'error': 'No location provided'}), 400

    lat, lon, full_address = geocode_location(location)
    if lat is None:
        return jsonify({
            'success': False,
            'error': f'Could not find location: {location}',
        }), 404

    features = query_osm(lat, lon, radius_m, category)
    map_html = generate_map(lat, lon, radius_m, features, location, category)

    return jsonify({
        'success':   True,
        'location':  full_address,
        'category':  category,
        'radius_km': radius_km,
        'count':     len(features),
        'features':  features,
        'map_html':  map_html,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'GIS Agent Backend running ✅'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
