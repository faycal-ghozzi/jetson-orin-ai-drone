"""
Telemetry normalization + light-weight geo math.

- normalize_incoming(js) -> dict with:
    { ts, lat, lon, alt_m, yaw_deg, pitch_deg, roll_deg, enu:{x,y,z}, quat:{x,y,z,w} }

- ENU origin is configurable via env:
    TELEM_ORIGIN_LAT, TELEM_ORIGIN_LON, TELEM_ORIGIN_ALT
  (defaults are harmless demo values)
- Optional yaw offset (deg) via env TELEM_YAW_OFFSET

Math notes:
- ENU: small-area approximation (accurate to ~meter scale over city-size regions).
- Quaternion: Z (yaw) -> Y (pitch) -> X (roll) intrinsic order.
"""

import os, math
from typing import Dict, Any

# --- Configurable origin (demo defaults: Paris-ish) ---------------------------
_ORIGIN_LAT = float(os.getenv("TELEM_ORIGIN_LAT", "48.8566"))
_ORIGIN_LON = float(os.getenv("TELEM_ORIGIN_LON", "2.3522"))
_ORIGIN_ALT = float(os.getenv("TELEM_ORIGIN_ALT", "35.0"))
_YAW_OFFSET = float(os.getenv("TELEM_YAW_OFFSET", "0.0"))

def set_origin(lat: float, lon: float, alt_m: float = 0.0):
    """Override ENU origin at runtime if desired."""
    global _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT
    _ORIGIN_LAT, _ORIGIN_LON, _ORIGIN_ALT = float(lat), float(lon), float(alt_m)

# --- Geo helpers --------------------------------------------------------------
def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0

def _latlon_to_enu_m(lat: float, lon: float, alt_m: float):
    """
    Convert WGS84 lat/lon/alt -> local ENU (meters) using a good small-area approximation.
    Precision ~0.5–2 m across typical drone footprints.
    """
    lat0 = _deg2rad(_ORIGIN_LAT)
    lat_r = _deg2rad(lat)
    d_lat = lat_r - lat0
    d_lon = _deg2rad(lon - _ORIGIN_LON)

    # meters per degree (better than constant 111km approach)
    sin2 = math.sin(lat0)**2
    coslat = math.cos(lat0)
    m_per_deg_lat = (111132.92 - 559.82*math.cos(2*lat0) + 1.175*math.cos(4*lat0) - 0.0023*math.cos(6*lat0))
    m_per_deg_lon = (111412.84*math.cos(lat0) - 93.5*math.cos(3*lat0) + 0.118*math.cos(5*lat0))

    # Convert rad deltas to deg deltas, then to meters
    dlat_deg = (lat - _ORIGIN_LAT)
    dlon_deg = (lon - _ORIGIN_LON)
    north_m = dlat_deg * m_per_deg_lat
    east_m  = dlon_deg * m_per_deg_lon
    up_m    = float(alt_m) - _ORIGIN_ALT

    # (x=east, y=north, z=up)
    return float(east_m), float(north_m), float(up_m)

def _euler_zyx_to_quat(yaw_deg: float, pitch_deg: float, roll_deg: float):
    """
    Intrinsic Z-Y-X (yaw, then pitch, then roll). Angles in degrees.
    Returns quaternion (x, y, z, w).
    """
    y = _deg2rad(yaw_deg)
    p = _deg2rad(pitch_deg)
    r = _deg2rad(roll_deg)

    cy = math.cos(y*0.5); sy = math.sin(y*0.5)
    cp = math.cos(p*0.5); sp = math.sin(p*0.5)
    cr = math.cos(r*0.5); sr = math.sin(r*0.5)

    # ZYX composition
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    yq = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return {"x": float(x), "y": float(yq), "z": float(z), "w": float(w)}

def normalize_incoming(js: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts a raw packet (phone JSON, TCP JSON, etc) and returns a normalized dict:
      {
        "ts": float,
        "lat": float, "lon": float, "alt_m": float,
        "yaw_deg": float, "pitch_deg": float, "roll_deg": float,
        "enu": {"x": float, "y": float, "z": float},
        "quat": {"x": float, "y": float, "z": float, "w": float}
      }
    Missing fields are filled with safe defaults (0).
    Applies TELEM_YAW_OFFSET.
    """
    ts   = float(js.get("ts", 0.0))
    lat  = float(js.get("lat", 0.0))
    lon  = float(js.get("lon", 0.0))
    alt  = float(js.get("alt_m", js.get("alt", 0.0)))
    yaw   = float(js.get("yaw_deg",   js.get("yaw",   0.0))) + _YAW_OFFSET
    pitch = float(js.get("pitch_deg", js.get("pitch", 0.0)))
    roll  = float(js.get("roll_deg",  js.get("roll",  0.0)))

    enu_x, enu_y, enu_z = _latlon_to_enu_m(lat, lon, alt)
    quat = _euler_zyx_to_quat(yaw, pitch, roll)

    return {
        "ts": ts,
        "lat": lat, "lon": lon, "alt_m": alt,
        "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
        "enu": {"x": enu_x, "y": enu_y, "z": enu_z},
        "quat": quat
    }

