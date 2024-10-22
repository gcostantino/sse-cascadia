import joblib


def load_cascadia_geometry():
    geometry, recs = joblib.load('../../DATA/sse-cascadia/cascadia_geometry_info/geometry_cascadia')
    green_td = joblib.load('../../DATA/sse-cascadia/cascadia_geometry_info/green_td_cascadia')
    return geometry, green_td
