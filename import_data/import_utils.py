from shapely.geometry import Point
import datetime
import time
import geopandas as gpd
import pandas as pd
from shapely import wkt

convert_to_timestamp = lambda x: time.mktime(
    datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").timetuple()
)


def lon_lat_to_geom(data, lon_name="LON", lat_name="LAT"):
    geom_col = []
    for i, row in data.iterrows():
        geom = Point(row[lon_name], row[lat_name])
        geom_col.append(geom)
    data["geom"] = geom_col
    data = gpd.GeoDataFrame(data, geometry="geom")
    return data


def write_geodataframe(gdf, out_path):
    geo_col_name = gdf.geometry.name
    df = pd.DataFrame(gdf, copy=True)
    df[geo_col_name] = gdf.geometry.apply(wkt.dumps)
    df.to_csv(out_path, index=True)


def read_geodataframe(in_path, geom_col="geom"):
    df = pd.read_csv(in_path)
    df[geom_col] = df[geom_col].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=geom_col)
    return gdf