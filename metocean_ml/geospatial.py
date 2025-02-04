import numpy as np
import pandas as pd
import xarray as xr
import geopy
import geopy.distance
import spectra_tools
import geopandas as gpd
import contextily as ctx
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import pyproj
import shapely
import geographiclib
import rasterio
import h5py
import rasterio.transform
import os

from roaring_landmask import RoaringLandmask
from roaring_landmask import LandmaskProvider
provider = LandmaskProvider.Osm
landmask = RoaringLandmask.new_with_provider(provider)

from affine import Affine
from shapely.geometry import Point
from global_land_mask import globe
from tqdm import tqdm

def is_land(lat,lon):
    '''
    Check if points are on land or sea.
    Uses https://github.com/gauteh/roaring-landmask
    '''
    # Landmask can only handle floats.
    lat = np.array(lat,dtype=float)
    lon = np.array(lon,dtype=float)
    
    return landmask.contains_many(lon,lat).reshape(lat.shape)

def haversine_distances_pairwise(lat0,lon0,lat1,lon1):
    '''
    Calculate the great-circle distance between latitude and longitude pairs.
    This implementation is pair-wise, meaning all arrays must have the same shape!
    
    Arguments:
    ----------
    lat0 : float or array
    lon0 : float or array
    lat1 : float or array
    lon1 : float or array
    
    Returns:
    --------
    distances : array
        Arc length, multiply by earth radius ~6378 to get km.
    '''

    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    
    dLat = lat1-lat0
    dLon = lon1-lon0
    
    d = np.sin(dLat/2)**2 + np.cos(lat0)*np.cos(lat1)*np.sin(dLon/2)**2
    d = 2*np.arcsin(np.sqrt(d))

    return d

def haversine_distances(lat0,lon0,lat1,lon1):
    '''
    Calculate haversine distances between all combinations of lat/lon points.
    
    Arguments:
    ----------
    lat0, lon0 : floats or arrays
        The first set of points. These must have equal shape.
    lat1, lon1 : floats or arrays
        The second set of points. These must have equal shape.

    Returns:
    --------
    distances : array
        Array with shape (N_0, N_1) representing the arc distance
        from each point in the first set to each in the second,
        where N_0 = lat0.size = lon0.size and N_1 = lat1.size = lon1.size.
        Multiply by earth's radius (6378km) to get distance.
    '''
    lat0 = np.array(lat0)
    lon0 = np.array(lon0)
    lat1 = np.array(lat1)
    lon1 = np.array(lon1)
    
    lat0,lat1 = np.meshgrid(lat0,lat1,indexing='ij')
    lon0,lon1 = np.meshgrid(lon0,lon1,indexing='ij')
    
    return haversine_distances_pairwise(lat0,lon0,lat1,lon1)


def merge_directions_points(points,directions,drop_duplicates="points",on_index=True,merge_tolerance=1):
    '''
    Add information from dataframe of directions to dataframe of points.
    drop_duplicates: remove duplicated columns from either points or directions
    '''
    for param in ["r","dir_from"]:
        assert param in points.columns

    keep = "dir_idx" if on_index else "dir_from"
    if drop_duplicates == "points": 
        drop = [c for c in points.columns if c in directions.columns and c!=keep]
        points = points.drop(drop,axis=1)
    if drop_duplicates == "directions": 
        drop = [c for c in directions.columns if c in points.columns and c!=keep]
        directions = directions.drop(drop,axis=1)

    if on_index:
        if directions["dir_idx"].nunique() != points["dir_idx"].nunique():
            raise IndexError("Non-matching indices. Verify that tables are generated with the same set of directions.")
        points = pd.merge(points, directions, on="dir_idx", how='left')
    else:
        points = pd.merge_asof(
            points.sort_values('dir_from'), 
            directions.sort_values('dir_from'),
            on = 'dir_from',
            direction = 'nearest',
            tolerance = merge_tolerance)

    points["delay"] = (1000/3600)*points["r"]/points["group_vel"]
    points["delay"] = points["delay"].fillna(0)
    return points

def wave_properties_from_source_points(origin, spec, lat, lon, drop_duplicates=True):
    '''
    Add info from spec to a list of points given by lat and lon.
    Works by interpolating the spectra to the list of directions
    from origin to each point.
    '''
    origin = np.array(origin)
    points = []
    for i,(x,y) in enumerate(zip(lon,lat)):
        distance = calculate_distance(origin[0],origin[1],y,x)
        dir_from = calculate_bearing(origin[0],origin[1],y,x)
        dir_to = (dir_from + 180) % 360
        points.append({"node":i,"dir_from":dir_from,"dir_to":dir_to,"r":distance,"lat":y,"lon":x})
    points = pd.DataFrame(points).sort_values("dir_to").reset_index(drop=True)
    points["dir_idx"] = np.arange(len(points))

    directions = spectra_tools.directional_spec_info(spec)

    points = merge_directions_points(points,directions,on_index=False)

    return points

def create_wedge(center_lat, center_lon, dir_from, dir_to, radius_km, edge_resolution=100, arc_resolution=10):
    """
    Create a wedge polygon centered at (center_lon, center_lat) 
    with given direction, radius, and span.
    """
    geod = pyproj.Geod(ellps="WGS84")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    radius_m = radius_km * 1000  # Convert kilometers to meters
    angles = np.linspace(dir_from, dir_to, arc_resolution)
    long_side = np.linspace(radius_m,0,edge_resolution)

    l_side = [geod.fwd(center_lon,center_lat,angles[0],r)[:2] for r in long_side][::-1]
    r_side = [geod.fwd(center_lon,center_lat,angles[-1],r)[:2] for r in long_side]
    boundary_points = [geod.fwd(center_lon, center_lat, angle, radius_m)[:2] for angle in angles]

    # Close the polygon by adding the center point
    #boundary_points = [(center_lon, center_lat)] + boundary_points + [(center_lon, center_lat)]
    boundary_points = l_side+boundary_points+r_side
    wedge_polygon = shapely.geometry.Polygon(boundary_points)

    # Transform to Web Mercator
    wedge_polygon_mercator = shapely.ops.transform(lambda x, y: transformer.transform(x, y), wedge_polygon)
    return wedge_polygon_mercator


def line_of_sight_plot(center_lat,
                       center_lon,
                       points:pd.DataFrame,
                       as_sectors=True,
                       map_detail=3,
                       color_var="",
                       cmap=plt.get_cmap("Spectral_r"),
                       alpha=0.75,
                       ):
    
    '''
    For a given location, visualize potential wave sources on map.
    '''
    
    fig,ax = plt.subplots(figsize=[10,10])
    if as_sectors:
        wedges = points.loc[points.groupby(["dir_from"])["r"].idxmax()].reset_index()
        wedges["dir-"] = wedges["dir_from"] - 0.5*wedges["dir_from"].diff(1)
        wedges.loc[0,"dir-"] = wedges.loc[0,"dir_from"] - 0.5*(wedges.loc[0,"dir_from"] - wedges["dir_from"].iloc[-1] + 360)
        wedges["dir+"] = np.roll(wedges["dir-"],-1)
        wedges.loc[len(wedges)-1,"dir+"] += 360
        wedges['wedge_mercator'] = wedges.apply(lambda row: create_wedge(center_lat, center_lon, row['dir-'], row["dir+"], row['r']),axis=1)
        
        data = {'latitude': points["lat"], 'longitude': points["lon"]}
        gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)
        gdf.plot(ax = ax,color="tab:green", alpha=0, markersize=3)

        if color_var in points.columns:
            norm = mcolors.Normalize(vmin=0,vmax=wedges[color_var].max())
            wedges["sector_color"] = norm(wedges[color_var])
        else:
            wedges["sector_color"] = 0.75
            if color_var:
                print(f"Warning: color_var not found in columns of points: {points.columns}")

        for i,row in tqdm(wedges.iterrows(),total=len(wedges)):
            x,y = row["wedge_mercator"].exterior.xy
            ax.fill(x,y,alpha=alpha,color=cmap(row["sector_color"]),edgecolor="none")

        if color_var in points.columns:
            sm = plt.cm.ScalarMappable(norm,cmap)
            fig.colorbar(sm,ax=ax,label=color_var)

    else:
        data = {'latitude': points["lat"], 'longitude': points["lon"]}
        gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)
        gdf.plot(ax = ax,color="tab:green", alpha=alpha, markersize=3)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=map_detail)
    plt.xticks([],[])
    plt.yticks([],[])
    return fig,ax

def calculate_bearing(lat0, lon0, lat1, lon1):
    '''Get bearing from point 0 towards point 1, in degrees clockwise from north.'''
    return geographiclib.geodesic.Geodesic.WGS84.Inverse(lat0,lon0,lat1,lon1)["azi1"] % 360

def calculate_distance(lat0,lon0,lat1,lon1):
    '''Get distance from each point in (lon0,lat0) to each point in (lon1,lat1), in km.'''

    return np.squeeze(haversine_distances(lat0,lon0,lat1,lon1))*6378

def distribute_points(lat0:float,
                      lon0:float,
                      bearing:float|np.ndarray,
                      distance:float|np.ndarray,
                      )->tuple[np.ndarray,np.ndarray]:
    '''
    Calculate new latitude/longitude points,
    based on an origin point, direction and distance.
    
    Arguments:
    ----------
    lat0 : float
        Origin latitude
    lon0 : float
        Origin longitude
    bearing : float or array
        Direction(s) to distribute points, in degrees.
    distances : float or array
        Distance(s) to place points, in km.
    
    Returns:
    --------
    points : tuple(array, array)
        Tuple of two arrays in the order (latitudes, longitudes).
        Each array is the shape (n_bearings, n_distances).
    '''
    
    if hasattr(lat0,'__len__') or (lat0<-90) or (lat0>90): 
        raise ValueError(f'Expected {-90}<lat<{90}, got {lat0}.')
    if hasattr(lon0,'__len__') or (lon0<-180) or (lon0>180): 
        raise ValueError(f'Expected {-180}<lon<{180}, got {lon0}.')
    
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    bearing = np.radians(np.array(bearing)%360)
    
    distance = np.array(distance)/6371 # earth average radius
    bearing,distance = np.meshgrid(bearing,distance,indexing='ij')
    
    lat1 = np.arcsin( np.sin(lat0)*np.cos(distance) + np.cos(lat0)*np.sin(distance)*np.cos(bearing))
    lon1 = lon0 + np.arctan2(np.sin(bearing)*np.sin(distance)*np.cos(lat0),np.cos(distance)-np.sin(lat0)*np.sin(lat1))

    return np.degrees(lat1), np.degrees(lon1)

def get_fetch(
    center_lat,
    center_lon,
    directions=360,
    max_fetch=1000,
    step_size=0.01):
    '''
    Calculate the fetch distance for all directions,
    for a given lat/lon point.
    
    Arguments:
    ----------
    center_lat : float
        The latitude of the point of interest.
    center_lon : float
        The longitude of the point of interest.
    directions : int
        Number of directions.
    max_fetch : int
        Maximum distance [km] to check for land.
    step_size : float
        Interval of points [km] to check for land.
        
    Returns:
    --------
    fetch : pandas Series
        Series of fetch distances by direction.
    '''
    
    directions = np.linspace(0,360,directions,endpoint=False)
    distances = np.arange(0,max_fetch,step_size)
    lat,lon = distribute_points(center_lat,center_lon,directions,distances)
    land = is_land(lat,lon)
    if np.any(land[:,0]): raise ValueError(f'Point {center_lat},{center_lon} is on land.')
    fetch = np.argmax(land,axis=1)
    fetch[fetch==0] = land.shape[1]
    fetch = fetch * step_size

    return pd.Series(data=fetch,index=directions,name='Fetch').rename_axis('Direction')