from dask.distributed import Client, LocalCluster
from concurrent.futures import ProcessPoolExecutor, as_completed
import dask.dataframe as dd
import pandas as pd
import math
from pyproj import Transformer
import rioxarray as rxr
import numpy as np
from osgeo import gdal, gdalconst
from numpy import floor, ceil, sqrt, rad2deg, gradient, arctan2
import pystac_client
import planetary_computer
from tqdm import tqdm
from dask import delayed
import dask
from requests.exceptions import HTTPError

def process_single_location(args):
    lat, lon, tile = args

    signed_asset = planetary_computer.sign(tile.assets["data"])
    elevation = rxr.open_rasterio(signed_asset.href)
    
    slope = elevation.copy()
    aspect = elevation.copy()

    transformer = Transformer.from_crs("EPSG:4326", elevation.rio.crs, always_xy=True)
    xx, yy = transformer.transform(lon, lat)

    tilearray = np.around(elevation.values[0]).astype(int)
    geo = (math.floor(float(lon)), 90, 0.0, math.ceil(float(lat)), 0.0, -90)

    driver = gdal.GetDriverByName('MEM')
    temp_ds = driver.Create('', tilearray.shape[1], tilearray.shape[0], 1, gdalconst.GDT_Float32)

    temp_ds.GetRasterBand(1).WriteArray(tilearray)
    #temp_ds.GetRasterBand(1).SetNoDataValue(no_data_value)
    #temp_ds.SetProjection('EPSG:4326')
    #temp_ds.SetGeoTransform(geo)

    tilearray_np = temp_ds.GetRasterBand(1).ReadAsArray()
    grad_y, grad_x = gradient(tilearray_np)

    # Calculate slope and aspect
    slope_arr = np.sqrt(grad_x**2 + grad_y**2)
    aspect_arr = rad2deg(arctan2(-grad_y, grad_x)) % 360 
    
    slope.values[0] = slope_arr
    aspect.values[0] = aspect_arr

    elev = round(elevation.sel(x=xx, y=yy, method="nearest").values[0])
    slop = round(slope.sel(x=xx, y=yy, method="nearest").values[0])
    asp = round(aspect.sel(x=xx, y=yy, method="nearest").values[0])

    return elev, slop, asp

def process_data_in_chunks(tile_group, tiles):
    chunk_results = []
    index_id = int(tile_group['index_id'].iloc[0])
    tile  = tiles[index_id]

	######Add the signed asset and elevation here itself#########

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_single_location, (lat, lon, tile)) for lat, lon in tqdm(zip(tile_group['lat'], tile_group['lon']))]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                chunk_results.append(future.result(timeout=5))
            except concurrent.futures.TimeoutError:
                print("Processing location timed out.")
                continue
            except HTTPError as e:
                print(f"Failed to process a location due to HTTPError: {e}")
                continue  
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue

    return pd.DataFrame(chunk_results, columns=['Elevation_m', 'Slope_Deg', 'Aspect_L'])

def extract_terrain_data(df):
    
    client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", ignore_conformance=True)

    area_of_interest = {'type': 'Polygon', 
                        'coordinates': [[[-120.37693519839556, 36.29213061937931],
                                        [-120.37690215328962, 38.8421802805432], 
                                        [-118.29165268221286, 38.84214595220293],
                                        [-118.2917116398743, 36.29209713778364], 
                                        [-120.37693519839556, 36.29213061937931]]]}
    
    search = client.search(collections=["cop-dem-glo-90"], intersects=area_of_interest)
    
    tiles = list(search.items())

    results = df.groupby('index_id').apply(lambda group: process_data_in_chunks(group, tiles))

    return results
