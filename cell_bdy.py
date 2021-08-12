from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import cv2
from scipy import ndimage
import argparse
# import imutils
import scipy as sp
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer
import skimage.measure
import skimage.color

import numpy as np
import xarray as xr
import pandas as pd
import datetime

from shapely.geometry import Polygon, Point, MultiPoint

# - Getting cell edges - 1st approach - skimage

def edges_skimage(total_data, new_cgridx, new_cgridy, ntime, nlevel, thres):

        blobs = total_data['reflectivity'].values[ntime,nlevel,:,:].copy()
        blobs = np.nan_to_num(blobs, 0)
        blobs[np.where(blobs < thres)[0],np.where(blobs < thres)[1]] = 0
        blobs[np.where(blobs >= thres)[0],np.where(blobs >= thres)[1]] = 2
        # Perform CCA on the mask
        labeled = skimage.measure.label(blobs, connectivity=2, return_num=True)

        # labeled[cgridx,cgridy]  
        trk = labeled[0][new_cgridy[ntime],new_cgridx[ntime]]

        for i in np.arange(np.shape(skimage.measure.regionprops(labeled[0]))[0]):
            if (skimage.measure.regionprops(labeled[0])[i].label == trk):
                trk_pop = i
                print(i)

        try:
            shape = skimage.measure.regionprops(labeled[0])[trk_pop].coords
            edge = []
            for i in np.arange(np.shape(skimage.measure.find_contours(blobs, level=1))[0]):
                verts = skimage.measure.find_contours(blobs, level=1)[i]
                edge.append(np.shape(np.where(skimage.measure.points_in_poly(shape, verts)==True)[0])[0])

            trk_per = np.where(np.array(edge) == np.array(edge).max())[0][0]

            edges = skimage.measure.find_contours(blobs, level=1)[trk_per]
            edges = np.round(edges, 0)+0.5
            edges = edges.astype('int') #### fix 
        
        except: 
            print('Returning zero')
            return np.zeros((1,2))
        # UnboundLocalError
        
        return edges
    
def cell_data(radar, edges, dict_cell, dict_key, ntime, data_final):

    edge_lon = radar.get_point_longitude_latitude(level = 0, edges = 'True')[0][edges[:,0], edges[:,1]]
    edge_lat = radar.get_point_longitude_latitude(level = 0, edges = 'True')[1][edges[:,0], edges[:,1]]
    time_center = data_final.time.values[ntime]    
    
    XRAD,YRAD = idx_win_cell(edges)
    center_lon = radar.get_point_longitude_latitude(level = 0, edges = 'True')[0][XRAD,YRAD]
    center_lat = radar.get_point_longitude_latitude(level = 0, edges = 'True')[1][XRAD,YRAD]
    ref = data_final['reflectivity'][ntime,:,XRAD,YRAD].values
    zdr = data_final['differential_reflectivity'][ntime,:,XRAD,YRAD].values
    kdp = data_final['KDP_CSU'][ntime,:,XRAD,YRAD].values
#     rhohv = data_final['cross_correlation_ratio'][ntime,:,XRAD,YRAD].values
#     phidp = data_final['differential_phase'][ntime,:,XRAD,YRAD].values
    d0 = data_final['D0'][ntime,:,XRAD,YRAD].values
    nw = data_final['NW'][ntime,:,XRAD,YRAD].values
    mu = data_final['MU'][ntime,:,XRAD,YRAD].values
    mw = data_final['MW'][ntime,:,XRAD,YRAD].values
    mi = data_final['MI'][ntime,:,XRAD,YRAD].values
    
    dict_cell.add(dict_key, [edges, edge_lon, edge_lat, time_center,
                                     center_lon, center_lat,
                                     ref, zdr, kdp, 
#                                      rhohv,
                                     d0, nw, mu, mw, mi])
    return dict_cell
    
def labels_watershed(total_data, cgridx_all, cgridy_all, ntime, nlevel, size=401):

    ref = total_data['reflectivity'].values[ntime, nlevel,:,:].copy()
    norm = ((ref - np.nanmin(ref)) / (np.nanmax(ref) - np.nanmin(ref)))
    image = norm*255
    percentile_value = 70
    image[image < ((np.nanpercentile(ref, percentile_value) - np.nanmin(ref)) / (np.nanmax(ref) - np.nanmin(ref)))*255] = 0
    image[image > ((np.nanpercentile(ref, percentile_value) - np.nanmin(ref)) / (np.nanmax(ref) - np.nanmin(ref)))*255] = 255
    test = Image.fromarray(np.uint8(image))
    test.save("file.jpeg")
    image = cv2.imread("file.jpeg")
    sp = 15;    sr = 100
    shifted = cv2.pyrMeanShiftFiltering(src = image, sp = sp, sr = sr)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    D = ndimage.distance_transform_edt(thresh)

    localMax = np.full((size, size), False)    
    localMax[cgridy_all, cgridx_all] = True
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    return labels

def edges_watershed(labels, ntime, cgridx, cgridy, size=401):
    
    trks = labels[cgridy[ntime],cgridx[ntime]]
    edge_mx = np.zeros((size,size))
    edge_mx[np.where(labels==trks)[1], np.where(labels==trks)[0]] = 1
    edges = skimage.measure.find_contours(edge_mx, level=0)[0]
    return edges
    
# For cell info
def idx_win_cell(edge):
    # Given cell edges, return the idx point grid within them 
    cell_poly_grid = Polygon(edge)

    grid_x = np.meshgrid(np.arange(401), np.arange(401))[0].flatten()
    grid_y = np.meshgrid(np.arange(401), np.arange(401))[1].flatten()

    # -- Radar centers inside cell shape
    radar_shp = MultiPoint(tuple(np.vstack((grid_x, grid_y)).transpose()))
    XRAD = [] 
    YRAD = []
    for i in np.arange(len(grid_x)):
        if cell_poly_grid.contains(radar_shp[i]) == True:
            XRAD.append(np.asarray(radar_shp[i].coords[0])[0])
            YRAD.append(np.asarray(radar_shp[i].coords[0])[1])
    XRAD = np.asarray(XRAD, dtype='int')
    YRAD = np.asarray(YRAD, dtype='int')

    return XRAD,YRAD

def get_first_scanidx(filenames, first_time):
    for idx in np.arange(len(filenames)):
        rdata = xr.open_dataset(filenames[idx])
        if first_time.astype('datetime64[s]') == rdata.time.data[0].astype('datetime64[s]'):
            break
    return idx
