import numpy as np
import xarray as xr
import pandas as pd
import datetime

# For cell info
def max_cell_area(dict_cell, ntime):
    dict_keys = ['edges'+str(x) for x in np.arange(ntime + 1)]
    cell_areas = []
    [cell_areas.append(np.shape(dict_cell[dict_keys[k]][4])[0]) for k in np.arange(ntime)]
    largest_area = max(cell_areas)
    return largest_area

def ds_cell_1time(dict_cell, largest_area, nt):
    """Creates a xarray file for the cell at a given ntime"""
    
    # Creating the dataset for the cell during their tracks
    # Coords - time, height, x, y
    dict_keys = 'edges'+str(nt)

    time = datetime.datetime.utcfromtimestamp(dict_cell[dict_keys][3].tolist()/1e9)
    time = pd.date_range(time, periods=1)
    alt = np.arange(0,15500,500) # 31 levels
    x = np.arange(largest_area)  # size of the cell at its largest size
    y = np.arange(largest_area)

    # Vars - Latitude, Longitude, REF, ZDR, KDP, PHIDP, RHOHV, (D), NW, MU, MW, MI)
    # Matrices in 1x31xlen(x)xlen(y)


    ref = dict_cell[dict_keys][6]
    REF = np.zeros((1,len(alt),len(x),len(y)))
    REF[:] = 'nan'
    REF[:,:, 0:len(ref[0,:,0]), 0:len(ref[0,0,:])] = ref

    zdr = dict_cell[dict_keys][7]
    ZDR = np.zeros((1,len(alt),len(x),len(y)))
    ZDR[:] = 'nan'
    ZDR[:,:, 0:len(zdr[0,:,0]), 0:len(zdr[0,0,:])] = zdr

    kdp = dict_cell[dict_keys][8]
    KDP = np.zeros((1,len(alt),len(x),len(y)))
    KDP[:] = 'nan'
    KDP[:,:, 0:len(kdp[0,:,0]), 0:len(kdp[0,0,:])] = kdp

    rhohv = dict_cell[dict_keys][9]
    RHOHV = np.zeros((1,len(alt),len(x),len(y)))
    RHOHV[:] = 'nan'
    RHOHV[:,:, 0:len(rhohv[0,:,0]), 0:len(rhohv[0,0,:])] = rhohv

    ### DSD

    d0 = dict_cell[dict_keys][10]
    D0 = np.zeros((1,len(alt),len(x),len(y)))
    D0[:] = 'nan'
    D0[:,:, 0:len(d0[0,:,0]), 0:len(d0[0,0,:])] = d0

    nw = dict_cell[dict_keys][11]
    NW = np.zeros((1,len(alt),len(x),len(y)))
    NW[:] = 'nan'
    NW[:,:, 0:len(nw[0,:,0]), 0:len(nw[0,0,:])] = nw

    mu = dict_cell[dict_keys][12]
    MU = np.zeros((1,len(alt),len(x),len(y)))
    MU[:] = 'nan'
    MU[:,:, 0:len(mu[0,:,0]), 0:len(mu[0,0,:])] = mu

    mw = dict_cell[dict_keys][13]
    MW = np.zeros((1,len(alt),len(x),len(y)))
    MW[:] = 'nan'
    MW[:,:, 0:len(mw[0,:,0]), 0:len(mw[0,0,:])] = mw

    mi = dict_cell[dict_keys][14]
    MI = np.zeros((1,len(alt),len(x),len(y)))
    MI[:] = 'nan'
    MI[:,:, 0:len(mi[0,:,0]), 0:len(mi[0,0,:])] = mi

    # - Meshgrid on lat and lon and then in 31 levels
    lon, lat = np.meshgrid(dict_cell[dict_keys][4], dict_cell[dict_keys][5])

    LON = np.zeros((1,len(alt),len(x),len(y)))
    LON[:] = 'nan'
    LAT = np.zeros((1,len(alt),len(x),len(y)))
    LAT[:] = 'nan'

    for i in np.arange(len(alt)):
        LON[0,i,0:len(lon[:,0]),0:len(lon[:,0])], LAT[0,i,0:len(lat[:,0]),0:len(lat[:,0])] = np.meshgrid(dict_cell[dict_keys][4], dict_cell[dict_keys][5]) 

    # Create netcdf xarray dataset for 1 time
    ds = xr.Dataset(
        data_vars=dict(
            reflectivity=(["time", "alt", "x", "y"], REF),
            differential_reflectivity=(["time", "alt", "x", "y"], ZDR),
            KDP_CSU=(["time", "alt", "x", "y"], KDP),
            RHOHV=(["time", "alt", "x", "y"], RHOHV),
            D0=(["time", "alt", "x", "y"], D0),
            NW=(["time", "alt", "x", "y"], NW),
            MU=(["time", "alt", "x", "y"], MU),
            MW=(["time", "alt", "x", "y"], MW),
            MI=(["time", "alt", "x", "y"], MI),
            latitude=(["time", "alt", "x", "y"],LAT),
            longitude=(["time", "alt", "x", "y"],LON)    
        ),
        coords=dict(
            time=("time",time),
            alt=("alt", alt),
            x=("x", x),
            y=("y",y)))
    
    return ds
    
    
# For edges

def max_cell_edges(dict_cell, ntime):
    dict_keys = ['edges'+str(x) for x in np.arange(ntime)]
    cell_edges = []
    [cell_edges.append(np.shape(dict_cell[dict_keys[k]][1])[0]) for k in np.arange(ntime)]
    largest_edges = max(cell_edges)
    return largest_edges

def ds_celledges_1time(dict_cell, largest_edges, nt):
    
    dict_keys = 'edges'+str(nt)
    time = datetime.datetime.utcfromtimestamp(dict_cell[dict_keys][3].tolist()/1e9)
    time = pd.date_range(time, periods=1)
    xy = np.arange(largest_edges)  # size of the cell at its largest size

    lon_edge = dict_cell[dict_keys][1]
    LON_EDGE = np.zeros((1,len(xy)))
    LON_EDGE[:] = 'nan'
    LON_EDGE[:, 0:len(lon_edge)] = lon_edge

    lat_edge = dict_cell[dict_keys][2]
    LAT_EDGE = np.zeros((1,len(xy)))
    LAT_EDGE[:] = 'nan'
    LAT_EDGE[:, 0:len(lat_edge)] = lat_edge

    
    # Create netcdf xarray dataset for 1 time
    dse = xr.Dataset(
        data_vars=dict(
            latitude_edges=(["time","xy"],LAT_EDGE),
            longitude_edges=(["time","xy"],LON_EDGE)    
        ),
        coords=dict(time=("time",time), xy=("xy", xy) ))
    
    return dse
