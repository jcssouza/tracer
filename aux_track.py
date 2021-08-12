import math
import numpy as np
import xarray as xr

def backward_propagation(filenames, nscans, min_peak, nlevel, avg_area, cgridx, cgridy):
    """
    Backward propagation searching for maximum reflectivity (min_peak) within a given area (avg_area).
    filenames: list of radar files locations used on the previous track
    nscans: number of scans the cell was initially tracked
    nlevel: height level index
    cgridx, cgridy: centroid location for the cell identified by the track
    
    Returns:
    total_data: xarray file including files from the original track plus the new files found backward.
    past_scan: number of scans found backward
    new_cgridx, new_cgridy: location for the cell identified by the track
    new_nscans: original scans the cell was identified plus the ones it was previously present 
    """

    if nscans[0] == 0:
        total_data = xr.open_dataset(filenames[nscans[0]])
        for i in np.arange(1,len(nscans)):
            data = xr.open_dataset(filenames[nscans[i]])
            total_data = xr.combine_by_coords([total_data, data])
            del(data)

        new_cgridx = cgridx
        new_cgridy = cgridy
        new_nscans = nscans
        
        return total_data, 0, new_cgridx, new_cgridy, new_nscans

    else:
        back_cgridy = []
        back_cgridx = []
        peak_ref = []
        cgridx0 = cgridx[0]
        cgridy0 = cgridy[0]        
        # past_scan = 0
        for past_i in np.arange(nscans[0]): 
            data = xr.open_dataset(filenames[nscans[0] - past_i])
            peak = np.nanmax(data['reflectivity'].values[0, nlevel, 
                                                         cgridy0-avg_area: cgridy0+avg_area,
                                                         cgridx0-avg_area:cgridx0+avg_area])
            if peak < min_peak or math.isnan(peak):
                past_scan=past_i
                break   
            peak_ref.append(peak)
            past_scan = past_i
            len_past_scan = past_i+1
            back_cgridy.append(np.where(data['reflectivity'].values[0,nlevel,:,:] == peak_ref[past_i])[0][0])
            back_cgridx.append(np.where(data['reflectivity'].values[0,nlevel,:,:] == peak_ref[past_i])[1][0])

            # update center
            cgridx0 = back_cgridx[past_i]
            cgridy0 = back_cgridy[past_i]
            del(data)

        # - Combining idx for the backward prop and the one done by tint
        new_nscans = np.zeros(len(nscans) + len_past_scan)
        new_cgridy, new_cgridx = np.zeros(len(nscans) + len_past_scan), np.zeros(len(nscans) + len_past_scan)
        for idx, scan in zip(np.arange(past_scan + 1), np.arange(past_scan, -1, -1)):
            new_nscans[idx] = nscans[0]-scan-1
            new_cgridy[scan] = back_cgridy[idx]
            new_cgridx[scan] = back_cgridx[idx]
        new_nscans[len_past_scan:] = nscans
        new_nscans = new_nscans.astype(int)
        new_cgridy[len_past_scan:] = cgridy
        new_cgridx[len_past_scan:] = cgridx
        new_cgridx = new_cgridx.astype(int)
        new_cgridy = new_cgridy.astype(int)

        # - All the data for the cell
        # - -past_scan to include the cell info before tint track
        if back_cgridx:
            total_data = xr.open_dataset(filenames[nscans[0] - len_past_scan])
            for i in np.arange(past_scan-1,0,-1):
                data = xr.open_dataset(filenames[nscans[0]-i])
                total_data = xr.combine_by_coords([total_data, data])
                del(data)


            for i in np.arange(len(nscans)):
                data = xr.open_dataset(filenames[nscans[i]])
                total_data = xr.combine_by_coords([total_data, data])
                del(data)
        else:
            total_data = xr.open_dataset(filenames[nscans[0]])
            for i in np.arange(1, len(nscans)):
                data = xr.open_dataset(filenames[nscans[i]])
                total_data = xr.combine_by_coords([total_data, data])
                del(data)

            new_cgridx = cgridx
            new_cgridy = cgridy
            new_nscans = nscans

        return total_data, past_scan, new_cgridx, new_cgridy, new_nscans
        
        
def forward_propagation(filenames, total_data, nscans, min_peak, nlevel, avg_area, cgridx, cgridy):

    if filenames[-1] == filenames[nscans[-1]]:
        
        return total_data, 0, cgridx, cgridy, nscans
    else:
        
        future_cgridy = []
        future_cgridx = []
        peak_ref = []
        cgridxf = cgridx[-1]
        cgridyf = cgridy[-1]
        for future_i in np.arange(len(filenames)-nscans[-1]): 
            data = xr.open_dataset(filenames[nscans[-1] + future_i])
            peak = np.nanmax(data['reflectivity'].values[0, nlevel, 
                                                         cgridyf - avg_area:cgridyf + avg_area,
                                                         cgridxf - avg_area:cgridxf + avg_area])
            if peak < min_peak or math.isnan(peak):
                future_scan=future_i
                break   
            peak_ref.append(peak)
            future_scan = future_i + 1
            future_cgridy.append(np.where(data['reflectivity'].values[0,nlevel,:,:] == peak_ref[future_i])[0][0])
            future_cgridx.append(np.where(data['reflectivity'].values[0,nlevel,:,:] == peak_ref[future_i])[1][0])

            # update center
            cgridxf = future_cgridx[future_i]
            cgridyf = future_cgridy[future_i]
            del(data)

        fw_nscans = np.zeros(len(nscans) + future_scan)
        fw_cgridy, fw_cgridx = np.zeros(len(nscans) + future_scan), np.zeros(len(nscans) + future_scan)
        for idx in np.arange(future_scan):
            fw_nscans[len(nscans)+idx] = nscans[-1] + idx +1
            fw_cgridy[len(nscans)+idx] = future_cgridy[idx]
            fw_cgridx[len(nscans)+idx] = future_cgridx[idx]
        fw_nscans[:len(nscans)] = nscans
        fw_cgridy[:len(nscans)] = cgridy
        fw_cgridx[:len(nscans)] = cgridx
        fw_nscans = fw_nscans.astype(int)
        fw_cgridx = fw_cgridx.astype(int)
        fw_cgridy = fw_cgridy.astype(int)

        if future_cgridx:
            for i in np.arange(1,future_scan):
                if len(filenames) >= nscans[-1] + i:
                    data = xr.open_dataset(filenames[nscans[-1] + i])
                    total_data = xr.combine_by_coords([total_data, data])
                    del(data)

        return total_data, future_i, fw_cgridx, fw_cgridy, fw_nscans
