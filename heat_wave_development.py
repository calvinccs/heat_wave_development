#!/usr/bin/env python
"""
The objective of this project is to:
1) Create a time series variables at any point in the globle
   (e.g. surface temperature) using NCEP/NCAR reanalysis 2 data
2) Run a loop to find the heat wave events from 2011 to 2020 according to the 
   definition given by the Met Office, but using a different method to calculate the threshold value
3) Plot the geopotential heights at upper level (e.g. 925 hPa / 850 hPa), it will provide 
   useful insights on how heat wave is develop. The functin can also create a video of the changing
   variables.
   
Note that, to run this program, it is required to download NCEP/NCAR Reanalysis 2 data:
1) Create a directory above the directory of this file, name it as 'climate_data'
2) Download NCEP/NCAR reanalysis data:
3) Surface temperature from 2011 to 2020 
    (https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=61&tid=90982&vid=4237)
4) Locate the Surface temperature nc files in 'climate_data/surface_air/'
5) Geopotential heights in 2020 (due to the size of data, we will concentrated on heat waves in 2020
    (https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=61&tid=90982&vid=1454)
6) Locate the Geopotential heights nc files in 'climate_data/hgt/'

Additionally, there is a csv file contain lat lon information of major cities.

"""
# change thess variables for other analysis, e.g. target_location = Paris, New York etc.
# pressure_level = 925

target_location = 'Paris' 
var = 'air'
pressure_level = 850 # part 3
before_event = 2
after_event = 2

import os
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import KDTree
import math
import pandas as pd
import moviepy.video.io.ImageSequenceClip
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

df_c = pd.read_csv('city_lonlat.csv')
df_c['lonlat'] = df_c.apply(lambda x: [x.lng, x.lat], axis = 1)
city_dict = df_c.set_index('city').lonlat.to_dict()
data_path = '../climate_data/'

class idw_Interpolation(object):
    """
    Use IDW interpolation to find the estimate any variable (e.g. surface temperature) 
    in any point in the globe from the NCEP/NCAR reanalysis data
    """
    def __init__(self, nc, variable, target, r = 2):
        if type(target) == str:
            if target in city_dict.keys():
                target = city_dict[target]
#                 print(target)
            else:
                print('Target location is not recognise!')
#         else:
        if target[0] < r:
            target[0] = target[0] + 360
        d = Dataset(nc, "r", format="NETCDF4")
        # to fix the idw problem in location longitude near zero by overlapping 
        lon = np.append(d['lon'][:].data, d['lon'][:].data[:5]+360) 
        lat = d['lat'][:].data
        var = d[variable][:]
        lon_size = d['lon'][:].data.shape[0]
        pts, pts_o = find_nearest_pts(lon, lat, target, r = r)
        pts_d = [str(round(harvesine(pts[i][0], pts[i][1], target[0], target[1]), 0))+'km' for i in range(len(pts))]
        ts = [pd.to_datetime(d['time'].units[12:]) + np.timedelta64(int(d['time'].actual_range[0]) + i*6, 'h')
              for i in range(d['time'].shape[0])]
        for pt in pts_o:
            if pt[0] >= lon_size:
                pt[0] = pt[0] - lon_size
                
        var_df = pd.DataFrame()
        for i in range(len(pts_o)):
            var_df[i] = d['air'][:, 0, pts_o[i][1], pts_o[i][0]].data - 273.15
      
        var_df['idw'] = var_df.apply(lambda x: idwr([x.T], 
                                                    [p[0] for p in pts], 
                                                    [p[1] for p in pts], 
                                                    target[0], target[1]), axis = 1)
        var_df.columns = [str(pts[i]) + ' ' + pts_d[i] for i in range(len(pts))] + ['idw']
        var_df['ts'] = ts
        var_df['t_unit'] = d['time'][:].data
        self.var_df = var_df
        self.idw_ts = var_df.set_index('ts')[['idw', 't_unit']]
        self.d = d
        self.pts = pts
        self.pts_o = pts_o
        self.pts_d = pts_d
        self.target = target

class heat_wave(object):
    """
    Heatwave is met when a location records a period of at least three consecutive 
    days with maximum temperatures meeting or exceeding a heatwave temperature threshold. 
    The threshold varies by UK county in the range 25–28°C. In this study, we use 95th percentile 
    of the daily maximum temperature as the threshold.
    """
    def __init__(self, T, threshold_percentile = 0.95):
        t = T.idw
        Tmax = t.loc[t.groupby(pd.Grouper(freq='D')).idxmax()]
        threshold = round(Tmax.quantile(.95), 2)
        # create a loop to find days that match the heat wave definition
        d = 0
        hw_dict = {'start': [],
                   'start_t_unit': [],
                   'end': [],
                   'end_t_unit': [],
                   'duration': [],
                   'mean_Tmax': [],
                   'min_Tmax': [],
                   'max_Tmax': [],
                   }
        while d < len(Tmax):
            # Tmax is higher than T1 for three consecutive days or more
            if (Tmax[d] > threshold) and (Tmax[d+1] > threshold) and (Tmax[d+2] > threshold):
                hw_Tmax = [Tmax[d], Tmax[d+1], Tmax[d+2]]
                more_days = 1
                while (Tmax[d+2+more_days] > threshold):
                    hw_Tmax.append(Tmax[d+2+more_days])
                    more_days += 1
                hw_dict['start'].append(str(Tmax.index[d].date()))
                hw_dict['start_t_unit'].append(T.loc[Tmax.index[d]].t_unit)
                hw_dict['end'].append(str(Tmax.index[d+2+more_days].date()))
                hw_dict['end_t_unit'].append(T.loc[Tmax.index[d+2+more_days]].t_unit)
                hw_dict['duration'].append(3+more_days)
                hw_dict['mean_Tmax'].append(sum(hw_Tmax)/len(hw_Tmax))
                hw_dict['min_Tmax'].append(min(hw_Tmax))
                hw_dict['max_Tmax'].append(max(hw_Tmax))
                
                d += 2 + more_days
            d += 1
        self.hw_df = pd.DataFrame(hw_dict)
        self.Tmax = Tmax
        self.threshold = threshold
        self.t = t
        return        

class gph_map(object):
    # generate contour plot for geopotential heights
    def __init__(self, gph, city, t_unit_start, t_unit_end, p_level, before = 0, after = 0):
        time = gph['time'][:].data
        t_start = int(np.where(time == t_unit_start)[0])-before
        t_end = int(np.where(time == t_unit_end)[0])+after
        t_unit_range = list(range(int(t_unit_start) - before*24, int(t_unit_end) + after*24+1))[0::6]
        level = gph['level'][:].data
        p = int(np.where(level == p_level)[0])

        lon = np.append(gph['lon'][:].data, gph['lon'][:].data[:9]+360)
        lat = gph['lat'][:].data
        target = city_dict[city]
        if target[0] < 0:
            target[0] = target[0] + 360
        margin = [round((target[0] - 15), 1), round((target[1] - 15), 1), 
                  round((target[0] + 15), 1), round((target[1] + 15), 1)]
        if margin[0] <= 0:
            margin[0] += 360
            margin[2] += 360

        lon_m = np.where((lon > margin[0]-2.5) & (lon < margin[2]+2.5))[0]
        lon_m = np.where(lon_m >= 144, lon_m-144, lon_m)
        lat_m = np.where((lat > margin[1]-2.5) & (lat < margin[3]+2.5))[0]
        
        lon_p = lon[lon_m]
        hgt_min = 1500
        hgt_max = 0
        hgt_profile = []
        for t in range(t_start, t_end+1):
            if lon_m[-1] < lon_m[0]: # lon at zero in the plotting area
                lon_p = np.where(lon_p < 300, lon_p + 360, lon_p)
                hgt = gph['hgt'][t, 1, lat_m[0]:lat_m[-1]+1, np.r_[lon_m[0]:144, 0:lon_m[-1]+1]]
                hgt_min = min(hgt_min, hgt.min())
                hgt_max = max(hgt_max, hgt.max())
                hgt_profile.append(hgt)
            else:
                hgt = gph['hgt'][t, 1, lat_m[0]:lat_m[-1]+1, lon_m[0]:lon_m[-1]+1]
                hgt_min = min(hgt_min, hgt.min())
                hgt_max = max(hgt_max, hgt.max())
                hgt_profile.append(hgt)
        
        image_path = 'Heat_wave_' + t_unit2ts(t_unit_start)[:10].replace('-', '_') + '_' + city
        image_path = image_path.replace(' ', '')
        
        if image_path not in os.listdir():
            os.mkdir(image_path)
        for i in range(len(hgt_profile)):
            plt.figure(figsize=(10,10))
            map = Basemap(projection='merc',llcrnrlon=margin[0],llcrnrlat=margin[1],  # 'merc'
                      urcrnrlon=margin[2],urcrnrlat=margin[3],resolution='l')
            map.drawcoastlines()
            map.drawcountries()
            lons,lats= np.meshgrid(lon_p,lat[lat_m])
            x, y = map(lons, lats)
        
            p_scale = np.linspace(hgt_min, hgt_max, 9)
            gph_map = map.contourf(x,y,hgt_profile[i], levels = p_scale)
            cb = map.colorbar(gph_map,"bottom", size="5%", pad="2%")
            plt.title('Geopotential Heights at %d hPa level (%s) \n %s'%(p_level, city, 
                                                                      t_unit2ts(t_unit_range[i])
                     ))
            cb.set_label('Geopotential Heights (m)')
            plt.savefig(image_path + '/hw_%s.png'%"{0:03}".format(i))
            
        self.image_path = image_path
        return
    
    def make_video(self):
        image_folder = self.image_path
        fps=1
        image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
        image_files.sort()
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(image_folder + '/%s_clip.mp4'%image_folder)
        print('Video created!!')
        return
        
def find_nearest_pts(lon, lat, target, r):
    # lon, lat are the grid in the NetCDF file
    # target 
    x, y = np.meshgrid(lon, lat)
    pts = []
    pts_o = []
    for i in range(lat.shape[0]):
        for j in range(lon.shape[0]):
            pts.append((x[i][j], y[i][j]))
            pts_o.append([j, i])
    pts = np.array(pts)
    pts_o = np.array(pts_o)
    T = KDTree(pts)
    idx = T.query_ball_point(target,r=r)
    return pts[idx], pts_o[idx]

# Distance calculation, degree to km (Haversine method)
def harvesine(lon1, lat1, lon2, lat2):
    rad = math.pi / 180  # degree to radian
    R = 6378.1  # earth average radius at equador (km)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat1 * rad) * \
        math.cos(lat2 * rad) * (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

def idwr(z, x, y, xi, yi):
    # single point
    lstdist = []
    for s in range(len(x)):
        d = harvesine(x[s], y[s], xi, yi)
        lstdist.append(d)
    sumsup = list((1 / np.power(lstdist, 2)))
    suminf = np.sum(sumsup)
    sumsup = np.sum(np.array(sumsup) * np.array(z))
    u = sumsup / suminf
    return u

def var_ts(variable, path, target):
    list_of_files = os.listdir(path)
    list_of_files.sort()
    var_ts = pd.DataFrame()
    print('Reading NetCDF files...')
    for f in list_of_files:
        print(f)
        product = idw_Interpolation(path + f, variable, target)
        var_ts = var_ts.append(product.idw_ts)
    return var_ts

def t_unit2ts(t_unit):
    ts = str(pd.to_datetime('1800-1-1 00:00:0.0') + np.timedelta64(int(t_unit), 'h'))
    return ts

if __name__ == "__main__":
    print("Running heat wave development program")
    t_location = var_ts('air', data_path + 'surface_air/', target_location)
    hw_location = heat_wave(t_location)
    print('Heat wave threshold: ', hw_location.threshold)
    
    events = hw_location.hw_df[hw_location.hw_df.start.str[:4] == '2020'].reset_index(drop = True)
    events.index += 1
    print(events)
    which_event = input('Which event do you want to review?')
    which_event = int(which_event)
    gph = Dataset(data_path + 'hgt/hgt.2020.nc', "r", format="NETCDF4")
    heatwave_event = gph_map(gph, target_location,
                    events.loc[which_event].start_t_unit,
                    events.loc[which_event].end_t_unit, 
                    pressure_level, before = before_event, after = after_event)
    heatwave_event.make_video()
    
