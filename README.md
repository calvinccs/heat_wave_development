# heat_wave_development
This program will find the heatwaves using NCEP/NCAR reanalysis data and then create graph and video on the heat wave development.

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



![Hong Kong](https://user-images.githubusercontent.com/15905401/115422838-31d47a00-a1f5-11eb-9173-6094135ff2d8.png)
![Paris](https://user-images.githubusercontent.com/15905401/115472984-c7dac580-a232-11eb-8390-add8a89c9501.png)

