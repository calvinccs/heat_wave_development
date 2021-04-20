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
1) Surface temperature from 2011 to 2020
2) Geopotential heights in 2020 (due to the size of data, we will concentrated on heat waves in 2020

Additionally, there is a csv file contain lat lon information of major cities.
Due to size matter, folders contain the NetCDF file will be included but not the actual nc files.
