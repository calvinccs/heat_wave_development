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




https://user-images.githubusercontent.com/15905401/115473320-6bc47100-a233-11eb-8361-625b6f53c6fe.mp4

https://user-images.githubusercontent.com/15905401/115473338-741cac00-a233-11eb-911d-401df1d11493.mp4

https://user-images.githubusercontent.com/15905401/115473371-872f7c00-a233-11eb-81e8-7507d120d2ce.mp4

https://user-images.githubusercontent.com/15905401/115473084-fce71800-a232-11eb-9a50-9bce40e1b83f.mp4
