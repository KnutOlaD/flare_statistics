''' 
Script making ksdensity plot of the kernel density estimate of the flare/gas seep data at the MASOX site
author: @KnutOlaD

'''
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import numpy as np
from KDEpy import TreeKDE
import pandas as pd
#import utm as utm

#load the data from the xlsx file
df = pd.read_excel(r'C:\Users\kdo000\Dropbox\post_doc\Marie_project\src\Density_kernel\MASOX_JFflarescompiled_byMS_precluster_OK.xlsx', sheet_name='Sheet1')

data = dict()

data['flow'] = df['Flow_Rate_realBRS']
data['lat'] = df['Average_Lat_C_Foot']
data['lon'] = df['Average_Lon_C_Foot']
data['UTMx'] = df['Average_X_C_Foot']
data['UTMy'] = df['Average_Y_C_Foot']

survey_area_corners_lonlat = np.array([[9.267, 78.65], [9.438, 78.661], [9.703, 78.492], [9.543, 78.481]])
#survey_area_corners_utm = utm.from_latlon(survey_area_corners_lonlat[:,1], survey_area_corners_lonlat[:,0])

#create a dictionary with only data within the survey area
data_in_survey_area = dict()
data_in_survey_area['flow'] = []
data_in_survey_area['lat'] = []
data_in_survey_area['lon'] = []
data_in_survey_area['UTMx'] = []
data_in_survey_area['UTMy'] = []

for i in range(len(data['UTMx'])):
    if data['lat'][i] < np.max(survey_area_corners_lonlat[:,1]) and data['lon'][i] < np.max(survey_area_corners_lonlat[:,0]) and data['lon'][i] > np.min(survey_area_corners_lonlat[:,0]) and data['lon'][i] > np.min(survey_area_corners_lonlat[:,0]):
        data_in_survey_area['flow'].append(data['flow'][i])
        data_in_survey_area['lat'].append(data['lat'][i])
        data_in_survey_area['lon'].append(data['lon'][i])
        data_in_survey_area['UTMx'].append(data['UTMx'][i])
        data_in_survey_area['UTMy'].append(data['UTMy'][i])

data = data_in_survey_area


###########################
####### PLOTTING ##########
###########################

#Plot the utm x and y coordinates with flow rate as color in a scatter plot
fig, ax = plt.subplots()
sc = ax.scatter(data['UTMx'], 
                data['UTMy'], 
                c=data['flow'], 
                cmap='viridis',
                vmin=0,
                vmax=10,
                s=10)
#set colorbar ranges 
ax.set_xlabel('UTMx')
ax.set_ylabel('UTMy')
ax.set_title('UTMx and UTMy coordinates of the flares with flow rate as color')
plt.colorbar(sc)
plt.show()

#Create a kernel density estimate of the flare locations, use flow rate as weights and a bandwidth of 100 m
# Create 2D data of shape (obs, dims)
data_locs = np.array([data['UTMx'], data['UTMy']]).T
weights = np.array(data['flow'])
grid_points = 2**6  # Grid points in each dimension
N = 16  # Number of contours
bandwidth = 100

fig = plt.figure(figsize=(12, 4))

#just make one plot using the 2 norm and tree kde
fig, ax = plt.subplots()
ax.set_title(f'Norm $p={2}$, bw = 100, tree kde')
# Compute the kernel density estimate
kde = TreeKDE(kernel='epa',bw = bandwidth, norm=2)
grid, points = kde.fit(data_locs,weights).evaluate(grid_points)

# The grid is of shape (obs, dims), points are of shape (obs, 1)
x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
z = points.reshape(grid_points, grid_points).T

#set range of coloraxis

import matplotlib as mpl

# Plot the kernel density estimate
sc2d = ax.contourf(x, y, z, N, cmap="RdBu_r", vmin=np.min(z), vmax=0.000001)
#add a scatter plot with black dots on top
ax.scatter(data['UTMx'], 
                data['UTMy'], 
                c='k', 
                s=1,
                alpha=0.5)

# Create a "fake" colorbar that has the desired range
norm = mpl.colors.Normalize(vmin=0, vmax=0.000001)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")

#plt.colorbar(mappable)
#put label on colorbar
plt.colorbar(mappable).set_label('Kernel density estimate, gas flow per area')

plt.show()