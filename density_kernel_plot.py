''' 
Script making ksdensity plot of the kernel density estimate of the flare/gas seep data at the MASOX site
author: @KnutOlaD
'''

import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import numpy as np
from KDEpy import TreeKDE
import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker
import utm as utm

###########################
####### FUNCTIONS #########
###########################

#formatter function 
def fmt(x):
    return f'{np.exp(x):.2e}'

#nan remover
def nan_remover(data,
                remove_zeros = False):
    '''Function for removing nan values from the data. Can also remove zeroes if remove_zeros is set to True.'''
    #loop through all the keys in the dictionary and remove nan values
    if remove_zeros == False:
        for key in data.keys():
            data[key] = data[key][~np.isnan(data[key])]
    else:
        #remove all data points which is zero
        for key in data.keys():
            data[key] = data[key][data[key] != 0] #this is a boolean mask ([data[key] != 0])
            data[key] = data[key][~np.isnan(data[key])]

    return data


def only_within_area(data_pos,
                        survey_area_corners_lonlat='None',
                        survey_area_corners_utm='None',
                        lonlat=True):
        '''Function for removing all data points outside the survey area. The survey area is defined by the corners of the survey area.
        The corners can be given in lonlat or utm coordinates. If the corners are given in lonlat coordinates, they will be converted to utm coordinates.
        The survey area corners should be given as a list of two lists, where the first list contains the latitudes and the second list contains the longitudes.
        I.e. in this format [[lat1, lat2, lat3, lat4], [lon1, lon2, lon3, lon4]]. Lat1 and lon1 should be the lower left corner of the survey area, lat2 and lon2
        should be the lower right corner of the survey area, lat3 and lon3 should be the upper right corner of the survey area and lat4 and lon4 should be the upper
        left corner of the survey area. Returns the indices of the data points which are within the survey area. Data can be in 
        lonlat or utm coordinates. Default is lonlat coordinates, i.e. lonlat = True. Uses utm.from_latlon to convert lonlat to utm coordinates.'''
        index_list = []
        
        #check if data is in lonlat or utm coordinates
        if lonlat == True:
            #convert to utm
            data_pos = utm.from_latlon(data_pos['lat'], data_pos['lon'])

        #convert the survey area corners to utm coordinates
        if survey_area_corners_lonlat != 'None':
            survey_area_corners_utm = utm.from_latlon(survey_area_corners_lonlat[0], survey_area_corners_lonlat[1])
        else:
            pass
        #loop through all the keys in the dictionary and remove all data points outside the survey area
        for key in data_pos.keys():
            #loop through all the data points
            for i in range(len(data_pos[key])):
                #check if the data point is within the survey area

                if data_pos[0] < survey_area_corners_utm[0][0] or data_pos[0] > survey_area_corners_utm[0][2] or data_pos[1] < survey_area_corners_utm[1][0] or data_pos[1] > survey_area_corners_utm[1][1]:
                    #if the data point is not within the survey area, remove it
                    index_list.append(i)
                else:
                    pass
    
        return index_list


def load_data(filename,
              survey_area_corners_lonlat='None',
              interactive=False):
    '''Function for loading the data from the xlsx file. Has an interactive mode where the
    user can select which columns to add in the output dictionary.'''

    #load the data
    df = pd.read_excel(filename)
    #define dictionary
    data = dict()

    ##### INTERACTIVE MODE TRIGGER #####
    #if interactive mode is on, let the user select which columns to add to the dictionary
    if interactive == True:
        #get the column names
        column_names = df.columns
        #loop through the column names and ask the user if they want to add it to the dictionary
        for i in range(len(column_names)):
            #ask the user if they want to add the column to the dictionary. Show column name and the first 5 values in the column
            #first 5 values in the column
            first_5 = df[column_names[i]][0:5]
            add_column = input(f'Do you want to add the column {column_names[i]} to the dictionary? (y/n) \n {first_5}')
            #if the user wants to add the column to the dictionary, add it
            if add_column == 'y':
                data[column_names[i]] = df[column_names[i]]
            #if the user doesn't want to add the column to the dictionary, skip it
            else:
                pass
    #if interactive mode is off, add the columns to the dictionary
    else:
        #add the columns to the dictionary
        data['flow'] = df['Flow_Rate_realBRS']
        data['lat'] = df['Average_Lat_C_Foot']
        data['lon'] = df['Average_Lon_C_Foot']
        data['UTMx'] = df['Average_X_C_Foot']
        data['UTMy'] = df['Average_Y_C_Foot']

    return data

#if __name__ == '__main__':

###########################
####### PLOTTING ##########
###########################

######################################
####### 2 DIMENSIONAL KDE PLOT #######
######################################

#load the data
filename = r'C:\Users\kdo000\Dropbox\post_doc\Marie_project\data\SB_flare_data\CAGE_16_4_Merged_Manuel.xlsx'
data = load_data(filename, interactive=True)
data = nan_remover(data,remove_zeros=True)
#data = only_within_area(data, survey_area_corners_lonlat=[58.5, 59.5])

#Create a kernel density estimate of the flare locations, use flow rate as weights and a bandwidth of 100 m
# Create 2D data of shape (obs, dims)
data_locs = np.array([data['UTMx'], data['UTMy']]).T
weights = np.array(data['flow'])

#####################
### PLOT SETTINGS ###
#####################

#fig = plt.figure(figsize=(18, 6))
grid_points = 2**7  # Grid points in each dimension
N = 26  # Number of contours
bandwidth = 100 # Bandwidth for the KDE
colormap_name = 'inferno'
#set background color
background_color = []
#select lonlat or utm coordinates
plot_lonlat = 0
#set the whole figure background color
plt.style.use("dark_background")

# Compute the kernel density estimate
kde = TreeKDE(kernel='gaussian',bw = bandwidth, norm=2)
grid, points = kde.fit(data_locs,weights).evaluate(grid_points)

# The grid is of shape (obs, dims), points are of shape (obs, 1)
x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
z = points.reshape(grid_points, grid_points).T

#define limits
vminlim = -20
vmaxlim = np.max(np.log(z))
#define levels for the contour plot
levels = np.linspace(vminlim, vmaxlim, N)

#just make one plot using the 2 norm and tree kde
fig, ax = plt.subplots()
ax.set_title(f'Norm $p={2}$, bw = {bandwidth} , tree kde')
if background_color:
    ax.set_facecolor(background_color)

#set range of coloraxis

# Plot the kernel density estimate
sc2d = ax.contourf(x, 
                   y, 
                   np.log(z), 
                   levels, 
                   cmap=colormap_name, 
                   vmin=vminlim, 
                   vmax=vmaxlim)
#plot some contour lines (only 10)
#add a scatter plot with black dots on top
ax.scatter(data['UTMx'], 
                data['UTMy'], 
                c='k', 
                s=1,
                alpha=0.5)

#find all points between the limits which are orders of magnitude, i.e. 10, 100, 1000, 10000 etc
# Define the orders of magnitude for the contours
#find the order of the minimum value in levels
min_order = np.floor(np.log10(np.exp(levels.min()))).astype(int)
#find the order of the maximum value in levels
max_order = np.ceil(np.log10(np.exp(levels.max()))).astype(int)


# Create an array of contour levels that are powers of 10
levels_order = np.logspace(min_order, max_order, num=max_order-min_order+1)

# Plot the contours at the specified levels
contour_set = ax.contour(x, 
                         y, 
                         z,  # use the original data, not the logarithm
                         levels=levels_order[1:],  # use the levels we defined
                         colors='w',
                         linestyles='solid',
                         linewidths=0.5)

# Add labels to the contours with the custom formatter
labels = ax.clabel(contour_set, 
                   inline=True, 
                   fontsize=8, 
                   fmt=ticker.FuncFormatter(lambda x, pos: f'{x:.2e}'))

# Create a "fake" colorbar that matches the range of levels
vmin, vmax = levels.min(), levels.max()
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colormap_name)
# Create the colorbar
cb = plt.colorbar(mappable)
# Set the ticks and labels on the colorbar
cb.set_ticks(np.linspace(vmin, vmax, 6))
cb.set_ticklabels([f'{np.exp(i):.2e}' for i in np.linspace(vmin, vmax, 6)])
# Label the colorbar
cb.set_label('Gas flow [ml/min/m$^2$]')

#replace ticks with latlon coordinates

if plot_lonlat == 1:
    #get the x ticks (utm x coordinates)
    xticks = ax.get_xticks()
    #get the y ticks (utm y coordinates)
    yticks = ax.get_yticks()
    #convert the x ticks to latlon coordinates
    xticks_latlon = utm.to_latlon(xticks, np.mean(data['UTMy']))
    #convert the y ticks to latlon coordinates
    yticks_latlon = utm.to_latlon(np.mean(data['UTMx']), yticks)
    #replace the ticks at the ticklocations in the plot
    ax.set_xticklabels(xticks_latlon[1],)
    ax.set_yticklabels(yticks_latlon[0])
    #add label
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
else:
    #add label
    ax.set_xlabel('UTMx [m]')
    ax.set_ylabel('UTMy [m]')


#save the x,y,z data to a xlsx file
df = pd.DataFrame(data=z)
df.to_excel(r'C:\Users\kdo000\Dropbox\post_doc\Marie_project\data\kde_data\new_data\kde_data.xlsx', index=False)
dfx = pd.DataFrame(data=x)
dfx.to_excel(r'C:\Users\kdo000\Dropbox\post_doc\Marie_project\data\kde_data\new_data\kde_data_x.xlsx', index=False)
dfy = pd.DataFrame(data=y)
dfy.to_excel(r'C:\Users\kdo000\Dropbox\post_doc\Marie_project\data\kde_data\new_data\kde_data_y.xlsx', index=False)

#save the figure
plt.savefig(r'C:\Users\kdo000\Dropbox\post_doc\Marie_project\results\kde_plots\new_kde_plots\kde_plot.png', dpi=300, bbox_inches='tight')


plt.show()


###############################
#KDE HISTOGRAM
###############################

#mirror the data
#make a sorted array of the data
data_sorted = np.sort(data['flow'], axis=0)
#make a mirrored array
data_mirrored = np.concatenate((data_sorted, -data_sorted))
#do a kernel density estimate of the mirrored data
# Create 2D data of shape (obs, dims)
#data_mirrored = np.array([data_mirrored, data_mirrored]).T.flatten()
# the bandwidth to be a function of the flow rate
#make a list of bandwidths
bandwidths = []
for i in range(len(data_mirrored)):
    bandwidths.append(np.abs(data_mirrored[i])*0.3)

#KDE HISTOGRAM
#determine the number of bins to be used
#set resolution of the histogram
grid_points = 3000 # Grid points in each dimension

x, y = TreeKDE(kernel = 'gaussian',bw=bandwidths,norm=2).fit(data_mirrored).evaluate(grid_points)
plt.plot(x,y)
plt.xlim(0,50)
#plot the bandwithd ona twinx axis

#plot a histogram
plt.hist(data_mirrored, bins=1000, density=True)
plt.xlim(0,50)
#plot legend
plt.legend(['Kernel density estimate', 'Histogram'])
plt.ylabel('Probability density')
plt.xlabel('Flowrate per single seep')
#plt.show()

#save the figure
plt.savefig(r'C:\Users\kdo000\Dropbox\post_doc\Marie_project\results\kde_plots\new_kde_plots\kde_histogram.png', dpi=300, bbox_inches='tight')