""" 
Script making ksdensity plot of the kernel density estimate of the flare/gas seep data at the MASOX site
author: @KnutOlaD
"""

import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import numpy as np
from KDEpy import TreeKDE
import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker
import utm as utm
import matplotlib.path as mpltPath

###########################
####### FUNCTIONS #########
###########################

# formatter function
def fmt(x):
    return f"{np.exp(x):.2e}"

# nan remover
def nan_remover(data, remove_zeros=False):
    """Function for removing nan values from the data.
    Can also remove zeroes if remove_zeros is set to True."""
    # loop through all the keys in the dictionary and remove nan values
    if remove_zeros == False:
        for key in data.keys():
            data[key] = data[key][~np.isnan(data[key])]
    else:
        # remove all data points which is zero
        for key in data.keys():
            data[key] = data[key][
                data[key] != 0
            ]  # this is a boolean mask ([data[key] != 0])
            data[key] = data[key][~np.isnan(data[key])]
    return data

def only_within_area(data, survey_area_corners_lonlat):
    """Function for removing all data points outside the survey area.
    The survey area corners should be given as a list of two lists, where the first list contains the 
    latitudes and the second list contains the longitudes. I.e. in this format [[upper left],upper right],
    [lower right],[lower left]] in [lon,lat] Returns the indices of the data points which are within the 
    survey area. Data can be in lonlat or utm coordinates. Default is lonlat coordinates, i.e. lonlat = True. 
    Uses utm.from_latlon to convert lonlat to utm coordinates.
    """
    index_list = []
    # loop through all the keys in the dictionary and remove all data points outside the survey area
    quadrilateral = mpltPath.Path(np.array(survey_area_corners_lonlat))
    # loop through and remove all data points outside the survey area
    for i in range(len(data["lon"])):
        # check if the data point is within the survey area
        if quadrilateral.contains_point((data["lon"][i], data["lat"][i])) == False:
            # if the data point is not within the survey area, remove it
            pass
        else:
            index_list.append(i)
    return index_list


def load_data(filename, interactive=False):
    """Function for loading the data from the xlsx file. Has an interactive mode where the
    user can select which columns to add in the output dictionary."""

    # load the data
    df = pd.read_excel(filename)
    # define dictionary
    data = dict()

    ##### INTERACTIVE MODE TRIGGER #####
    # if interactive mode is on, let the user select which columns to add to the dictionary
    if interactive == True:
        # get the column names
        column_names = df.columns
        # loop through the column names and ask the user if they want to add it to the dictionary
        for i in range(len(column_names)):
            # ask the user if they want to add the column to the dictionary. Show column 
            # first 5 values in the column name and the first 5 values in the column
            first_5 = df[column_names[i]][0:5]
            add_column = input(
                f"Do you want to add the column {column_names[i]} to the dictionary? (y/n) \n {first_5}"
            )
            # if the user wants to add the column to the dictionary, add it
            if add_column == "y":
                data[column_names[i]] = df[column_names[i]]
                # if there's no column name, i.e. the column name is 'Unnamed: xx' prompt the 
                # user to enter a column name
                if column_names[i][0:8] == "Unnamed:":
                    new_column_name = input(
                        "Enter a column name or press enter to use default: "
                    )
                    # if the user doesn't enter a column name, use the default column name
                    if new_column_name == "":
                        data[column_names[i]] = df[column_names[i]]
                    # if the user enters a column name, use that column name
                    else:
                        data[new_column_name] = df[column_names[i]]
            # if the user doesn't want to add the column to the dictionary, skip it
            else:
                pass
    # if interactive mode is off, add the columns to the dictionary
    else:
        # add the columns to the dictionary
        data["flow"] = df["Flow_Rate_realBRS"]
        data["lat"] = df["Average_Lat_C_Foot"]
        data["lon"] = df["Average_Lon_C_Foot"]
        data["UTMx"] = df["Average_X_C_Foot"]
        data["UTMy"] = df["Average_Y_C_Foot"]
    return data


def replace_dict_keys(data, old_keys, new_keys, interactive=False):
    """Function for replacing the keys in the dictionary with new keys. 
    Takes a list of the old keys and a list of the new keys as input."""
    # loop through all the old keys and replace them one by one interactively or automatically
    if interactive == True:
        for i in range(len(old_keys)):
            # ask the user if they want to replace the key
            replace_key = input(
                f"What do you want to replace the key {old_keys[i]} with? (press enter to use old key) "
            )
            # if the user wants to replace the key, replace it
            if replace_key == "":
                data[new_keys[i]] = data.pop(old_keys[i])
            # if the user doesn't want to replace the key, skip it
            else:
                pass

    else:
        # loop through all the old keys and replace them one by one
        for i in range(len(old_keys)):
            data[new_keys[i]] = data.pop(old_keys[i])
        return data


def plot_survey_area(
    data,
    remove_idxs,
    survey_area_corners_lonlat,
    cmap_name="spring",
    savefigure="None",
    show_plot=True,
):
    """Function for plotting the survey area and the data points within the survey area.
    Takes the data dictionary, the indices of the data points within the survey area and
    the survey area corners as input."""

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    colormap_name = cmap_name

    ax[0].scatter(data["lon"], data["lat"], c=data["flow"], s=1, cmap=colormap_name)
    ax[0].set_title("All data")
    ax[0].set_xlabel("Longitude")
    ax[0].set_ylabel("Latitude")

    # in the plot with only the data within the survey area
    ax[1].scatter(
        data["lon"][remove_idxs],
        data["lat"][remove_idxs],
        c=data["flow"][remove_idxs],
        s=1,
        cmap=colormap_name,
    )
    ax[1].set_title("Survey area")
    ax[1].set_xlabel("Longitude")
    # add a rectangle around the survey area
    x = [
        survey_area_corners_lonlat[0][0],
        survey_area_corners_lonlat[1][0],
        survey_area_corners_lonlat[2][0],
        survey_area_corners_lonlat[3][0],
        survey_area_corners_lonlat[0][0],
    ]
    y = [
        survey_area_corners_lonlat[0][1],
        survey_area_corners_lonlat[1][1],
        survey_area_corners_lonlat[2][1],
        survey_area_corners_lonlat[3][1],
        survey_area_corners_lonlat[0][1],
    ]
    ax[0].plot(x, y, c="c")
    ax[1].plot(x, y, c="c")
    # put the y ticks on the right side of the second plot
    ax[1].yaxis.tick_right()
    # choose a colormap that works well with black background
    if savefigure != "None":
        plt.savefig(savefigure, dpi=300, bbox_inches="tight")

    if show_plot == True:
        plt.show()


###########################
####### PLOTTING ##########
###########################

######################################
####### 2 DIMENSIONAL KDE PLOT #######
######################################

# load the data if its not already a pickle file for it
zoom_in = 1
load_data_trigger = 1
if load_data_trigger == 1:
    filename = r"C:\Users\kdo000\Dropbox\post_doc\Marie_project\data\SB_flare_data\CAGE_16_4_Merged_Manuel.xlsx"
    data = load_data(filename, interactive=True)
    data = nan_remover(data, remove_zeros=True)
    # data = only_within_area(data, survey_area_corners_lonlat=[58.5, 59.5])
    # replace keys
    old_keys = list(data.keys())
    new_keys = ["lon", "lat", "utmzone", "UTMx", "UTMy", "flow"]
    data = replace_dict_keys(data, old_keys, new_keys, interactive=False)
    # save the data to a file that is easy to load using pickle
    from pickle import dump

    dump(
        data,
        open(
            r"C:\Users\kdo000\Dropbox\post_doc\Marie_project\data\kde_data\new_data\kde_data.pkl",
            "wb",
        ),
    )
else:
    # load the data
    from pickle import load

    data = load(
        open(
            r"C:\Users\kdo000\Dropbox\post_doc\Marie_project\data\kde_data\new_data\kde_data.pkl",
            "rb",
        )
    )

if zoom_in == 1:
    # zoom in on the data
    survey_area_corners_lonlat = [
        [9.267, 78.65],
        [9.438, 78.661],
        [9.703, 78.492],
        [9.543, 78.481],
    ]
    # position data
    data_pos = dict()
    data_pos["lon"] = data["lon"]
    data_pos["lat"] = data["lat"]
    remove_idxs = only_within_area(data, survey_area_corners_lonlat)

    plot_survey_area(
        data,
        remove_idxs,
        survey_area_corners_lonlat,
        cmap_name="spring",
        savefigure=r"C:\Users\kdo000\Dropbox\post_doc\Marie_project\results\kde_plots\new_kde_plots\survey_area_plot.png",
        show_plot=True,
    )

    # remove all data points outside the survey area
    data = {key: data[key][remove_idxs] for key in data.keys()}

################
### PLOT KDE ###
################

### KDE SETTINGS ###
# Create 2D data of shape (obs, dims)
data_locs = np.array([data["UTMx"], data["UTMy"]]).T
weights = np.array(data["flow"])
# fig = plt.figure(figsize=(18, 6))
grid_points = 2**7  # Grid points in each dimension
N = 26  # Number of contours
bandwidth = 100  # Bandwidth for the KDE

### PLOT SETTINGS ###
colormap_name = "inferno"
# set background color
background_color = []
# select lonlat or utm coordinates
plot_lonlat = 0
# set the whole figure background color
plt.style.use("dark_background")
# use white background instead
# plt.style.use("default")

### COMPUTE KDE ###
kde = TreeKDE(kernel="gaussian", bw=bandwidth, norm=2)
grid, points = kde.fit(data_locs, weights).evaluate(grid_points)
# The grid is of shape (obs, dims), points are of shape (obs, 1)
x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
z = points.reshape(grid_points, grid_points).T

### DEFINE LIMITS AND LEVELS ###
vminlim = -20
vmaxlim = np.max(np.log(z))
# define levels for the contour plot
levels = np.linspace(vminlim, vmaxlim, N)

### MAKE KDE PLOT ###
fig, ax = plt.subplots()
ax.set_title(f"Norm $p={2}$, bw = {bandwidth} , tree kde")
if background_color:
    ax.set_facecolor(background_color)

# Plot the KDE as a contour color plot
sc2d = ax.contourf(
    x, y, np.log(z), levels, cmap=colormap_name, vmin=vminlim, vmax=vmaxlim
)

# Plot the data points themselves
ax.scatter(data["UTMx"], data["UTMy"], c="w", s=1, alpha=0.5)

# Plot some contour lines on top of the KDE plot, here at the orders of magnitude 10^-4, 10^-3, ..., 10^2
min_order = np.floor(np.log10(np.exp(levels.min()))).astype(int)
# Find the order of the maximum value in levels
max_order = np.ceil(np.log10(np.exp(levels.max()))).astype(int)
# Create an array of contour levels that are powers of 10
levels_order = np.logspace(min_order, max_order, num=max_order - min_order + 1)
# Plot the contours at the specified levels
contour_set = ax.contour(
    x,
    y,
    z,  # use the original data, not the logarithm
    levels=levels_order[1:],  # use the levels we defined
    colors="w",
    linestyles="solid",
    linewidths=0.5,
)

# Add labels to the contours with the custom formatter
labels = ax.clabel(
    contour_set,
    inline=True,
    fontsize=6,
    fmt=ticker.FuncFormatter(lambda x, pos: f"{x:.1e}"),
)

# Create a "fake" colorbar that matches the range of levels
vmin, vmax = levels.min(), levels.max()
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colormap_name)
# Create the colorbar
cb = plt.colorbar(mappable)
# Set the ticks and labels on the colorbar
cb.set_ticks(np.linspace(vmin, vmax, 6))
cb.set_ticklabels([f"{np.exp(i):.2e}" for i in np.linspace(vmin, vmax, 6)])
# Label the colorbar
cb.set_label("Gas flow [ml/min/m$^2$]")

# Replace ticks with latlon coordinates

if plot_lonlat == 1:
    # get the x ticks (utm x coordinates)
    xticks = ax.get_xticks()
    # get the y ticks (utm y coordinates)
    yticks = ax.get_yticks()
    # convert the x ticks to latlon coordinates
    xticks_latlon = utm.to_latlon(xticks, np.mean(data["UTMy"]))
    # convert the y ticks to latlon coordinates
    yticks_latlon = utm.to_latlon(np.mean(data["UTMx"]), yticks)
    # replace the ticks at the ticklocations in the plot
    ax.set_xticklabels(
        xticks_latlon[1],
    )
    ax.set_yticklabels(yticks_latlon[0])
    # add label
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
else:
    # add label
    ax.set_xlabel("UTMx [m]")
    ax.set_ylabel("UTMy [m]")


# save the x,y,z data to a xlsx file
df = pd.DataFrame(data=z)
df.to_excel(
    r"C:\Users\kdo000\Dropbox\post_doc\Marie_project\data\kde_data\new_data\kde_data.xlsx",
    index=False,
)
dfx = pd.DataFrame(data=x)
dfx.to_excel(
    r"C:\Users\kdo000\Dropbox\post_doc\Marie_project\data\kde_data\new_data\kde_data_x.xlsx",
    index=False,
)
dfy = pd.DataFrame(data=y)
dfy.to_excel(
    r"C:\Users\kdo000\Dropbox\post_doc\Marie_project\data\kde_data\new_data\kde_data_y.xlsx",
    index=False,
)
# save the figure
plt.savefig(
    r"C:\Users\kdo000\Dropbox\post_doc\Marie_project\results\kde_plots\new_kde_plots\kde_plot.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

###############################
# KDE HISTOGRAM
###############################

# mirror the data
# make a sorted array of the data
data_sorted = np.sort(data["flow"], axis=0)
# make a mirrored array
data_mirrored = np.concatenate((data_sorted, -data_sorted))
# do a kernel density estimate of the mirrored data
# Create 2D data of shape (obs, dims)
# data_mirrored = np.array([data_mirrored, data_mirrored]).T.flatten()
# the bandwidth to be a function of the flow rate
# make a list of bandwidths
bandwidths = []
for i in range(len(data_mirrored)):
    bandwidths.append(np.abs(data_mirrored[i]) * 0.1)

# KDE HISTOGRAM
# determine the number of bins to be used
# set resolution of the histogram
grid_points = 5000  # Grid points in each dimension

x, y = (
    TreeKDE(kernel="epa", bw=bandwidths, norm=2)
    .fit(data_mirrored)
    .evaluate(grid_points)
)
plt.plot(x, y)
plt.xlim(0, 300)
# plot the bandwithd ona twinx axis

# plot a histogram
plt.hist(data_mirrored, bins=2000, density=True)
plt.xlim(0, 150)
# plot legend
plt.legend(["Kernel density estimate", "Histogram"])
plt.ylabel("Probability density (kde)")
plt.xlabel("Flowrate per single seep")
# plt.show()

# save the figure
plt.savefig(
    r"C:\Users\kdo000\Dropbox\post_doc\Marie_project\results\kde_plots\new_kde_plots\kde_histogram.png",
    dpi=300,
    bbox_inches="tight",
)
