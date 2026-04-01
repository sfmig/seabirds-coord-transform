"""A notebook to express cleaned DLC bird trajectories in a boat coordinate system.

Requirements: following installation instructions for `movement`
https://movement.neuroinformatics.dev/latest/user_guide/installation.html

Also install: plotly

Then run this notebook in that conda environment.

"""
# %%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    rolling_filter,
    savgol_filter,
)
from movement.io import load_poses, save_poses
from movement.kinematics import compute_pairwise_distances
from movement.plots import plot_occupancy
from movement.utils.vector import compute_norm, convert_to_unit
from scipy.spatial.transform import Rotation as R

# Hide attributes globally
xr.set_options(display_expand_attrs=False)

# %%%%%%%%%%%%%%%%%%%%%%%
# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%
# Input data paths
notebook_path = Path(__file__).resolve()
data_dir = notebook_path.parent / "data"
filepath = (
    data_dir
    / "trayectorias_AT"
    / "FILE00009_sDLC_DekrW32_seabirdNov6shuffle1_snapshot_170_el_filtered_split_interpolated.h5"
)
output_dir = notebook_path.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# Vessel size: 8.55 x 2.95 m
boat_max_length_in_m = 8.55  # m
boat_max_width_in_m = 2.95  # m

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper functions


def get_data_for_load_from_numpy(df):
    """Get array from dataframe to use "from numpy" function"""
    list_individuals = sorted(df.columns.get_level_values("individuals").unique())
    list_keypoints = sorted(df.columns.get_level_values("bodyparts").unique())
    n_keypoints = len(list_keypoints)
    n_individuals = len(list_individuals)

    # position array
    df_position = df.drop(columns=[col for col in df.columns if "likelihood" in col])

    # get number of frames
    position_array = df_position.to_numpy()
    position_array = position_array.reshape(
        df.shape[0],
        2,
        n_keypoints,
        n_individuals,
        order="F",
    )

    # confidence array
    df_confidence = df.drop(
        columns=[col for col in df.columns if "likelihood" not in col]
    )
    confidence_array = df_confidence.to_numpy()
    confidence_array = confidence_array.reshape(
        df.shape[0],
        n_keypoints,
        n_individuals,
        order="F",
    )

    return position_array, confidence_array, list_individuals, list_keypoints


def add_z_coord_to_position_array(position_array):
    """Add z coordinate to position array"""
    return xr.concat(
        [
            position_array,
            xr.full_like(
                position_array.sel(space="x"),
                0,
            ).expand_dims(space=["z"]),
        ],
        dim="space",
    )


def export_dataarray_as_csv(da_position, output_path):
    """Export as a tidy dataframe with x,y separate columns."""
    df = da_position.to_dataframe().reset_index()

    # drop rows with NaN positions
    df = df.dropna(subset=["position"])

    # Pivot space to get x and y as separate columns
    columns_to_keep = [idx for idx in df.columns if idx not in ["space", "position"]]
    df_wide = df.pivot(
        index=columns_to_keep,
        columns="space",
        values="position",
    ).reset_index()

    # Flatten column names
    df_wide.columns.name = None

    # Export to CSV
    df_wide.to_csv(output_path, index=False)

    return output_path


def export_as_ds(da_position, da_confidence, output_path):
    """Export dataset with given position array and nan confidence."""
    ds = xr.Dataset(
        {
            "position": da_position,
            "confidence": da_confidence,
            # xr.full_like(da_position.isel(space=0, drop=True), np.nan),
        }
    )
    ds.attrs["ds_type"] = "poses"
    ds.to_netcdf(output_path)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read input data as pandas dataframe
df = pd.read_hdf(filepath)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get dataset with bird data only
if (filepath.parent / (filepath.stem + "_birds.h5")).exists():
    ds_birds = load_poses.from_dlc_file(filepath.parent / (filepath.stem + "_birds.h5"))
else:
    columns_to_drop = [
        col for col in df.columns if col[-2] in ["boatBL", "boatBR", "boatTip"]
    ]
    df_birds = df.drop(columns=columns_to_drop)

    position_array, confidence_array, list_individuals, list_keypoints = (
        get_data_for_load_from_numpy(df_birds)
    )

    ds_birds = load_poses.from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=list_individuals,
        keypoint_names=list_keypoints,
        # fps=30,
    )

    # export to file importable in napari
    # To visualise exported file, follow this guide:
    # https://movement.neuroinformatics.dev/user_guide/gui.html
    save_poses.to_dlc_file(ds_birds, filepath.parent / (filepath.stem + "_birds.h5"))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get dataset with boat data only
if (filepath.parent / (filepath.stem + "_boat.h5")).exists():
    ds_boat = load_poses.from_dlc_file(filepath.parent / (filepath.stem + "_boat.h5"))
else:
    columns_to_keep = [
        col for col in df.columns if col[-2] in ["boatBL", "boatBR", "boatTip"]
    ]
    df_boat = df.loc[:, columns_to_keep]

    position_array, confidence_array, list_individuals, list_keypoints = (
        get_data_for_load_from_numpy(df_boat)
    )

    ds_boat = load_poses.from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=list_individuals,
        keypoint_names=list_keypoints,
        # fps=30,
    )

    # Rename individual name for boat
    # (it is set as "bird24")
    ds_boat["individuals"] = ["boat"]

    # export for importable in napari
    save_poses.to_dlc_file(
        ds_boat, filepath.parent / (filepath.stem + "_boat.h5"), split_individuals=False
    )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Filter low-confidence values in boat keypoint trajectories and interpolate
# (values below the threshold are set to nan)
confidence_threshold = 0.5
boat_position = filter_by_confidence(
    ds_boat.position, ds_boat.confidence, threshold=confidence_threshold
)

# Linearly interpolate gaps in boat trajectory
boat_position_interp = interpolate_over_time(
    boat_position,
    method="linear",
    print_report=True,
)  # there should be no nans after interp


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute axes of BCS (boat coordinate system)
# - origin : centroid of all boat keypoints per frame
# - y-axis: vector from boat centroid to boat tip keypoint
# - x-axis: perpendicular to y-axis, points to right side of the boat

# compute origin
boat_position_3d = add_z_coord_to_position_array(boat_position_interp)
boat_centroid_3d = boat_position_3d.mean("keypoints")
boat_centroid_3d = boat_centroid_3d.drop_vars("individuals").squeeze()

# compute BCS y-axis unit vector in image coordinate system (ICS)
boat_y_axis_3d = (
    convert_to_unit(boat_position_3d.sel(keypoints="boatTip") - boat_centroid_3d)
    .drop_vars(["keypoints"])
    .drop_vars("individuals")
    .squeeze()
)

# compute BCS z-axis in image coordinate system (ICS)
# (negative of ICS z-axis, which is positive going into the paper)
boat_z_axis_3d = xr.DataArray(data=[0, 0, -1], coords={"space": ["x", "y", "z"]})

# compute boat x-axis in image coordinate system (ICS)
boat_x_axis_3d = xr.cross(boat_y_axis_3d, boat_z_axis_3d, dim="space")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute rotation matrix from BCS axes to ICS axes == change of basis
# matrix from ICS coordinates to BCS coordinates
# R.apply(x_BCS) = x_ICS

# The rotation is approximately a 180 deg rotation
# about the x=y diagonal (axis x=1, y=1, z=0). It essentially
# swaps x and y and flips z

rotation2boat = xr.apply_ufunc(
    lambda xv, yv, zv: R.from_matrix(np.array([xv, yv, zv])),
    boat_x_axis_3d,
    boat_y_axis_3d,
    boat_z_axis_3d,
    input_core_dims=[["space"], ["space"], ["space"]],
    vectorize=True,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute bird keypoints in BCS (translated and rotated)
birds_position_3d = add_z_coord_to_position_array(ds_birds.position)

birds_position_3d_BCS = xr.apply_ufunc(
    lambda rot, trans, vec: rot.apply(vec - trans),
    rotation2boat,  # rotation to BCS
    boat_centroid_3d,  # translation to BCS
    birds_position_3d,  # trajectories in ICS
    input_core_dims=[[], ["space"], ["space"]],
    output_core_dims=[["space"]],
    vectorize=True,
)

# drop z coordinate for clarity
birds_position_BCS = birds_position_3d_BCS.drop_sel(space="z")

# reorder coordinates (space is moved last after apply_ufunc)
birds_position_BCS.transpose("time", "space", "keypoints", "individuals")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Apply same transform to boat points
boat_position_3d_BCS = xr.apply_ufunc(
    lambda rot, trans, vec: rot.apply(vec - trans),
    rotation2boat,  # rot
    boat_centroid_3d,  # trans
    boat_position_3d,  # vec
    input_core_dims=[[], ["space"], ["space"]],
    output_core_dims=[["space"]],
    vectorize=True,
)

# drop z coordinate for clarity
boat_position_BCS = boat_position_3d_BCS.drop_sel(space="z")

# reorder coordinates (space is moved last after apply_ufunc)
boat_position_BCS.transpose("time", "space", "keypoints", "individuals")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Apply scaling

# Compute boat width per frame in pixels
boat_width = compute_pairwise_distances(
    boat_position_BCS,
    dim="keypoints",
    pairs={"boatBL": "boatBR"},
)

# Compute boat length per frame in pixels
boat_midpoint_BL_BR = boat_position_BCS.sel(
    keypoints=["boatBL", "boatBR"],
).mean(dim="keypoints")

boat_length = compute_norm(
    boat_position_BCS.sel(keypoints="boatTip") - boat_midpoint_BL_BR
).squeeze()


# check width, length variation with time
plt.figure()
boat_width.plot(label="width")
boat_length.plot(label="length")
plt.xlabel("time (frames)")
plt.ylabel("distance (pixels)")
plt.legend()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Express spatial coordinates in meters

# We use boat length to scale the data per frame
# (boat_width looks a bit nosier)
scale_factor = boat_max_length_in_m / boat_length

# Express boat and bird coords in meters
boat_position_BCS_in_m = boat_position_BCS * scale_factor
birds_position_BCS_in_m = birds_position_BCS * scale_factor

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Interpolate and smooth bird centroid trajectories

# Interpolation
# - Simplest: linear
# - For continuous 1st and 2nd derivative (speed): cubic spline -- but oscillations occur
# - For continous 1st derivative and less poly wiggle: monotone cubic interpolants
#
# both akima and pchip monotone cubic interpolants: these are constructed to be only once
# continuously differentiable, and attempt to preserve the local shape
# implied by the data.
# https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#monotone-interpolants

# interpolate
bird_centroid_BCS_in_m = birds_position_BCS_in_m.mean(dim='keypoints')
birds_centroid_BCS_in_m_interp = interpolate_over_time(
    bird_centroid_BCS_in_m, method="pchip"
)

# smooth with rolling median filter
birds_centroid_BCS_in_m_interp_smooth = rolling_filter(
    birds_centroid_BCS_in_m_interp,
    window=15,  # frames (video is 30fps)
)

# alternatively: smooth with SG filter
# https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
# birds_position_BCS_in_m_smooth = savgol_filter(
#     birds_position_BCS_in_m_interp,
#     window=15, # frames (video is 30fps)
#     polyorder=1,
# )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot centroid bird trajectories in BCS (plotly with WebGL)

data_options = {
    "": birds_position_BCS_in_m,
    "_interp_smooth": birds_centroid_BCS_in_m_interp_smooth,
}

# Select bird data to plot
tag = ""  # "" for raw data, "_interp_smooth" for interpolated+smooth
position_da = data_options[tag]

# Select a time slice for clarity
max_frame = position_da.time.max().values.item()
time_slice = slice(0, max_frame)

# prepare colors by individual
colors = np.vstack([plt.get_cmap("tab20").colors, plt.get_cmap("tab20b").colors])
color_array = colors[np.arange(len(position_da.individuals)) % len(colors)]

# plot bird data
fig_plotly = go.Figure()
for i, ind in enumerate(position_da.individuals):
    # compute centroid x,y coordinates
    x = position_da.sel(time=time_slice, individuals=ind, space="x")
    y = position_da.sel(time=time_slice, individuals=ind, space="y")
    if "keypoints" in x.dims:
        x = x.mean("keypoints")
        y = y.mean("keypoints")
    x, y = x.values, y.values

    rgb = color_array[i]
    color_str = f"rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})"
    fig_plotly.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=3, color=color_str),
            name=ind.item(),
        )
    )

# plot boat centroid, color by frame
# squeeze individuals dim (boat has a single "boat" individual)
boat_plot = boat_position_BCS_in_m.sel(time=time_slice).squeeze("individuals")
frame_idx = np.arange(time_slice.stop - time_slice.start + 1)
fig_plotly.add_trace(
    go.Scattergl(
        x=boat_plot.sel(space="x").mean("keypoints").values,
        y=boat_plot.sel(space="y").mean("keypoints").values,
        mode="markers",
        marker=dict(
            size=4,
            color=frame_idx,
            colorscale="Plasma",
            symbol="star",
            colorbar=dict(title="frames", x=-0.15),
        ),
        name="boat centroid",
    )
)

# plot boat keypoints, color by frame
for boat_keypoint in ["boatTip", "boatBL", "boatBR"]:
    fig_plotly.add_trace(
        go.Scattergl(
            x=boat_plot.sel(keypoints=boat_keypoint, space="x").values,
            y=boat_plot.sel(keypoints=boat_keypoint, space="y").values,
            mode="markers",
            marker=dict(size=6, color=frame_idx, colorscale="Plasma", showscale=False),
            name=boat_keypoint,
            showlegend=False,
        )
    )

# axes
fig_plotly.update_layout(
    xaxis_title="x_BCS (m)",
    yaxis_title="y_BCS (m)",
    yaxis_scaleanchor="x",
    yaxis_scaleratio=1,
    legend=dict(x=1.15, y=1, xanchor="left"),
    template="plotly_white",
)


fig_plotly.show()

fig_plotly.write_html(output_dir / f"bird_trajectories_BCS_centroid{tag}.html")
# %%%%%%%%%%%%%%%%%%%
# Plot heatmap with movement

# Set extension of the data
# birds_position_BCS_in_m.sel(space="x").min().values
# birds_position_BCS_in_m.sel(space="x").max().values
# birds_position_BCS_in_m.sel(space="y").min().values
# birds_position_BCS_in_m.sel(space="y").max().values
xmin, xmax = -70, 70
ymin, ymax = -115, 80

bin_edges_x = np.arange(xmin, xmax + 1, 1)  # 1m wide
bin_edges_y = np.arange(ymin, ymax + 1, 1)  # 1m wide

# excluding bird1 and 20
birds_to_exclude = ["bird1", "bird20"]

fig, ax, hist = plot_occupancy(
    birds_position_BCS_in_m,
    range=[[xmin, xmax], [ymin, ymax]],
    bins=[bin_edges_x, bin_edges_y],
    individuals=[
        ind.item()
        for ind in birds_position_BCS_in_m.individuals.values
        if ind not in birds_to_exclude
    ],
)

# plot mean position of boat keypoints in red
ax.scatter(
    boat_position_BCS_in_m.sel(space='x').mean(dim='time').values,
    boat_position_BCS_in_m.sel(space='y').mean(dim='time').values,
    10,
    c='r'
)

ax.set_aspect("equal")
ax.set_xlabel("x_BCS (m)")
ax.set_ylabel("y_BCS (m)")
fig.axes[-1].set_ylabel("counts")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save movement datasets as .nc files loadable in napari

# Export bird data
export_as_ds(
    birds_position_BCS_in_m, ds_birds.confidence, output_dir / "ds_birds_BCS_in_m.nc"
)

# Export boat data
export_as_ds(
    boat_position_BCS_in_m, ds_boat.confidence, output_dir / "ds_boat_BCS_in_m.nc"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export csvs

# Save bird trajectories in BCS (all keypoints)
export_dataarray_as_csv(
    birds_position_BCS_in_m, output_dir / "birds_position_BCS_in_m.csv"
)

# Save centroid (mean of all keypoints per frame)
export_dataarray_as_csv(
    birds_position_BCS_in_m.mean(dim="keypoints"),
    output_dir / "birds_position_BCS_in_m_centroid.csv",
)

# Save centroid interpolated and smoothed
export_dataarray_as_csv(
    birds_centroid_BCS_in_m_interp_smooth,
    output_dir / "birds_position_BCS_in_m_centroid_interp_smooth.csv",
)

# Save boat keypoints per frame in BCS
export_dataarray_as_csv(
    boat_position_BCS_in_m, output_dir / "boat_position_BCS_in_m.csv"
)

# %%
