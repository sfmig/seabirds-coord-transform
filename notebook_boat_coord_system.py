"""A notebook to express cleaned DLC trajectories from birds in a boat coordinate system.

Requirements: following installation instructions for `movement`
https://movement.neuroinformatics.dev/latest/user_guide/installation.html

Then run this notebook in that conda environment.

"""
# %%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from movement.filtering import filter_by_confidence, interpolate_over_time
from movement.io import load_poses, save_poses
from movement.kinematics import compute_pairwise_distances
from movement.utils.reports import report_nan_values
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
# Filter low-confidence values in boat keypoint trajectories
# (values below the threshold are set to nan)
confidence_threshold = 0.5
boat_position = filter_by_confidence(
    ds_boat.position, ds_boat.confidence, threshold=confidence_threshold
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Linearly interpolate boat points
# (gaps with nan are linearly inteprolated)
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

# compute boat y-axis unit vector
boat_y_axis_3d = (
    convert_to_unit(boat_position_3d.sel(keypoints="boatTip") - boat_centroid_3d)
    .drop_vars(["keypoints"])
    .drop_vars("individuals")
    .squeeze()
)

# compute boat z-axis
# (negative of ICS z-axis, which is positive going into the paper)
boat_z_axis_3d = xr.DataArray(data=[0, 0, -1], coords={"space": ["x", "y", "z"]})

# compute x-axis
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

# drop z coordinate
birds_position_BCS = birds_position_3d_BCS.drop_sel(space="z")

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

# drop z coordinate
boat_position_BCS = boat_position_3d_BCS.drop_sel(space="z")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Apply scaling

# Compute boat width per frame in pixels
boat_width = compute_pairwise_distances(
    boat_position_BCS,
    dim="keypoints",
    pairs={"boatBL": "boatBR"},
)
# boat_width.name = "position"

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

# We use boat length to scale the data
# (boat_width looks a bit nosier)
scale_factor = boat_max_length_in_m / boat_length

# Express boat and bird coords in meters
boat_position_BCS_in_m = boat_position_BCS * scale_factor
birds_position_BCS_in_m = birds_position_BCS * scale_factor


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot bird trajectories in BCS

# Select a time slice for clarity
time_slice = slice(0, 8999)

fig, ax = plt.subplots(1, 1)

# plot bird data and color by individual
# cmap = plt.get_cmap("tab20")  # + plt.get_cmap("tab20")
# color_array = cmap(np.arange(len(birds_position_BCS_in_m.individuals)))
colors = np.vstack([plt.get_cmap("tab20").colors, plt.get_cmap("tab20b").colors])
color_array = colors[np.arange(len(birds_position_BCS_in_m.individuals)) % len(colors)]

for i, ind in enumerate(birds_position_BCS_in_m.individuals):
    # bird centroids
    ax.scatter(
        birds_position_BCS_in_m.sel(time=time_slice, individuals=ind, space="x").mean(
            "keypoints"
        ),
        birds_position_BCS_in_m.sel(time=time_slice, individuals=ind, space="y").mean(
            "keypoints"
        ),
        5,
        color=color_array[i],
        label=ind.item(),
    )

ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1))

# plot boat centroid
sc = ax.scatter(
    boat_position_BCS_in_m.sel(time=time_slice, space="x").mean("keypoints"),
    boat_position_BCS_in_m.sel(time=time_slice, space="y").mean("keypoints"),
    10,
    c=np.arange(time_slice.stop - time_slice.start +1),
    cmap="plasma",
    marker="*",
)

# plot boat keypoints in time
for boat_keypoint in ["boatTip", "boatBL", "boatBR"]:
    ax.scatter(
        boat_position_BCS_in_m.sel(time=time_slice, keypoints=boat_keypoint, space="x"),
        boat_position_BCS_in_m.sel(time=time_slice, keypoints=boat_keypoint, space="y"),
        10,
        c=np.arange(time_slice.stop - time_slice.start + 1),
        cmap="plasma",
    )

ax.set_xlabel("x_BCS (m)")
ax.set_ylabel("y_BCS (m)")
ax.set_aspect("equal")

# add colorbar
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("frames")

# put legend outside
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save movement datasets
birds_position_BCS_in_m.to_netcdf(output_dir / "birds_position_BCS_in_m.nc")
boat_position_BCS_in_m.to_netcdf(output_dir / "boat_position_BCS_in_m.nc")
