# tracking seabirds

An example notebook demoing how to transform 2D keypoints to a new coordinate system using `movement`.

The data is a short clip of tracked seabirds flying around a boat. The notebook shows how to express the data in a coordinate system aligned with the boat.

## Installation

Create a conda environment and install the latest version of `movement`:

```sh
conda create -n movement-env -c conda-forge movement
```

Activate the environment:
```sh
conda activate movement-env
```

Install additional dependencies for Jupyter Notebooks:

```sh
pip install jupyter jupyterlab ipympl
```

If you wish to use the `movement` GUI, which additionally requires [napari](napari:),
you should replace the first command with:
```sh
conda create -n movement-env -c conda-forge movement napari pyqt
```
You may exchange `pyqt` for `pyside2` if you prefer.
See [napari's installation guide](napari:tutorials/fundamentals/installation.html)
for more information on the backends.


Please refer to the [movement installation guide](https://movement.neuroinformatics.dev/user_guide/installation.html) for more details.

To open the notebook in Jupyter Lab, run from the root of this repository:

```bash
jupyter lab notebook_seabirds.ipynb
```

Alternatively, you can run the notebook in VS Code using the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).


## Data

## Postprocessing steps

The DLC-predicted trajectories are expressed in the camera drone's coordinate system. Before transforming them to the boat's coordinate system, we applied the following postprocessing steps to the raw DLC data:

1. **Manual correction of ID errors** using a custom napari GUI.

2. **Trajectory splitting based on gaps and jumps.**
DLC assigns one of five predefined IDs to the birds visible in each frame. However, if a bird leaves and later re-enters the frame, DLC may reuse the same ID for a different bird. To prevent this, we reset an individual's ID whenever there is a sufficiently large temporal gap in its trajectory and the position after the gap is far from the last valid point before it.

3. **Filtering out low-confidence points and linear interpolation.**
We discard points with a DLC confidence score below 0.6, then linearly interpolate across the resulting gaps.

These steps are collected in the notebook [`demo_notebook.ipynb`](https://github.com/anna-teruel/seabirds-coord-transform/blob/main/demo_notebook.ipynb).


## Boat coordinate system

The notebook [`notebook_boat_coord_system.py`](notebook_boat_coord_system.py) transforms the cleaned DLC bird trajectories from the image coordinate system (ICS) to a boat coordinate system (BCS). The steps are:

1. **Separate bird and boat data** from the input DLC file into two `movement` datasets.

2. **Filter and interpolate boat keypoints**: low-confidence detections (below 0.5) are set to NaN and linearly interpolated over time.

3. **Define the boat coordinate system (BCS) per frame**:
   - Origin: mean of all boat keypoints per frame.
   - Y-axis: unit vector from the boat centroid to the boat tip.
   - X-axis: perpendicular to the y-axis, pointing to the right side of the boat.
   - Z-axis: cross product of the x and y axes, it is positive pointing out of the image plane.

4. **Express bird and boat trajectories in BCS** using the per-frame rotation matrix derived from the BCS axes definition.

5. **Scale trajectories to meters** using the boat length per frame as reference (vessel length: 8.55 m).

6. **Interpolate and smooth bird centroid trajectories** using monotone cubic interpolation (PCHIP) followed by a rolling median filter (window = 15 frames).

7. **Export** the transformed data as `.nc` files (loadable in napari), `.csv` files, and the plot of the raw (unsmoothed) data as a plotly `.html` file.
