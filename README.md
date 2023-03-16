Me trying to learn about how JWST images are processed.

## Requires several libraries:
* numpy
* skimage
* matplotlib
* astroquery
* PIL
* scipy
* astropy
* astroalign

Some many require downloading the associated wheel, as using `pip` did not work for all of these.

## To run:
### `py main.py query`
* Enter search parameters when asked
* A CSV will be downloaded to the `./query` directory
* Get the proposal ID of the items that you want to download
### `py main.py run`
* When prompted, enter the proposal ID of the items that you want. If you skip and press enter on an empty input, a default proposal ID of 2733 will be used, and the script will perform operations on the nebula NGC 3132
* You can enter parameters when prompted, or leave them blank to use default selections.
* All steps of the transformation and the final product will be saved to files in the appropriate directories under `./missions/<target name>/`