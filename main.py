import os
import numpy as np
import skimage
from skimage.transform import rescale
from skimage.util import crop
import csv
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations
from Metadata import Metadata
from Mission import Mission
from Image import Image
from Card import Card

# header[0]['GS_V3_PA'] = rotation of image? up to 14 digits past .
# header[0]['GS_RA'] = guide star right ascension?
# header[0]['GS_DEC'] = guide star declination?
# header[1]['RA_V1']
# header[1]['DEC_V1']
# header[1]['PA_V3']
# header[1]['PIXAR_SR'] = Nominal pixel area in steradians
# header[1]['PIXAR_A2'] = Nominal pixel area in arcsec^2

def main():
    title = 'A Strongly Magnified Individual Star and Parsec-Scale Clusters Observed in the First Billion Years at z = 6'
    mission = Mission(title)
    downloaded_metadata = download_mission(title)
    print("\nOperating on FITS files...")
    for meta in downloaded_metadata:
        # https://docs.astropy.org/en/stable/io/fits/index.html
        hdul = fits.open(meta.fits_path)
        print(meta.fits_path)
        # hdul.info()
        write_header(meta.fits_path, 0)
        write_header(meta.fits_path, 1)

        for hdu in ['SCI', 'ERR', 'WHT', 'VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT']:
            cards = []
            for c in hdul[hdu].header.cards:
                if c.keyword:
                    cards.append(c)
            this_image = Image(meta.row['filters'], hdu, hdul[hdu].data, cards, meta)
            mission.add_image(this_image)
        hdul.close()

    for im in mission.images:
        print(im)
    
    images = mission.search(filter='F277W', data_type='SCI')
    out = np.dstack((images[0].image_data, images[1].image_data, np.zeros(images[1].image_data.shape)))

    # # False Blue = F090W, False Green = F200W, False Red = F277W
    # new_image = np.dstack((sci_image['F090W'], sci_image['F200W'], sci_image['F277W']))
    fig, ax = plt.subplots()
    # ax.imshow(out, cmap='gray', vmin=0, vmax=1) # black white
    ax.imshow(out) # color
    ax.set_title(title)
    plt.show()
    return

def write_header(fits_path, index):
    hdul = fits.open(fits_path)
    file = open(f'{str(fits_path)}.{index}.txt', 'w')
    for c in hdul[index].header.cards:
        file.write(f'{str(c)}\n')
    file.close()
    hdul.close()

def download_mission(title):
    print(f"Querying for: {title}")
    obs_table = Observations.query_criteria(
        obs_title=[title],
        # objectname="Crab",
        dataRights=["PUBLIC"],
        # instrument_name=["NIRCAM"], # NIRCAM, NIRSPEC, MIRI, NIRISS
        obs_collection=["JWST"],
        dataproduct_type=["image"],
        intentType=["science"],
        calib_level=[3])
    # print(type(obs_table))
    # data_products = Observations.get_product_list(obs_table)
    # print(obs_table.keys())

    all_metadata = []
    # For each product in the query output
    for product in obs_table:
        # Create a dictionary object that contains all data from that row
        row = {}
        for index, entry in enumerate(obs_table.keys()):
            row[entry] = product[index]
        this_metadata = Metadata(row)

        # For the purpose of this project, we are only selecting the
        # data that uses filters in NIRCAM observations
        if row['filters'] != 'CLEAR':
            # get directories for files
            base_dir = row['target_name']
            data_path = os.path.join(base_dir, "data")
            preview_path = os.path.join(base_dir, "preview")

            # create directories if they do not exist
            print(f"\nChecking if mission directory exists for: {base_dir}")
            if not os.path.exists(os.path.join(os.getcwd(), base_dir)):
                os.makedirs(os.path.join(os.getcwd(), base_dir))
                os.makedirs(os.path.join(os.getcwd(), data_path))
                os.makedirs(os.path.join(os.getcwd(), preview_path))
                print("Directories created")

            # check if the file for this item exists
            fits_file = os.path.normpath(f'{base_dir}\data\\' + os.path.basename(row['dataURL']))
            preview_file = os.path.normpath(f'{base_dir}\preview\\' + os.path.basename(row['jpegURL']))
            print(f'Checking if {fits_file} exists...')
            if not os.path.isfile(fits_file) or not os.path.isfile(preview_file):
                # download the data and preview images if they do not exist
                # https://mast.stsci.edu/portal/Download/file?uri=mast:JWST/product/jw02282-o120_t001_nircam_clear-f277w_i2d.jpg
                # https://astroquery.readthedocs.io/en/latest/mast/mast.html
                os.chdir(data_path)
                print(f"Checking for {os.path.basename(row['dataURL'])} in {os.getcwd()}")
                (status, msg, url) = Observations.download_file(row['dataURL'])

                os.chdir(os.path.join("..", "..", preview_path))
                print(f"Checking for {os.path.basename(row['jpegURL'])} in {os.getcwd()}")
                (status, msg, url) = Observations.download_file(row['jpegURL'])
                os.chdir(os.path.join("..", ".."))
                print(f"Ending in {os.getcwd()}")

            # if the data and preview file exist, assume they are all good and set the internal path to those files
            else:
                print(f'Assuming {fits_file} exists already')

            this_metadata.set_files(
                f"{base_dir}\data\{os.path.basename(row['dataURL'])}",
                f"{base_dir}\preview\{os.path.basename(row['jpegURL'])}"
            )
            # add this information to the return object
            all_metadata.append(this_metadata)
    return all_metadata

if __name__ == '__main__':
    main()