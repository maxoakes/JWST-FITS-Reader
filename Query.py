import sys
import os
import datetime
import numpy as np
import skimage
import skimage.transform
from skimage.io import imsave, imread
from skimage.util import crop
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations
from ImageDescription import ImageDescription
from Mission import Mission
from Card import Card
from RawImage import RawImage

class Query:
    def run_query(params: dict, do_print: bool):
        print(f"Querying for: {params}")
        base_params = {
            'dataRights': ["PUBLIC"],
            'obs_collection': ["JWST"],
            'dataproduct_type': ["image"],
            'intentType': ["science"],
            'instrument_name': ['NIRCAM'],
            'calib_level': [3]
        }
        base_params.update(params)
        print("Querying with the following parameters:")
        print(base_params)
        obs_table = Observations.query_criteria(**base_params)
        if do_print:
            print(obs_table)

            print(f"Checking if query directory exists...")
            if not os.path.exists(os.path.join(os.getcwd(), "queries")):
                os.makedirs(os.path.join(os.getcwd(), "queries"))
            print("Writing query result to file...")
            now_string = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
            write_path = os.path.join(os.getcwd(), "queries", f'query{now_string}.csv')
            obs_table.write(write_path, format='ascii.csv')
            # data_products = Observations.get_product_list(obs_table)
            # print(obs_table.keys())
        return obs_table

    def download_mission(id: str) -> list[ImageDescription]:
        obs_table = Query.run_query({'proposal_id': [id]}, False)
        all_descriptions = []
        mission_path = ''
        # For each product in the query output
        for product in obs_table:
            # Create a dictionary object that contains all data from that row
            row = {}
            for index, entry in enumerate(obs_table.keys()):
                row[entry] = product[index]
            this_image = ImageDescription(row)

            # For the purpose of this project, we are only selecting the
            # data that uses filters in NIRCAM observations
            if row['filters'] != 'CLEAR':
                # get directories for files
                base_dir = os.path.join("missions", row['target_name'])
                mission_path = base_dir
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
                    print(f"Checking for {os.path.basename(row['dataURL'])} in {os.getcwd()}")
                    Observations.download_file(row['dataURL'], local_path=fits_file)

                    print(f"Checking for {os.path.basename(row['jpegURL'])} in {os.getcwd()}")
                    Observations.download_file(row['jpegURL'], local_path=preview_file)

                # if the data and preview file exist, assume they are all good and set the internal path to those files
                else:
                    print(f'Assuming {fits_file} exists already')

                this_image.set_files(
                    f"{base_dir}\data\{os.path.basename(row['dataURL'])}",
                    f"{base_dir}\preview\{os.path.basename(row['jpegURL'])}"
                )
                # add this information to the return object
                all_descriptions.append(this_image)
        return (all_descriptions, mission_path)