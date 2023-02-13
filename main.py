import sys
import os
import datetime
import numpy as np
import skimage
from skimage.transform import rescale
from skimage.util import crop
import csv
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations
from ImageDescription import ImageDescription
from Mission import Mission
from Card import Card

# header[0]['GS_V3_PA'] = rotation of image? up to 14 digits past .
# header[0]['GS_RA'] = guide star right ascension?
# header[0]['GS_DEC'] = guide star declination?
# header[1]['RA_V1']
# header[1]['DEC_V1']
# header[1]['PA_V3']
# header[1]['PIXAR_SR'] = Nominal pixel area in steradians
# header[1]['PIXAR_A2'] = Nominal pixel area in arcsec^2

# F090W: low infrared 900nm - wide
# F115W: low infrared 1150nm - wide
# F150W: low infrared 1500nm - wide
# F200W: low infrared 2000nm - wide
# F227W: infrared 2770nm - wide
# F356W: infrared 3560nm - wide
# F444W: infrared 4440nm - wide
# F410M: infrared 4410nm - medium

def main():
    if len(sys.argv) != 1:
        if (sys.argv[1] == 'query'):
            search_params = {}
            for p in [['target_name', 'Name of the object being imaged: '], 
                      ['obs_title', 'Formal title of Project: '],
                      ['proposal_id', 'Proposal ID: ']]:
                inp = input(p[1])
                if (inp):
                    search_params[p[0]] = [inp]
            query(search_params, True)
            return
        if (sys.argv[1] == 'run'):
            default_id = 2282 # 'A Strongly Magnified Individual Star and Parsec-Scale Clusters Observed in the First Billion Years at z = 6'
            # 1415 = mars
            # 2733 = NGC 3132, small nebulae
            id = input("Proposal ID: ")
            if not id:
                id = default_id

            downloaded_items = download_mission(id)
            mission = Mission(downloaded_items[0].get_metadata('target_name'))
            print("\nOperating on FITS files...")
            for item in downloaded_items:
                mission.add_item(item)
            print(f"Initialized {mission}")
            
            # blue_desc = mission.search(filter='F200W', sortby='DATE-OBS')
            # green_desc = mission.search(filter='F150W', sortby='DATE-OBS')
            # red_desc = mission.search(filter='F090W', sortby='DATE-OBS')
            all = mission.search()
            # for d in [blue_desc[0], green_desc[0], red_desc[0]]:
                # print(f"{d.get_filter_name()}: {d.get_card(0,'DATE-OBS')}, {d.get_image('SCI').shape}, {d.get_card(0,'GS_V3_PA')}, {d.get_card(0,'GS_RA')}, {d.get_card(0,'GS_DEC')}, {d.get_card(1,'PIXAR_A2')} ({d.get_card(1,'CRPIX1')},{d.get_card(1,'CRPIX2')})")
            # out = np.dstack((red_desc[0].get_image('SCI'), green_desc[0].get_image('SCI'), green_desc[0].get_image('SCI')))

            # new_image = np.dstack((sci_image['F090W'], sci_image['F200W'], sci_image['F277W']))
            fig, ax = plt.subplots()
            ax.imshow(all[0].get_image('SCI'), cmap='gray', vmin=-1, vmax=50) # black white
            # ax.imshow(out) # color
            ax.set_title(mission.get_title())
            plt.show()
            return
    else:
        print(f"Please run the following: `{sys.argv[0]} (run|query)`. If you want to `run`, please have a proposal ID ready to enter\nExiting...")

def query(params: dict, do_print: bool):
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
    obs_table = query({'proposal_id': [id]}, False)
    all_descriptions = []
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
    return all_descriptions

if __name__ == '__main__':
    main()