import sys
import os
import datetime
import numpy as np
import skimage
import skimage.transform
from skimage.util import crop
import csv
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations
from ImageDescription import ImageDescription
from Mission import Mission
from Card import Card
from Image import Image

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
            all = mission.search()
            if (len(sys.argv) > 2 and sys.argv[2] == 'headers'):
                print("Writing headers to file")
                for i in all:
                    i.print_fits_table()
                    for t in [0,1,2]:
                        i.write_fits_header_to_file(t)
            
            false_channels: dict[str,Image] = { }
            for color, filter in [('red', 'F090W'), ('blue', 'F356W'), ('green', 'F444W')]:
                t = 'SCI'
                desc = mission.search(filter=filter, sortby='DATE-OBS')[0]
                false_channels[color] = Image(filter, t, desc.get_image(t), 
                    desc.get_card(t,'PA_V3'), desc.get_card(t,'PIXAR_A2'), desc.get_card(t, 'PIXAR_SR'),
                    desc.get_card(t,'RA_V1'), desc.get_card(t,'DEC_V1'), 
                    desc.get_card(t,'CRPIX1'), desc.get_card(t,'CRPIX2'))
                print(false_channels[color])

            # false_channels = rescale_method_1(false_channels['red'], false_channels['green'], false_channels['blue'])
            false_channels = rescale_method_2(false_channels['red'], false_channels['green'], false_channels['blue'])

            fig, ax = plt.subplots()
            # ax.imshow(false_channels['red'].get_image(), cmap='gray', vmin=-1, vmax=20) # black white
            ax.imshow(np.dstack((false_channels['red'].get_image(), false_channels['green'].get_image(), false_channels['blue'].get_image()))) # color
            ax.set_title(mission.get_title())
            plt.show()
            return
    else:
        print("Please run one of the following:")
        print(f"`{sys.argv[0]} query` to search for project missions and print the query result to a file.")
        print(f"`{sys.argv[0]} run` to download and process mission files. Have a proposal ID ready to enter.")
        print("Exiting...")

def rescale_method_1(red: Image, green: Image, blue: Image):
    false_channels = {
        'red': red,
        'green': green,
        'blue': blue
    }

    # scale down all images to the size of the height of the smallest one
    smallestX = np.infty
    for color, image in false_channels.items():
        x = image.get_image_x()
        if x < smallestX:
            smallestX = x
    print(f"Smallest x is {smallestX}")

    # get scaled-down images
    for color, image in false_channels.items():
        if image.get_image_x() != smallestX:
            scale_factor = smallestX / image.get_image_x()
            rescaled = skimage.transform.rescale(image.get_image(), scale_factor, anti_aliasing=False)
            
            image.update_image(rescaled)
            image.update_data(image.get_rotation(), 
                image.get_arcsec_per_pixel() / scale_factor, 
                scale_factor * image.get_center_x(),
                scale_factor * image.get_center_y())
        print(f"rescaled: {image.get_image().shape}, {image.get_arcsec_per_pixel()}")

    # find the largest (rescaled) y size, to pad the ones that are short of that
    largestY = 0
    for color, image in false_channels.items():
        y = image.get_image_y()
        if y > largestY:
            largestY = y
    print(f"Largest y is {largestY}")

    # pad images so they are the same size
    for color, image in false_channels.items():
        if image.get_image_y() != largestY:
            import cv2
            padded = cv2.copyMakeBorder(image.get_image(), 0, largestY - image.get_image_y(), 0, 0, cv2.BORDER_REFLECT)
            image.update_image(padded)
        print(f"padded: {image.get_image().shape}, {image.get_arcsec_per_pixel()}")
    return false_channels

def rescale_method_2(red: Image, green: Image, blue: Image):
    false_channels = {
        'red': red,
        'green': green,
        'blue': blue
    }

    # scale down all images to the size of the height of the smallest one
    largest_arcsec_per_pixel = 0
    for color, image in false_channels.items():
        app = image.get_arcsec_per_pixel()
        if app > largest_arcsec_per_pixel:
            largest_arcsec_per_pixel = app
    print(f"Largest x is {largest_arcsec_per_pixel}")

    # get scaled-down images
    for color, image in false_channels.items():
        if not np.isclose(image.get_arcsec_per_pixel(), largest_arcsec_per_pixel):
            scale_factor = largest_arcsec_per_pixel / image.get_arcsec_per_pixel()
            rescaled = skimage.transform.rescale(image.get_image(), scale_factor, anti_aliasing=False)
            
            image.update_image(rescaled)
            image.update_data(image.get_rotation(), 
                image.get_arcsec_per_pixel() / scale_factor, 
                scale_factor * image.get_center_x(),
                scale_factor * image.get_center_y())
        print(f"rescaled: {image.get_image().shape}, {image.get_arcsec_per_pixel()}")

    # find the largest (rescaled) y size, to pad the ones that are short of that
    largestY = 0
    for color, image in false_channels.items():
        y = image.get_image_y()
        if y > largestY:
            largestY = y
    print(f"Largest y is {largestY}")

    # pad images so they are the same size
    for color, image in false_channels.items():
        if image.get_image_y() != largestY:
            import cv2
            padded = cv2.copyMakeBorder(image.get_image(), 0, largestY - image.get_image_y(), 0, 0, cv2.BORDER_REFLECT)
            image.update_image(padded)
        print(f"padded: {image.get_image().shape}, {image.get_arcsec_per_pixel()}")
    return false_channels

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