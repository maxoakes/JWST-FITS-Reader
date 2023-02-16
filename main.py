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
            default_id = 2733 # NGC 3132, small nebulae
            # 1415 = mars
            # 2282 # 'A Strongly Magnified Individual Star and Parsec-Scale Clusters Observed in the First Billion Years at z = 6'
            id = input("Proposal ID: ")
            if not id:
                id = default_id

            # Acquire imaging data
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
            
            # write working image data to memory
            false_channels: dict[str,Image] = { }
            for color, filter in [('red', 'F090W'), ('blue', 'F187N'), ('green', 'F444W')]:
                t = 'SCI'
                desc = mission.search(filter=filter, sortby='DATE-OBS')[0]
                false_channels[color] = Image(filter, t, desc.get_image(t), 
                    desc.get_card(t,'PA_V3'), desc.get_card(t,'PIXAR_A2'),
                    desc.get_card(t,'RA_V1'), desc.get_card(t,'DEC_V1'), 
                    desc.get_card(t,'CRPIX1'), desc.get_card(t,'CRPIX2'),
                    desc.get_card(t, 'XPOSURE'), 
                    parse_spatial_extent(desc.get_card(t, 'S_REGION')))

            # rescale images so each pixel measures the same area in space
            for color, image in false_channels.items():
                rescale_image(image, 0.07)
                print()
                print(image)
            
            for color, image in false_channels.items():
                mark_center(image)

            min_exposure_time = np.min([false_channels['red'].get_exposure_time(), false_channels['blue'].get_exposure_time(), false_channels['green'].get_exposure_time()])
            print(f"Min Exposure Time is {min_exposure_time} sec")
            # normalize_exposure(min_exposure_time, false_channels['red'], false_channels['blue'], false_channels['green'])

            max_mean_brightness = np.max([
                np.average(false_channels['red'].get_image()), 
                np.average(false_channels['blue'].get_image()), 
                np.average(false_channels['green'].get_image())])
            print(f"Max mean brightness {max_mean_brightness}")
            set_mean_brightness(max_mean_brightness, false_channels['red'], false_channels['blue'], false_channels['green'])

            # set all images to the same resolution
            pad_set(false_channels['red'], false_channels['green'], false_channels['blue'])
            # false_channels = rescale_method_1(false_channels['red'], false_channels['green'], false_channels['blue'])

            align_images(false_channels['red'], false_channels['blue'])
            align_images(false_channels['red'], false_channels['green'])

            normalize_color_channels(false_channels['red'], false_channels['green'], false_channels['blue'], min=0, max=50)
            # display image(s)
            # show_color_image(mission.get_title(), false_channels['red'], false_channels['green'], false_channels['blue'])
            show_all_channels(mission.get_title(), false_channels['red'], false_channels['green'], false_channels['blue'], min=0, max=1)
            # show_single_channel(mission.get_title(), false_channels['red'])
            return
    else:
        print("Please run one of the following:")
        print(f"`{sys.argv[0]} query` to search for project missions and print the query result to a file.")
        print(f"`{sys.argv[0]} run` to download and process mission files. Have a proposal ID ready to enter.")
        print("Exiting...")

def rescale_image(target_image: Image, target_size: float):
    # find the scale factor
    print(f"Starting size: {target_image.get_image_x()}*{target_image.get_image_y()}, pixel size: {target_image.get_pixel_side_length()}")
    scale_factor = target_size / target_image.get_pixel_side_length()
    print(f"scale factor to get to {target_size} pixel length: {scale_factor}x")

    # perform scaling operation
    rescaled = skimage.transform.rescale(target_image.get_image(), 1/scale_factor, anti_aliasing=False)
    target_image.update_image(rescaled)
    target_image.update_data(target_image.get_rotation(), (target_image.get_pixel_side_length() * scale_factor), 
        target_image.get_center_x()/scale_factor,
        target_image.get_center_y()/scale_factor)
    print(f"new listed scale: {target_image.get_image_x()}*{target_image.get_image_y()}, pixel size: {target_image.get_pixel_side_length()}")

def pad_set(red: Image, green: Image, blue: Image):
    largest = {'x': 0, 'y': 0}
    for i in [red, green, blue]:
        if i.get_image_x() > largest['x']:
            largest['x'] = i.get_image_x()
        if i.get_image_y() > largest['y']:
            largest['y'] = i.get_image_y()

    import cv2
    for i in [red, green, blue]:
        padded = cv2.copyMakeBorder(i.get_image(), 
        0, # top
        largest['y'] - i.get_image_y(), # bottom
        0, # left
        largest['x'] - i.get_image_x(), #right
        cv2.BORDER_CONSTANT, value=1000)
        i.update_image(padded)
    return

def mark_center(image: Image):
    marked = image.get_image()
    marked[int(image.get_center_x())][int(image.get_center_y())] = 0
    image.update_image(marked)

def align_images(target_image: Image, image_to_move: Image):
    difference = (int(target_image.get_center_x() - image_to_move.get_center_x()), int(target_image.get_center_y() - image_to_move.get_center_y()))
    print(f"Moving {image_to_move.get_filter()} by {difference}")
    rolled = np.roll(image_to_move.get_image(), difference[0] + 8, axis=1)
    rolled = np.roll(rolled, difference[1] - 7, axis=0)
    image_to_move.update_image(rolled)
    return

def normalize_color_channels(red: Image, blue: Image, green: Image, min=-1, max=100):
    for channel in [red, green, blue]:
        normalized = (channel.get_image() - min) / max
        channel.update_image(normalized)

def normalize_exposure(target_exposure: float, red: Image, blue: Image, green: Image):
    for channel in [red, green, blue]:
        exposure_diff = channel.get_exposure_time() - target_exposure
        print(f"Exposure difference {channel.get_filter()}: {exposure_diff}")
        # get brightness value per pixel for duration of excess exposure
        # (brightness per exposure)
        new_image = channel.get_image() - (channel.get_image() / channel.get_exposure_time()) * exposure_diff
        channel.update_image(new_image)

def set_mean_brightness(target_mean: float, red: Image, blue: Image, green: Image):
    for channel in [red, green, blue]:
        this_mean = np.average(channel.get_image())
        new_image = channel.get_image() * (target_mean / this_mean)
        channel.update_image(new_image)

def show_color_image(title, red: Image, blue: Image, green: Image):
    fig, ax = plt.subplots()
    ax.imshow(np.dstack((red.get_image(), green.get_image(), blue.get_image()))) # color
    ax.set_title(title)
    plt.show()

def show_all_channels(title, red: Image, blue: Image, green: Image, min=0, max=1, axis='on'):
    color = np.dstack((red.get_image(), green.get_image(), blue.get_image()))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
    ax = axes.ravel()

    # color
    ax[0].imshow(color)
    ax[0].axis(axis)
    ax[0].set_title(f'{title}: Color')

    # red
    ax[1].imshow(red.get_image(), cmap='gray', vmin=min, vmax=max)
    ax[1].axis(axis)
    ax[1].set_title(f'{title}: "red" {red.get_filter()}')

    # green
    ax[2].imshow(green.get_image(), cmap='gray', vmin=min, vmax=max)
    ax[2].axis(axis)
    ax[2].set_title(f'{title}: "green" {green.get_filter()}')

    # blue
    ax[3].imshow(blue.get_image(), cmap='gray', vmin=min, vmax=max)
    ax[3].axis(axis)
    ax[3].set_title(f'{title}: "blue" {blue.get_filter()}')

    plt.tight_layout()
    plt.show()

def show_single_channel(title, channel: Image, min=0, max=1):
    fig, ax = plt.subplots()
    ax.imshow(channel.get_image(), cmap='gray', vmin=min, vmax=max)
    ax.set_title(title)
    plt.show()

# S_REGION= 'POLYGON ICRS  151.721470971 -40.448319128 151.772509901 &'           CONTINUE  '-40.463765259 151.792874057 -40.424753501 151.741860051 &'           CONTINUE  '-40.409316329&'                                                      CONTINUE  ''
def parse_spatial_extent(extent_string: str) -> list[SkyCoord]:
    print(f"Input:{extent_string}")
    new_string = extent_string.replace('CONTINUE', '')
    new_string = new_string.replace('POLYGON ICRS', '')
    string_array = new_string.split(' ')
    string_array = list(filter(lambda coord: coord != '', string_array))
    print(f"Output:{string_array}")
    spatial_extent = []

    if len(string_array) % 2 == 0:
        while len(string_array) > 0:
            ra = float(string_array.pop(0))
            dec = float(string_array.pop(0))
            new = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
            spatial_extent.append(new)
            print(new)
    return spatial_extent

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