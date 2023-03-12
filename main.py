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

#   longest wave length
# red (1,0,0)
# orange (0.666, 0.333, 0)
# yellow (0.5, 0.5, 0)
# lime (0.333, 0.666, 0)
# green (0, 1, 0)
# seafoam (0, 0.666, 0.333)
# cyan (0, 0.5, 0.5)
# skyblue (0, 0.333, 0.666)
# blue (0, 0, 1)
# purple (0.333, 0, 0.666)
# magenta (0.5, 0, 0.5)
# violet (0.666, 0, 0.333)
#   shortest wave length
# selection max = 12

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
            downloaded_items, mission_path = download_mission(id)
            mission = Mission(downloaded_items[0].get_metadata('target_name'), mission_path)
            print(f"path:{mission.get_mission_path()}")
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
            filter_images: dict[str, RawImage] = { }
            smallest_shape = [20000, 20000]
            smallest_image_key = ''
            for description in mission.search(sortby='FILTER'):
                this_filter = f"{description.get_card(0, 'FILTER')}/{description.get_card(0,'PUPIL')}"
                print(f"Creating image for {this_filter}")
                filter_images[this_filter] = RawImage(description.get_card(0, 'FILTER'), description.get_card(0, 'PUPIL'), 'SCI', description)

                # get the smallest image resolution
                x = filter_images[this_filter].get_image().shape[0]
                y = filter_images[this_filter].get_image().shape[1]
                # print(f"[{x}, {y}] < {smallest_shape}")
                if (x < smallest_shape[0]) and (y < smallest_shape[1]):
                    smallest_shape = list(filter_images[this_filter].get_image().shape)
                    smallest_image_key = this_filter
            print(f"Smallest resolution image is {smallest_shape}: {smallest_image_key}")

            # brightness clamping
            decimals = 2
            for filter, image in filter_images.items():
                min = np.round(np.min(image.get_image()), decimals)
                max = np.round(np.max(image.get_image()), decimals)
                avg = np.round(np.average(image.get_image()), decimals)
                median = np.round(np.median(image.get_image()), decimals)
                upper_quartile = np.round(np.quantile(image.get_image(), 0.999))
                print(f"{filter}: Min: {min}, Max: {max}, Avg: {avg}, Median: {median}, Quartile:{upper_quartile}")
                clipped = np.clip(image.get_image(), 0, upper_quartile)
                clipped = clipped / upper_quartile
                print(f"{filter} clipped to [{np.min(clipped)}, {np.max(clipped)}]")
                image.update_image(clipped)

            # do all of my own alignment attempt steps
            # custom_alignment(filter_images)

            # alignment via astroalign
            aligned_filter_channels: dict[str, np.ndarray] = { }
            if os.path.exists(os.path.join(os.getcwd(), mission_path, "aligned")):
                # for each image in the aligned folder, just load that rather than make an alignment
                print(f"Alignment folder already exists. Assuming everything is valid. Skipping alignment.")
                for image_file in os.listdir(os.path.join(os.getcwd(), mission_path, "aligned")):
                    filter_name = image_file.replace('-','/').replace('.png','')
                    print(f"Loading {image_file} to {filter_name}...")
                    aligned_filter_channels[filter_name] = imread(os.path.join(os.getcwd(), mission_path, "aligned", image_file))
            else:
                import astroalign
                default_max = 100
                point_num = input(f"Number of control points for alignment (more=more time, more likely to get triangulation. Default={default_max}): ")
                if not point_num:
                    point_num = default_max
                print(f"Using max control points={point_num}")
                smallest_image_key = 'F090W/CLEAR'
                # if the aligned folder does not exist, we have no alignments done, so generate them
                for filter, image in filter_images.items():
                    file_name = f"{filter.replace('/','-')}.png"
                    sigma = 10
                    if (filter != smallest_image_key):
                        while(True):
                            print(f"Attempting to align {filter} with detection sigma {sigma}")
                            try:
                                new_image, mask = astroalign.register(filter_images[filter].get_image(), filter_images[smallest_image_key].get_image(), max_control_points=point_num, detection_sigma=sigma)
                                aligned_filter_channels[filter] = new_image
                                print(f"Alignment found for {filter}")
                                break
                            except astroalign.MaxIterError as exc:
                                print(exc)
                                print(f"Triangulation failure. Error given: {exc}")
                                break
                            except TypeError as exc:
                                print(f"Failed to find alignment, iterating detection sigma. Error given: {exc}")
                                sigma = sigma + 2
                    else:
                        print(f"Skipping {filter} since it is the base alignment")
                        aligned_filter_channels[filter] = filter_images[smallest_image_key].get_image()

                # for each generated alignment, save them to a file
                for filter, image in aligned_filter_channels.items():
                    print(f"Writing {filter} to disk...")
                    if not os.path.exists(os.path.join(os.getcwd(), mission_path, "aligned")):
                        os.makedirs(os.path.join(os.getcwd(), mission_path, "aligned"))
                    uint8_version = (aligned_filter_channels[filter] * 255).astype('uint8')
                    imsave(f"{mission_path}\\aligned\\{filter.replace('/','-')}.png", uint8_version)

            # create false color channels for each aligned image
            false_channel_bank = [
                ("red", (1, 0, 0)),
                ("orange", (1, 0.5, 0)),
                ("yellow", (1, 1, 0)),
                ("lime", (0.5, 1, 0)),
                ("green", (0, 1, 0)),
                ("seafoam", (0, 1, 0.5)),
                ("cyan", (0, 1, 1)),
                ("skyblue", (0, 0.5, 1)),
                ("blue", (0, 0, 1)),
                ("purple", (0.5, 0, 1)),
                ("magenta", (1, 0, 1)),
                ("violet", (1, 0, 0.5))
            ]

            # come up with false-color assignment for each filter
            keys = list(aligned_filter_channels.keys())
            print(keys)
            num_channels = len(aligned_filter_channels)
            max_channels = len(false_channel_bank)
            false_color_channels = { }
            final_shape = (0,0,0)
            mode = input("Choose one: (linspace|evenspace|first|last|random). Default is linspace: ")
            if not mode:
                mode = "linspace"
            if mode == "linspace":
                assigned_indicies = np.floor(np.linspace(0, max_channels-1, num=num_channels)).astype('int')
                assignments = zip(range(0, num_channels), assigned_indicies)
                print(list(assignments))
                for i, index in enumerate(assigned_indicies):
                    (mult_red, mult_green, mult_blue) = false_channel_bank[index][1]
                    false_color_name = false_channel_bank[index][0]
                    false_color_channels[false_color_name] = np.dstack((
                        aligned_filter_channels[keys[i]] * mult_red,
                        aligned_filter_channels[keys[i]] * mult_green,
                        aligned_filter_channels[keys[i]] * mult_blue
                    ))
                    final_shape = false_color_channels[false_color_name].shape

                    print(f"Created {false_color_name} image")
            else:
                pass

            # write the false-channel images to files
            channel_directory = "channels"
            if not os.path.exists(os.path.join(os.getcwd(), mission_path, channel_directory)):
                os.makedirs(os.path.join(os.getcwd(), mission_path, channel_directory))
                for color, image in false_color_channels.items():
                    print(f"Writing {color} to disk...")
                    uint8_version = (false_color_channels[color] * 255).astype('uint8')
                    imsave(f"{mission_path}\\{channel_directory}\\{color}.png", uint8_version)

            # add the multi-false-colors into a regular RGB image
            false_color_image = np.zeros(final_shape, dtype = float)

            print(f"Assigning {num_channels} to at most {max_channels} channels")
            # https://processing.org/reference/blend_.html
            from PIL import Image
            for color, image in false_color_channels.items():
                factor = 1.0
                false_color_image = Image.blend(Image.fromarray(np.uint8(false_color_image)), Image.fromarray(np.uint8(image)), 0.75) 
                fig, ax = plt.subplots()
                ax.imshow(false_color_image)
                ax.set_title(color)
                plt.show()
            
            # cast to uint8
            print(np.max(false_color_image))
            final_image = (false_color_image * (255/np.max(false_color_image))).astype('uint8')

            fig, ax = plt.subplots()
            ax.imshow(final_image, vmin=0, vmax=255)
            ax.set_title("Final Color Image")
            plt.show()
            return
    else:
        print("Please run one of the following:")
        print(f"`{sys.argv[0]} query` to search for project missions and print the query result to a file.")
        print(f"`{sys.argv[0]} run` to download and process mission files. Have a proposal ID ready to enter.")
        print("Exiting...")

def custom_alignment(image_dict: dict[str, RawImage]):
    # rescale images so each pixel measures the same area in space
    for color, image in image_dict.items():
        rescale_image(image, 0.07)
        print()
        print(image)
    
    for color, image in image_dict.items():
        mark_center(image)

    # optional: exposure time equalization, brightness equalization
    # needed: projection extent correction to align all channels

    # min_exposure_time = np.min([image_dict['red'].get_exposure_time(), image_dict['blue'].get_exposure_time(), image_dict['green'].get_exposure_time()])
    # print(f"Min Exposure Time is {min_exposure_time} sec")
    # # normalize_exposure(min_exposure_time, image_dict['red'], image_dict['blue'], image_dict['green'])

    # max_mean_brightness = np.max([
    #     np.average(image_dict['red'].get_image()), 
    #     np.average(image_dict['blue'].get_image()), 
    #     np.average(image_dict['green'].get_image())])
    # print(f"Max mean brightness {max_mean_brightness}")
    # set_mean_brightness(max_mean_brightness, image_dict['red'], image_dict['blue'], image_dict['green'])

    # set all images to the same resolution
    pad_set(image_dict['red'], image_dict['green'], image_dict['blue'])
    # image_dict = rescale_method_1(image_dict['red'], image_dict['green'], image_dict['blue'])

    align_images(image_dict['red'], image_dict['blue'])
    align_images(image_dict['red'], image_dict['green'])

def rescale_image(target_image: RawImage, target_size: float):
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

def pad_set(red: RawImage, green: RawImage, blue: RawImage):
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

def mark_center(image: RawImage):
    marked = image.get_image()
    marked[int(image.get_center_x())][int(image.get_center_y())] = 0
    image.update_image(marked)

def align_images(target_image: RawImage, image_to_move: RawImage):
    difference = (int(target_image.get_center_x() - image_to_move.get_center_x()), int(target_image.get_center_y() - image_to_move.get_center_y()))
    print(f"Moving {image_to_move.get_filter()} by {difference}")
    rolled = np.roll(image_to_move.get_image(), difference[0] + 8, axis=1)
    rolled = np.roll(rolled, difference[1] - 7, axis=0)
    image_to_move.update_image(rolled)
    return

def normalize_color_channels(red: RawImage, blue: RawImage, green: RawImage, min=-1, max=100):
    for channel in [red, green, blue]:
        normalized = (channel.get_image() - min) / max
        channel.update_image(normalized)

def normalize_exposure(target_exposure: float, red: RawImage, blue: RawImage, green: RawImage):
    for channel in [red, green, blue]:
        exposure_diff = channel.get_exposure_time() - target_exposure
        print(f"Exposure difference {channel.get_filter()}: {exposure_diff}")
        # get brightness value per pixel for duration of excess exposure
        # (brightness per exposure)
        new_image = channel.get_image() - (channel.get_image() / channel.get_exposure_time()) * exposure_diff
        channel.update_image(new_image)

def set_mean_brightness(target_mean: float, red: RawImage, blue: RawImage, green: RawImage):
    for channel in [red, green, blue]:
        this_mean = np.average(channel.get_image())
        new_image = channel.get_image() * (target_mean / this_mean)
        channel.update_image(new_image)

def show_color_image(title, color_image):
    fig, ax = plt.subplots()
    ax.imshow(color_image) # color
    ax.set_title(title)
    plt.show()

def show_all_channels(title, red: RawImage, blue: RawImage, green: RawImage, min=0, max=1, axis='on'):
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

def show_single_image(image: RawImage, axis='on', min=0, max=100):
    fig, ax = plt.subplots()
    ax.imshow(image.get_image(), cmap='gray', vmin=min, vmax=max)
    ax.axis(axis)
    ax.set_title(f'{image.get_filter()}')
    plt.show()

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

if __name__ == '__main__':
    main()