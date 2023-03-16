import sys
import os
import datetime
import numpy as np
import skimage
import skimage.transform
from skimage import io
from matplotlib import pyplot as plt
from astroquery.mast import Observations
from Mission import Mission
from RawImage import RawImage
from Query import Query
from PIL import Image

def main():
    # bad program start
    if len(sys.argv) == 1:
        print_help()
        return
    
    # run query to download csv
    if (sys.argv[1] == 'query'):
        search_params = {}
        for p in [['target_name', 'Name of the object being imaged: '], 
                    ['obs_title', 'Formal title of Project: '],
                    ['proposal_id', 'Proposal ID: ']]:
            inp = input(p[1])
            if (inp):
                search_params[p[0]] = [inp]
        Query.run_query(search_params, True)
        return
    
    # run the program, then give a proposal ID
    if (sys.argv[1] == 'run'):
        default_id = 2733 # NGC 3132, small nebulae
        id = input("Proposal ID: ")
        if not id:
            id = default_id

        # Acquire imaging data
        downloaded_items, mission_path = Query.download_mission(id)
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
            lower_lim = 0.0
            clipped = np.clip(image.get_image(), lower_lim, upper_quartile + lower_lim) - lower_lim
            # show_single_filter(filter, clipped)
            clipped = clipped / upper_quartile
            print(f"{filter} clipped to [{np.min(clipped)}, {np.max(clipped)}]")
            image.update_image(clipped)

        # do all of my own alignment attempt steps
        aligned_filter_channels: dict[str, np.ndarray] = { }
        if os.path.exists(os.path.join(os.getcwd(), mission_path, "aligned")):
            # for each image in the aligned folder, just load that rather than make an alignment
            print(f"Alignment folder already exists. Assuming everything is valid. Skipping alignment.")
            for image_file in os.listdir(os.path.join(os.getcwd(), mission_path, "aligned")):
                filter_name = image_file.replace('-','/').replace('.png','').replace('.tif','')
                print(f"Loading {image_file} to {filter_name}...")
                # aligned_filter_channels[filter_name] = io.imread(os.path.join(os.getcwd(), mission_path, "aligned", image_file)) / 255
                aligned_filter_channels[filter_name] = io.imread(os.path.join(os.getcwd(), mission_path, "aligned", image_file))
                aligned_filter_channels[filter_name] = load_image((os.path.join(os.getcwd(), mission_path, "aligned", image_file)), 'float32')
        else:
            # CUSTOM ALIGNMENT
            alignment_type = input("Alignment type (astroalign|custom)? (Default = astroalign): ")
            if not alignment_type:
                alignment_type = "astroalign"
            if alignment_type == "custom":
                print("Performing custom naive alignment")
                # rescale images so each pixel measures the same area in space
                for color, image in filter_images.items():
                    rescale_image(image, 0.07)
                    mark_center(image)

                # set all images to the same resolution and pad them
                largest = {'x': 0, 'y': 0}
                for filter, image in filter_images.items():
                    if image.get_image_x() > largest['x']:
                        largest['x'] = image.get_image_x()
                    if image.get_image_y() > largest['y']:
                        largest['y'] = image.get_image_y()

                import cv2
                for filter, image in filter_images.items():
                    padded = cv2.copyMakeBorder(image.get_image(), 
                    0, # top
                    largest['y'] - image.get_image_y(), # bottom
                    0, # left
                    largest['x'] - image.get_image_x(), #right
                    cv2.BORDER_CONSTANT, value=1.0)
                    image.update_image(padded)

                # align the images so they all have the same 'center'
                keys = list(filter_images.keys())
                target_image = filter_images[keys[0]]
                for filter, image in filter_images.items():
                    difference = (int(target_image.get_center_x() - image.get_center_x()), int(target_image.get_center_y() - image.get_center_y()))
                    print(f"Moving {filter} by {difference}")
                    rolled = np.roll(image.get_image(), difference[0], axis=1)
                    rolled = np.roll(rolled, difference[1], axis=0)
                    image.update_image(rolled)

                # move the images to the aligned channel dict
                for filter, image in filter_images.items():
                    aligned_filter_channels[filter] = image.get_image()

            # ASTRO ALIGN AUTO
            else:
                # alignment via astroalign
                print("Performing astroalign (automatic)")
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
                                sigma = sigma + 4
                    else:
                        print(f"Skipping {filter} since it is the base alignment")
                        aligned_filter_channels[filter] = filter_images[smallest_image_key].get_image()

            # for each generated alignment, save them to a file
            for filter, image in aligned_filter_channels.items():
                print(f"Writing {filter} to disk...")
                if not os.path.exists(os.path.join(os.getcwd(), mission_path, "aligned")):
                    os.makedirs(os.path.join(os.getcwd(), mission_path, "aligned"))
                uint8_version = (aligned_filter_channels[filter] * 255).astype('uint8')
                float32_version = np.array(aligned_filter_channels[filter], dtype='float32')
                save_image(float32_version, f"{mission_path}\\aligned", filter.replace('/','-'), 'float32')

        # print checkin
        for filter, image in aligned_filter_channels.items():
            print(f"{filter} range is [{np.min(image)}, {np.max(image)}]")
        
        # come up with false-color images for each mono-color filter
        default = "nasa"
        false_color_images = { }
        mode = input(f"Choose one: (basic|nasa). Default is {default}: ")
        if not mode:
            mode = default
        if mode == "basic":
            false_color_images = assign_basic_colors(aligned_filter_channels)
        elif mode == "nasa":
            false_color_images = assign_nasa_colors(aligned_filter_channels)
        else:
            pass

        # write the false-channel images to files
        channel_directory = "channels"
        if not os.path.exists(os.path.join(os.getcwd(), mission_path, channel_directory)):
            os.makedirs(os.path.join(os.getcwd(), mission_path, channel_directory))
            for color, image in false_color_images.items():
                print(f"Writing {color} to disk...")
                uint8_version = ((false_color_images[color] * 255)).astype('uint8')
                save_image(uint8_version, f"{mission_path}\\{channel_directory}", color.replace('/','-'), 'uint8')

        # initialize decomposed final image
        final_shape = aligned_filter_channels[list(aligned_filter_channels.keys())[0]].shape
        final_image_channels = {
            "red": np.zeros(final_shape),
            "green": np.zeros(final_shape),
            "blue": np.zeros(final_shape),
        }

        # order from blue to red
        keys = list(false_color_images.keys())
        keys.sort()
        keys = keys[:-1]
        print(f"Order: {keys}")
        max_color_value = 0.0
        for color in keys:
            red = get_channel(false_color_images[color], 'red')
            green = get_channel(false_color_images[color], 'green')
            blue = get_channel(false_color_images[color], 'blue')
            # https://processing.org/reference/blend_.html
            for name, data in (('red', red), ('green', green), ('blue', blue)):
                # final_image_channels[name] = (data * 1) + final_image_channels[name] # blend
                # final_image_channels[name] = np.minimum((data) + final_image_channels[name], 1.0) # add
                final_image_channels[name] = np.maximum(data*1, final_image_channels[name]) # lighten
                if np.max(final_image_channels[name]) > max_color_value:
                    max_color_value = np.max(final_image_channels[name])
    
        # # normalize
        # print(f"Found max color value of {max_color_value}")
        # for color, image in final_image_channels.items():
        #     final_image_channels[color] = final_image_channels[color] / max_color_value

        # output image
        final_image = show_color_image_from_dict(final_image_channels)
        timestamp_string = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        io.imsave(f"{mission_path}\\output\\{timestamp_string}.png", (np.clip(final_image, 0, 1.0) * 255).astype('uint8'))
        print("Write complete!")
        return
        # end
    print_help()

def print_help():
    print("Please run one of the following:")
    print(f"`{sys.argv[0]} query` to search for project missions and print the query result to a file.")
    print(f"`{sys.argv[0]} run` to download and process mission files. Have a proposal ID ready to enter.")
    print("Exiting...")

def rescale_image(target_image: RawImage, target_size: float) -> None:
    # find the scale factor
    print(f"{target_image.get_filter()}: Starting size: {target_image.get_image_x()}*{target_image.get_image_y()} with pixel size: {str(np.round(target_image.get_pixel_side_length(), 5))}")
    scale_factor = target_size / target_image.get_pixel_side_length()
    print(f"{target_image.get_filter()}: Scale factor to get to {target_size} pixel length: {scale_factor}x")

    # perform scaling operation
    rescaled = skimage.transform.rescale(target_image.get_image(), 1/scale_factor, anti_aliasing=False)
    target_image.update_image(rescaled)
    target_image.update_data(target_image.get_rotation(), (target_image.get_pixel_side_length() * scale_factor), 
        target_image.get_center_x()/scale_factor,
        target_image.get_center_y()/scale_factor)
    print(f"{target_image.get_filter()}: New listed scale: {target_image.get_image_x()}*{target_image.get_image_y()} with pixel size: {str(np.round(target_image.get_pixel_side_length(), 5))}")

def mark_center(image: RawImage):
    marked = image.get_image()
    marked[int(image.get_center_x())][int(image.get_center_y())] = 0
    image.update_image(marked)

def normalize_exposure(target_exposure: float, red: RawImage, blue: RawImage, green: RawImage):
    for channel in [red, green, blue]:
        exposure_diff = channel.get_exposure_time() - target_exposure
        print(f"Exposure difference {channel.get_filter()}: {exposure_diff}")
        # get brightness value per pixel for duration of excess exposure
        # (brightness per exposure)
        new_image = channel.get_image() - (channel.get_image() / channel.get_exposure_time()) * exposure_diff
        channel.update_image(new_image)

# my original method
def assign_basic_colors(aligned_filter_channels):
    # create false color channels for each aligned image
    basic_colors = [
        ("violet", (1, 0, 0.5)),
        ("magenta", (1, 0, 1)),
        ("purple", (0.5, 0, 1)),
        ("blue", (0, 0, 1)),
        ("skyblue", (0, 0.5, 1)),
        ("cyan", (0, 1, 1)),
        ("seafoam", (0, 1, 0.5)),
        ("green", (0, 1, 0)),
        ("lime", (0.5, 1, 0)),
        ("yellow", (1, 1, 0)),
        ("orange", (1, 0.5, 0)),
        ("red", (1, 0, 0))
    ]

    false_color_images = { }
    keys = list(aligned_filter_channels.keys())
    print(keys)
    num_channels = len(aligned_filter_channels)
    max_channels = len(basic_colors)
    total_channel_additions = np.array([0, 0, 0], float)
    assigned_indicies = np.floor(np.linspace(0, max_channels-1, num=num_channels)).astype('int')
    assignments = zip(range(0, num_channels), assigned_indicies)
    print(list(assignments))
    for i, index in enumerate(assigned_indicies):
        (mult_red, mult_green, mult_blue) = basic_colors[index][1]
        total_channel_additions = np.array([mult_red, mult_green, mult_blue], float) + total_channel_additions
        this_color_image = np.dstack((
            aligned_filter_channels[keys[i]] * mult_red,
            aligned_filter_channels[keys[i]] * mult_green,
            aligned_filter_channels[keys[i]] * mult_blue
        ))
        print(f"Created {basic_colors[index][0]} color image")
        false_color_images[basic_colors[index][0]] = this_color_image
        # show_decomposed_channels("image", this_color_image)
    return false_color_images

def assign_nasa_colors(aligned_filter_channels):
    # derived from https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters
    nasa_colors_dict = {
        "F150W2/CLEAR": (127/255,205/255,255/255),
        "F070W/CLEAR": (127/255,127/255,216/255),
        "F090W/CLEAR": (127/255,127/255,240/255),
        "F115W/CLEAR": (127/255,135/255,255/255),
        "F150W/CLEAR": (127/255,175/255,255/255),
        "F200W/CLEAR": (127/255,231/255,255/255),
        "F140M/CLEAR": (127/255,165/255,255/255),
        "F150W2/F162M": (127/255,191/255,255),
        "F182M/CLEAR": (127/255,215/255,255/255),
        "F210M/CLEAR": (128/255,243/255,249/255),
        "F150W2/F164N": (127/255,193/255,255/255),
        "F187N/CLEAR": (127/255,219/255,255/255),
        "F212N/CLEAR": (131/255,247/255,246/255),
        "F322W2/CLEAR": (255/255,242/255,127/255),
        "F277W/CLEAR": (191/255,255/255,186/255),
        "F356W/CLEAR": (255/255,234/255,127/255),
        "F444W/CLEAR": (255/255,147/255,127/255),
        "F250M/CLEAR": (165/255,255/255,212/255),
        "F300M/CLEAR": (210/255,255/255,167/255),
        "F335M/CLEAR": (244/255,255/255,133/255),
        "F360M/CLEAR": (244/255,255/255,133/255),
        "F410M/CLEAR": (255/255,181/255,127/255),
        "F430M/CLEAR": (255/255,158/255,127/255),
        "F460M/CLEAR": (238/255,127/255,127/255),
        "F480M/CLEAR": (216/255,127/255,127/255),
        "F356W/F323N": (233/255,255/255,144/255),
        "F444W/F405N": (255/255,182/255,127/255),
        "F444W/F466N": (234/255,127/255,127/255),
        "F444W/F470N": (227/255,127/255,127/255)
    }
    false_color_images = { }

    length = len(aligned_filter_channels)
    for filter, image in aligned_filter_channels.items():
        (red_mult, green_mult, blue_mult) = nasa_colors_dict[filter]
        this_color_image = np.dstack((
            (image * red_mult),
            (image * green_mult),
            (image * blue_mult)
        ))
        false_color_images[filter] = this_color_image
        print(f"Created {filter} color image")
        # show_decomposed_channels("image", this_color_image)
    return false_color_images

def get_channel(image, channel: str):
    if channel == 'red':
        return image[ :, :, 0]
    if channel == 'green':
        return image[ :, :, 1]
    if channel == 'blue':
        return image[ :, :, 2]
    return np.zeros((image.shape[0], image.shape[1]))

def show_decomposed_channels(title, image: np.ndarray, min=0, max=1, axis='on'):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
    ax = axes.ravel()

    # color
    ax[0].imshow(image, vmin=min, vmax=max)
    ax[0].axis(axis)
    ax[0].set_title(f'{title}: Color')

    # red
    ax[1].imshow(get_channel(image, 'red'), cmap='gray', vmin=min, vmax=max)
    ax[1].axis(axis)
    ax[1].set_title(f'{title}: "red"')

    # green
    ax[2].imshow(get_channel(image, 'green'), cmap='gray', vmin=min, vmax=max)
    ax[2].axis(axis)
    ax[2].set_title(f'{title}: "green"')

    # blue
    ax[3].imshow(get_channel(image, 'blue'), cmap='gray', vmin=min, vmax=max)
    ax[3].axis(axis)
    ax[3].set_title(f'{title}: "blue"')

    plt.tight_layout()
    plt.show()

def show_color_image_from_dict(dict, axis='on', min=0, max=1):
    fig, ax = plt.subplots()
    color_image = np.dstack((dict['red'], dict['green'], dict['green']))
    ax.imshow(color_image)
    ax.axis(axis)
    ax.set_title(f'Composed Color Image')
    plt.show()
    return color_image

def show_single_filter(title, image, axis='on'):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis(axis)
    ax.set_title(title)
    plt.show()

def show_histogram(image):
    plt.hist(image.flatten())
    plt.xlim([0,12000])
    plt.show()

def save_image(data, path, name, dtype):
    if dtype=='float32':
        Image.fromarray(data).save(f'{path}\\{name}.tif')
        print(f"Wrote {name} to disk as {dtype}")
    else:
        print(f"Writing {name} to disk as color image(?)")
        io.imsave(f"{path}\\{name}.png", data)
        print(f"Wrote {name} to disk")

def load_image(name, dtype):
    if dtype=='float32':
        return np.array(Image.open(name))
    else:
        return io.imread(name)

if __name__ == '__main__':
    main()