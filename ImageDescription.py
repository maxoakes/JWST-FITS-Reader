from astropy.io import fits
from Card import Card
import numpy as np

class ImageDescription:
    # dataproduct_type
    # calib_level
    # obs_collection
    # obs_id
    # target_name
    # s_ra
    # s_dec
    # t_min
    # t_max
    # t_exptime
    # wavelength_region
    # filters
    # em_min
    # em_max
    # target_classification
    # obs_title
    # t_obs_release
    # instrument_name
    # proposal_pi
    # proposal_id
    # proposal_type
    # project
    # sequence_number
    # provenance_name
    # s_region
    # jpegURL
    # dataURL
    # dataRights
    # mtFlag
    # srcDen
    # intentType
    # obsid
    # objID
    __row = {}
    __fits_path = ""
    __preview_path = ""

    def __init__(self, r):
        self.__row = r

    # Setters
    def set_files(self, data, preview):
        self.__fits_path = data
        self.__preview_path = preview
    
    # Getters
    def get_filter_name(self) -> str:
        return self.__row['filters']

    def get_fits_path(self) -> str:
        return self.__fits_path
    
    def get_image(self, fits_index: int | str):
        hdul = fits.open(self.__fits_path)
        image = hdul[fits_index].data
        hdul.close()
        return np.array(image)
    
    def get_card(self, fits_index: int | str, card: str):
        hdul = fits.open(self.__fits_path)
        card_data = hdul[fits_index].header[card]
        hdul.close()
        return card_data
    
    def get_metadata(self, index: int | str) -> str:
        return self.__row[index]
    
    # Writing
    def write_fits_header_to_file(self, fits_index: int | str):
        hdul = fits.open(self.__fits_path)
        file = open(f'{str(self.__fits_path)}.{fits_index}.txt', 'w')
        for c in hdul[fits_index].header.cards:
            file.write(f'{str(c)}\n')
        file.close()
        hdul.close()

    def __str__(self):
        return self.__row['obs_id']
