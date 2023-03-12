import math
import numpy as np
import skimage
import skimage.transform
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from Card import Card
from ImageDescription import ImageDescription

class RawImage:
    __filter: str
    __pupil: str
    __data_type: str
    __image_data: np.ndarray #images as np arrays
    __coords: SkyCoord
    __rotation: float # deg
    __area_per_pixel: float # Nominal pixel area in arcsec^2
    __centerX: float
    __centerY: float
    __exposure: float # seconds
    __extent: list[SkyCoord]
    __matrix_transform: list[list[float]]

    def __init__(self, filter: str, pupil: str, index: int | str, description: ImageDescription):
        header = description.get_header(index)
        self.__image_data =  np.array(description.get_image(index), dtype=float)
        self.__filter = filter
        self.__pupil = pupil
        self.__coords = SkyCoord(ra=header['RA_V1']*u.degree, dec=header['DEC_V1']*u.degree, frame='icrs')
        self.__rotation = header['PA_V3']
        self.__area_per_pixel = header['PIXAR_A2']
        self.__centerX = header['CRPIX1']
        self.__centerY = header['CRPIX2']
        self.exposure = header['XPOSURE'], 
        self.extent = RawImage.parse_spatial_extent(header['S_REGION']),
        self.__matrix_transform = [[header['PC1_1'],header['PC1_2']],[header['PC2_1'],header['PC2_2']]]
        self.__data_type = index

    # S_REGION= 'POLYGON ICRS  151.721470971 -40.448319128 151.772509901 &'           CONTINUE  '-40.463765259 151.792874057 -40.424753501 151.741860051 &'           CONTINUE  '-40.409316329&'                                                      CONTINUE  ''
    def parse_spatial_extent(extent_string: str) -> list[SkyCoord]:
        # print(f"Input:{extent_string}")
        new_string = extent_string.replace('CONTINUE', '')
        new_string = new_string.replace('POLYGON ICRS', '')
        string_array = new_string.split(' ')
        string_array = list(filter(lambda coord: coord != '', string_array))
        # print(f"Output:{string_array}")
        spatial_extent = []

        if len(string_array) % 2 == 0:
            while len(string_array) > 0:
                ra = float(string_array.pop(0))
                dec = float(string_array.pop(0))
                new = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                spatial_extent.append(new)
                # print(new)
        return spatial_extent

    def get_filter(self):
        return self.__filter
    
    def get_image_type(self):
        return self.__data_type

    def get_image(self):
        return self.__image_data
        
    def get_upright_image(self) -> np.ndarray:
        print(f"Input shape {self.__image_data.shape}")
        upright = skimage.transform.rotate(image=self.__image_data, angle=-1*self.__rotation, resize=True, center=(self.__centerX, self.__centerY))
        print(f"Output shape {upright.shape}")
        return upright
    
    def get_image_x(self):
        return self.__image_data.shape[1]
    
    def get_image_y(self):
        return self.__image_data.shape[0]
    
    def get_rotation(self):
        return self.__rotation
    
    def get_area_per_pixel(self):
        return self.__area_per_pixel
    
    def get_pixel_side_length(self):
        return math.sqrt(self.__area_per_pixel)
    
    def get_extent(self, index: int) -> SkyCoord:
        return self.__extent[index]
    
    def get_exposure_time(self) -> float:
        return self.__exposure
    
    def get_matrix(self) -> list[list[float]]:
        return self.__matrix_transform
    
    def update_data(self, rotation, app, centerX, centerY):
        self.__rotation = rotation
        self.__area_per_pixel = app
        self.__centerX = centerX
        self.__centerY = centerY
        return

    def get_center_x(self):
        return self.__centerX
    
    def get_center_y(self):
        return self.__centerY
    
    def update_image(self, new_image):
        self.__image_data = new_image
        return

    def __str__(self):
        return f"Data for {self.__data_type} filtered with {self.__filter}. Pixel side len: {self.__area_per_pixel}. Size: {self.get_image_x()}*{self.get_image_y()}, angle: {self.__rotation}deg, {self.__coords}"