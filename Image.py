import math
import numpy as np
import skimage
import skimage.transform
from astropy import units as u
from astropy.coordinates import SkyCoord
from Card import Card

class Image:
    __filter: str
    __data_type: str
    __image_data: np.ndarray #images as np arrays
    __coords: SkyCoord
    __rotation: float # deg
    __arcsec_per_pixel: float # Nominal pixel area in arcsec^2
    __centerX: float
    __centerY: float

    def __init__(self, filter, data_type, image_data, rotation, app, spp, ra, dec, x, y):
        self.__filter = filter
        self.__data_type = data_type
        self.__image_data = np.array(image_data, dtype=float)
        self.__rotation = rotation
        self.__arcsec_per_pixel = app
        self.__coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        self.__centerX = x
        self.__centerY = y

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
    
    def get_arcsec_per_pixel(self):
        return self.__arcsec_per_pixel
    
    def update_data(self, rotation, app, centerX, centerY):
        self.__rotation = rotation
        self.__arcsec_per_pixel = app
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
        return f"Data for {self.__data_type} filtered with {self.__filter}. Pixel side len: {self.__arcsec_per_pixel}. Size: {self.get_image_x()}*{self.get_image_y()}, angle: {self.__rotation}deg, {self.__coords}"