import numpy as np
from Card import Card

class Image:
    filter = '__unset__'
    data_type = '__unset__'
    header = [] # Array of Cards
    image_data = [[]] #images as np arrays
    parent_meta = False
    x = 0 # int
    y = 0 # int

    def __init__(self, filter, data_type, image_data, header, meta):
        self.filter = filter
        self.data_type = data_type
        self.image_data = image_data
        self.y, self.x = image_data.shape
        for el in header:
            self.header.append(Card(el.keyword, el.value, el.comment))
        self.parent_meta = meta

    def __str__(self):
        return f"Data for {self.data_type} filtered with {self.filter}: [{self.x}, {self.y}]. Header with {len(self.header)} elements."