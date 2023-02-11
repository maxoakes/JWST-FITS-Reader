class Asset:
    metadata = False
    science_image = []
    
    data = False

    def __init__(self, metadata):
        self.metadata = metadata

    def __str__(self):
        return f"Asset for {self.metadata} with data {self.data}"
    


