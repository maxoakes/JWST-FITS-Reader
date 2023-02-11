from Image import Image

class Mission:
    title = ""
    images = []

    def __init__(self, title):
        self.title = title

    def add_image(self, image):
        self.images.append(image)

    def search(self, filter='', data_type=''):
        found = []
        if filter == '':
            found = self.images
        else:
            for im in self.images:
                if im.filter == filter:
                    found.append(im)
        if data_type == '':
            return found
        else:
            returning = []
            for f in found:
                if f.data_type == data_type:
                    returning.append(f)
        return returning

    def __str__(self):
        return f"Mission: {self.title} with {len(self.images)} images"
    


