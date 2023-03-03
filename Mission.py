from ImageDescription import ImageDescription

class Mission:
    __title = ""
    __items = []
    __mission_path = ''

    def __init__(self, title, path):
        self.__title = title
        self.__mission_path = path

    def add_item(self, item):
        self.__items.append(item)

    def search(self, filter='', sortby='DATE-OBS') -> list[ImageDescription]:
        found = []
        if filter == '':
            found = self.__items
        else:
            for item in self.__items:
                if item.get_filter_name() == filter:
                    found.append(item)
        return sorted(found, key=lambda im: im.get_card(0, sortby))

    def get_title(self) -> str:
        return self.__title
    
    def get_mission_path(self) -> str:
        return self.__mission_path
    
    def __str__(self):
        return f"Mission: {self.__title} with {len(self.__items)} items"