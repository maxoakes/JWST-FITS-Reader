class Card:
    keyword = ""
    value = ""
    comment = ""

    def __init__(self, k, v, c):
        self.keyword = str(k).strip()
        self.value = str(v).strip()
        self.comment = str(c).strip()

    def parse_header_to_list(header) -> list:
        all_cards = []
        for card in header:
            if (card.keyword):
                c = Card(card.keyword, card.value, card.comment)
                all_cards.append(c)
        return all_cards
    
    def parse_header_to_dict(header) -> dict:
        all_cards = {}
        for card in header:
            if (card.keyword):
                all_cards[card.keyword] = card.value
        return all_cards

    def __str__(self):
        return f'{self.keyword}: {self.value} # {self.comment}'