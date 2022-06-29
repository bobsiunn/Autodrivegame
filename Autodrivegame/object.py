class detectedObject():
    def __init__(self, type, typeid, pos, score, mask):
        self.type = type
        self.typeid = typeid
        self.pos = pos
        self.score = score
        self.mask = mask
        self.id = 0

        self.track_on_screen = False
        self.track_count = 0
        
    def setId(self, id):
        self.id = id

    def setObject(self, _object):
        self.pos = _object.pos
        self.mask = _object.mask
        self.score = _object.score

