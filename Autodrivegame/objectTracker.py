import numpy as np
from collections import deque

class Tracker:
    def __init__(self, threshold=0.4, target_types=[0, 2]): # basically person and car
        self.tracking_list = deque()
        self.target_types = target_types

        self.threshold = threshold
        self.objectId = 1


    def mask_comparison(self, src, target):
        score = np.sum(np.concatenate(src & target))/np.sum(np.concatenate(src))
        return score


    def parseObject(self, new_objects):
        tmp = []
        for _object in new_objects:
            if _object.typeid in self.target_types:
                tmp.append(_object)
        return tmp

    def track(self, new_objects):
        new_objects = self.parseObject(new_objects)

        if len(self.tracking_list) == 0:
            for _object in new_objects:
                self.tracking_list.append(_object)
        else:
            self.reset()
            for _object in new_objects:
                flag = True
                for v in self.tracking_list:
                    if v.typeid == _object.typeid:
                        s = self.mask_comparison(_object.mask, v.mask)
                        if s > self.threshold:
                            if v.id == 0:
                                v.setId(self.objectId)
                                self.objectId += 1
                            v.setObject(_object)
                            v.track_on_screen = True
                            flag=True
                            break
                        else:
                            flag = False
                if flag == False:
                    _object.setId(self.objectId)
                    self.objectId+=1
                    self.tracking_list.append(_object)
        
        self.updateInfos()

        return list(self.tracking_list)
    
    def reset(self):
        for v in self.tracking_list:
            v.track_on_screen = False

    def updateInfos(self,):
        for j, v in enumerate(list(self.tracking_list)):
            if v.track_on_screen is False:
                v.track_count += 1
            if v.track_count > 5:
                if len(self.tracking_list) == 1:
                    self.tracking_list = deque()
                else:
                    del self.tracking_list[j]
