import logging
import threading
from readwrite import write_to_disk

class Tags():
    def __init__(self, logger):
        self.tags_dict = {}
        self.next_id = 0
        self._lock = threading.Lock()
        self.logger = logger
    
    def update_tags(self, tags):
        tag_ids = []
        with self._lock:
            for tag in tags:
                if tag not in self.tags_dict:
                    self.tags_dict[tag] = self.next_id
                    self.logger.info("{},{}".format(tag, self.next_id))
                    self.next_id += 1
                tag_ids.append(self.tags_dict[tag])
        return tag_ids