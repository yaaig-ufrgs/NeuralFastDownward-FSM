import logging
from time import perf_counter

_log = logging.getLogger(__name__)

class Timer(object):
    def __init__(self, time_limit: int):
        self.last_start_time = 0.0
        self.accumulated_time = 0.0
        self.time_limit = time_limit
        self.timeout = False

    def start(self):
        if self.last_start_time:
            raise RuntimeError("Timer is already started!")
        else:
            self.last_start_time = perf_counter()
            return self

    def pause(self):
        if self.start_time:
            self.accumulated_time += perf_counter() - self.last_start_time
            self.start_time = None
        else:
            raise RuntimeError("Timer has not been started!")

    def check_timeout(self) -> bool:
        if not self.timeout:
            self.timeout = self.accumulated_time + perf_counter() - \
                self.last_start_time > self.time_limit
        return self.timeout
