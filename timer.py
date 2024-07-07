import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.paused_time = None
        self.total_pause_duration = 0

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()
        else:
            if self.paused_time is not None:
                # If the timer was paused, we account for the time it was paused
                self.total_pause_duration += time.time() - self.paused_time
                self.paused_time = None

    def pause(self):
        if self.start_time is not None and self.paused_time is None:
            self.paused_time = time.time()

    def current_time(self):
        if self.start_time is None:
            return 0
        if self.paused_time is not None:
            return self.paused_time - self.start_time - self.total_pause_duration
        return time.time() - self.start_time - self.total_pause_duration

    def reset(self):
        self.start_time = None
        self.paused_time = None
        self.total_pause_duration = 0

if __name__ == "__main__":
    timer = Timer()
    timer.start()
    time.sleep(2)  # wait for 2 seconds
    print(timer.current_time())  # should be close to 2 seconds
    timer.pause()
    time.sleep(1)  # wait for 1 second
    print(timer.current_time())  # should still be close to 2 seconds because timer is paused
    timer.start()  # resume
    time.sleep(2)  # wait for 2 seconds
    print(timer.current_time())  # should be close to 4 seconds now
    timer.reset()
    print(timer.current_time())  # should be 0
