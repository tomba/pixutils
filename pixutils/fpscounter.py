import time

class FPSCounter:
    def __init__(self, name: str=''):
        self.start_time = None
        self.frame_count = 0
        if name:
            name += ' '
        self.name = name

    def tick(self):
        if self.start_time is None:
            self.start_time = time.monotonic()
            self.frame_count = 0

        self.frame_count += 1
        current_time = time.monotonic()
        elapsed_time = current_time - self.start_time

        if elapsed_time >= 2:
            fps = self.frame_count / elapsed_time
            print(f'{self.name}FPS: {fps:.2f}')
            self.start_time = current_time
            self.frame_count = 0
