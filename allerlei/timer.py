import time


class Timer(object):
    def __init__(self, msg=None, suppress_print=False, verbose=True):
        self.msg = msg
        self._suppress = suppress_print
        self._verbose = verbose

    def __enter__(self):
        self.tstart = time.time()
        if self._verbose and self.msg is not None:
            print(f"Starting: {self.msg}")
        return self

    def __exit__(self, type, value, traceback):
        dt = time.time() - self.tstart
        self.dt = dt
        if self.msg is not None and not self._suppress:
            print(self.get_message())

    def get_elapsed_time(self):
        return self.dt

    def get_message(self):
        prefix = 'Elapsed' if self.msg is None else self.msg
        return f"{prefix}: {str(self)}"

    def __repr__(self):
        return self.get_message()

    def __str__(self):
        if self.dt < 1:
            ss = f"{1e3 * self.dt:.2f}ms"
        else:
            ss = f'{self.dt:.2f}s'
            if self.dt > 60:
                ss += f' ({self.dt / 60:.2f}min)'
        return ss
