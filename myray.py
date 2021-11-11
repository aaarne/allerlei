import ray
from .tools import Progressbar


def __gen(enqueue_func, it, chunk):
    futures = []
    for _ in range(chunk):
        if it:
            futures.append(enqueue_func(it.pop(0)))
        else:
            break

    while (len(it) + len(futures)) > 0:
        finished, futures = ray.wait(futures)

        for _ in finished:
            if it:
                futures.append(enqueue_func(it.pop(0)))

        yield from ray.get(finished)


def raymap(remotefunc, iterable, show_progress=True, chunk_size=12):
    data = [x for x in iterable]
    gen = __gen(lambda x: remotefunc.remote(x), data, chunk_size)
    if show_progress:
        return Progressbar(len(data))(gen)
    else:
        return gen


def raystarmap(remotefunc, iterable, show_progress=True, chunk_size=12):
    data = [x for x in iterable]
    gen = __gen(lambda x: remotefunc.remote(*x), data, chunk_size)
    if show_progress:
        return Progressbar(len(data))(gen)
    else:
        return gen
