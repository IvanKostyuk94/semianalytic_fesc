import numpy as np
import time
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


def square(x):
    return x**2


def make_global(a):
    # Declare 'global_cache' to be Global
    global global_a
    # Update 'global_cache' with a value, now *implicitly* accessible in func
    global_a = a


if __name__ == "__main__":

    a = np.random.rand(int(1e9))
    a = [i for i in a]
    # start = time.time()
    # test = list(map(square, a))
    # end = time.time()
    # print(end - start)

    Nprocess = 40
    start = time.time()
    chunks = np.array_split(a, Nprocess)

    start = time.time()
    with Pool(Nprocess) as executor:
        pool_result = executor.map(square, chunks)
    # with ThreadPoolExecutor(max_workers=Nprocess) as executor:
    #     # Map each element to the square function concurrently
    #     pool_result = executor.map(square, chunks)

    # with Pool(Nprocess, initializer=make_global, initargs=(a,)) as pool:
    #     results = pool.map(square, a)
    end = time.time()

    # Flatten the results list of lists into a single list
    # pool_result = np.concatenate(list(results))

    print(end - start)
    # print(np.allclose(test, pool_result))
