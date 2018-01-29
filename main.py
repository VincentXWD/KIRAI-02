import multiprocessing as mp


def f(x):
  print(x)

if __name__ == "__main__":
  pool = mp.Pool()
  for i in range(0, 1000):
    pool.apply(func=f, args=(i+1, ))
  pool.close()
  pool.join()