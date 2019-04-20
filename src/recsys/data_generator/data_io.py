import itertools
import multiprocessing
from csv import DictReader, DictWriter
from multiprocessing import Queue

STOP = 1


class DataIO:
    def __init__(self, limit=None):
        self.limit = limit

        self.reader_queue = multiprocessing.Queue(maxsize=1000)
        reader_process = multiprocessing.Process(target=self.csv_reader, args=(self.reader_queue,))
        reader_process.start()

        self.writer_queue = multiprocessing.Queue(maxsize=1000)
        writer_process = multiprocessing.Process(target=self.csv_writer, args=(self.writer_queue,))
        writer_process.start()

    def csv_writer(self, queue: Queue):
        with open("../../../data/events_sorted_trans.csv", "wt") as out:
            first_row = True
            while True:
                output_obs = queue.get()
                if output_obs == STOP:
                    break
                if first_row:
                    dw = DictWriter(out, fieldnames=output_obs.keys())
                    dw.writeheader()
                    first_row = False
                dw.writerow(output_obs)

    def csv_reader(self, queue: Queue, chunk_size=10000, limit=None):
        with open("../../../data/events_sorted.csv") as inp:
            dr = DictReader(inp)
            for i, row in enumerate(dr):
                if self.limit and i > self.limit:
                    queue.put(STOP)
                else:
                    queue.put(row)
        queue.put(STOP)

    def rows(self):
        while True:
            v = self.reader_queue.get()
            if v == STOP:
                break
            yield v
