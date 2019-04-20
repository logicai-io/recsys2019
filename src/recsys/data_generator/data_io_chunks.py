import itertools
import multiprocessing
from csv import DictReader, DictWriter
from multiprocessing import Queue

STOP = 1


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


class DataIOChunks:
    def __init__(self):
        self.reader_queue = multiprocessing.Queue(maxsize=10)
        reader_process = multiprocessing.Process(target=self.csv_reader, args=(self.reader_queue,))
        reader_process.start()

        self.writer_queue = multiprocessing.Queue(maxsize=10)
        writer_process = multiprocessing.Process(target=self.csv_writer, args=(self.writer_queue,))
        writer_process.start()

    def process(self, process_chunk_func):
        while True:
            chunk = self.reader_queue.get()
            print(len(chunk))
            if chunk == STOP:
                break
            print("Chunk size", len(chunk))
            new_ch = process_chunk_func(chunk)
            self.writer_queue.put(new_ch)
        self.reader_queue.join()
        self.writer_queue.join()

    def csv_writer(self, queue: Queue):
        with open("../../../data/events_sorted_trans.csv", "wt") as out:
            first_row = True
            while True:
                output_obs = queue.get()
                if output_obs == STOP:
                    break
                if first_row:
                    dw = DictWriter(out, fieldnames=output_obs[0].keys())
                    dw.writeheader()
                    first_row = False
                dw.writerows(output_obs)

    def csv_reader(self, queue: Queue, chunk_size=10000, limit=None):
        with open("../../../data/events_sorted.csv") as inp:
            dr = DictReader(inp)
            print("Reading rows")
            for ch in grouper(chunk_size, dr):
                queue.put(ch)
        queue.put(STOP)
