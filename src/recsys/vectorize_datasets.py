from recsys.log_utils import get_logger
from recsys.vectorizers import make_vectorizer_1, VectorizeChunks

logger = get_logger()

if __name__ == "__main__":
    logger = get_logger()
    logger.info("Starting vectorizing")
    vectorize_chunks = VectorizeChunks(
        vectorizer=lambda: make_vectorizer_1(),
        input_files="../../data/proc/raw_csv/*.csv",
        output_folder="../../data/proc/vectorizer_1/",
        n_jobs=7,
    )
    vectorize_chunks.vectorize_all()
