import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.semantic_embeddings import SemanticContentRecommender

log_file = os.path.join(config.REPORTS_DIR, "07_semantic_training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def train_semantic_embeddings():
    """
    Huan luyen va luu tru mo hinh Semantic Embeddings dua tren
    Sentence-Transformers (all-MiniLM-L6-v2) cho toan bo tap phim.
    """
    logger.info("Bat dau huan luyen Semantic Embeddings")

    movies_path = os.path.join(config.DATA_PROCESSED_DIR, "movies_clean.parquet")
    train_path = os.path.join(config.DATA_PROCESSED_DIR, "train_improved.parquet")

    logger.info("Doc du lieu phim tu %s va tap train tu %s", movies_path, train_path)
    df_movies = pd.read_parquet(
        movies_path,
        columns=["movieId", "title", "overview", "genres_text", "vote_average", "vote_count"]
    )
    df_train = pd.read_parquet(train_path, columns=["movieId"])

    train_movie_ids = set(df_train["movieId"].unique())
    logger.info("Tong so phim xuat hien trong tap train: %d", len(train_movie_ids))

    df_target = df_movies[df_movies["movieId"].isin(train_movie_ids)].copy()
    logger.info("So luong phim can ma hoa embedding: %d", len(df_target))

    recommender = SemanticContentRecommender(model_name=config.SEMANTIC_MODEL_NAME)
    recommender.fit(df_target, batch_size=256, max_seq_length=config.SEMANTIC_MAX_SEQ_LENGTH)

    recommender.save(config.MODELS_DIR)
    logger.info("Da luu Semantic Recommender tai: %s", config.MODELS_DIR)

    # Thu nghiem do tuong dong mau
    sample_movie_id = 1  # Toy Story
    if sample_movie_id in recommender.movie_idx_map:
        title = df_target[df_target["movieId"] == sample_movie_id]["title"].values[0]
        logger.info("Top 5 phim tuong dong ve ngu nghia voi '%s':", title)
        similars = recommender.get_similar_movies(sample_movie_id, n=5)
        for s_id, score, s_title in similars:
            logger.info("  Phim: %-35s | Tuong dong: %.4f", s_title, score)

    logger.info("Hoan tat qua trinh huan luyen Semantic Embeddings")


if __name__ == "__main__":
    train_semantic_embeddings()
