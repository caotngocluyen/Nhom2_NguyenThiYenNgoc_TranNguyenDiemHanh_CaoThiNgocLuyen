"""
04_prepare_data.py - Chia dữ liệu thành train/validation/test
MỤC ĐÍCH:
    Chia tập dữ liệu ratings đã làm sạch thành 3 tập:
    - Train (80%): Dùng để huấn luyện mô hình
    - Validation (10%): Dùng để tuning tham số và chọn mô hình tốt nhất
    - Test (10%): Dùng để đánh giá cuối cùng (CHỈ dùng 1 lần!)

    Chiến lược chia: Temporal Split (theo thời gian)
    Lý do: Tránh data leakage (rò rỉ dữ liệu tương lai vào quá khứ).
    Trong thực tế, ta chỉ biết lịch sử quá khứ để dự đoán tương lai.

ĐẦU VÀO:
    - data/processed/ratings_clean.parquet

ĐẦU RA:
    - data/processed/train.parquet
    - data/processed/val.parquet
    - data/processed/test.parquet
    - reports/split_stats.txt

CÁCH CHẠY:
    python src/04_prepare_data.py
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.REPORTS_DIR, "04_prepare_data.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def temporal_split(df_ratings):
    """
    Chia dữ liệu theo thời gian (Temporal Split).
    
    Nguyên lý:
    - Sắp xếp tất cả ratings theo timestamp (thời điểm đánh giá)
    - 80% đầu tiên  train (các rating cũ hơn)
    - 10% tiếp theo  validation
    - 10% cuối cùng  test (các rating mới nhất)
    
    Ưu điểm: Phản ánh kịch bản thực tế - ta dùng lịch sử để dự đoán tương lai.
    
    Vấn đề cần chú ý:
    - Người dùng hoặc phim trong val/test có thể chưa xuất hiện trong train
      (cold-start)  Xử lý khi đánh giá
    """
    logger.info("Chia dữ liệu theo thời gian (Temporal Split)")

    if "timestamp" in df_ratings.columns:
        # Sắp xếp theo timestamp
        df_sorted = df_ratings.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        logger.info("  Chia theo cột timestamp (thời gian thực tế)")
    else:
        # Nếu không có timestamp, dùng random split
        logger.warning("  Không có cột timestamp! Chuyển sang random split.")
        df_sorted = df_ratings.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)

    n = len(df_sorted)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)
    # n_test = phần còn lại (để đảm bảo n_train + n_val + n_test = n)

    df_train = df_sorted.iloc[:n_train].copy()
    df_val   = df_sorted.iloc[n_train : n_train + n_val].copy()
    df_test  = df_sorted.iloc[n_train + n_val:].copy()

    # Thống kê
    logger.info(f"\n Kết quả chia tập ")
    logger.info(f"  Tổng ratings: {n:,}")
    logger.info(f"  Train:       {len(df_train):,} ({len(df_train)/n*100:.1f}%)")
    logger.info(f"  Validation:  {len(df_val):,} ({len(df_val)/n*100:.1f}%)")
    logger.info(f"  Test:        {len(df_test):,} ({len(df_test)/n*100:.1f}%)")

    # Kiểm tra timestamp range
    if "timestamp" in df_sorted.columns:
        def ts2date(ts): return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        logger.info(f"\n Khoảng thời gian ")
        logger.info(f"  Train:  {ts2date(df_train['timestamp'].min())}  {ts2date(df_train['timestamp'].max())}")
        logger.info(f"  Val:    {ts2date(df_val['timestamp'].min())}  {ts2date(df_val['timestamp'].max())}")
        logger.info(f"  Test:   {ts2date(df_test['timestamp'].min())}  {ts2date(df_test['timestamp'].max())}")

    return df_train, df_val, df_test


def kiem_tra_data_leakage(df_train, df_val, df_test):
    """
    Kiểm tra xem có bị rò rỉ dữ liệu không.
    
    Data Leakage (rò rỉ dữ liệu) xảy ra khi:
    - Cùng cặp (userId, movieId) xuất hiện cả trong train lẫn val/test
    - Điều này làm metric đánh giá quá lạc quan (inflated metrics)
    
    Trong Temporal Split, điều này không xảy ra vì mỗi rating chỉ
    xuất hiện một lần và ở một tập duy nhất.
    """
    logger.info("\n Kiểm tra data leakage ")

    train_pairs = set(zip(df_train["userId"], df_train["movieId"]))
    val_pairs   = set(zip(df_val["userId"],   df_val["movieId"]))
    test_pairs  = set(zip(df_test["userId"],  df_test["movieId"]))

    # Kiểm tra chồng chéo
    train_val_overlap  = len(train_pairs & val_pairs)
    train_test_overlap = len(train_pairs & test_pairs)
    val_test_overlap   = len(val_pairs   & test_pairs)

    logger.info(f"  Chồng chéo train-val:  {train_val_overlap} cặp (userId, movieId)")
    logger.info(f"  Chồng chéo train-test: {train_test_overlap} cặp")
    logger.info(f"  Chồng chéo val-test:   {val_test_overlap} cặp")

    if train_val_overlap > 0 or train_test_overlap > 0:
        logger.warning("  CÓ data leakage! Cần xem lại cách chia.")
    else:
        logger.info("  Không có data leakage.")


def thong_ke_tap(df_train, df_val, df_test):
    """
    Thống kê overlap của users và movies giữa các tập.
    Quan trọng để ước tính mức độ cold-start.
    """
    logger.info("\n Thống kê overlap users/movies ")

    train_users  = set(df_train["userId"].unique())
    train_movies = set(df_train["movieId"].unique())
    val_users    = set(df_val["userId"].unique())
    val_movies   = set(df_val["movieId"].unique())
    test_users   = set(df_test["userId"].unique())
    test_movies  = set(df_test["movieId"].unique())

    # Users mới trong val/test (không có trong train)  Cold-start
    cold_users_val  = val_users - train_users
    cold_users_test = test_users - train_users
    cold_movies_val  = val_movies - train_movies
    cold_movies_test = test_movies - train_movies

    logger.info(f"  Users trong train:    {len(train_users):,}")
    logger.info(f"  Users trong val:      {len(val_users):,}")
    logger.info(f"    Mới (cold-start):   {len(cold_users_val):,} ({len(cold_users_val)/len(val_users)*100:.1f}%)")
    logger.info(f"  Users trong test:     {len(test_users):,}")
    logger.info(f"    Mới (cold-start):   {len(cold_users_test):,} ({len(cold_users_test)/len(test_users)*100:.1f}%)")
    logger.info(f"\n  Movies trong train:   {len(train_movies):,}")
    logger.info(f"  Movies cold-start val:  {len(cold_movies_val):,}")
    logger.info(f"  Movies cold-start test: {len(cold_movies_test):,}")

    return {
        "cold_users_val": len(cold_users_val),
        "cold_users_test": len(cold_users_test),
        "cold_movies_test": len(cold_movies_test),
    }


def luu_tap_du_lieu(df_train, df_val, df_test):
    """Lưu 3 tập dữ liệu ra file Parquet."""
    logger.info("\nLưu các tập dữ liệu")

    df_train.to_parquet(config.TRAIN_FILE, index=False)
    df_val.to_parquet(config.VAL_FILE, index=False)
    df_test.to_parquet(config.TEST_FILE, index=False)

    for name, path, df in [
        ("Train", config.TRAIN_FILE, df_train),
        ("Val",   config.VAL_FILE,   df_val),
        ("Test",  config.TEST_FILE,  df_test),
    ]:
        kich_thuoc = os.path.getsize(path) / 1024**2
        logger.info(f"  {name}: {len(df):,} dòng  {path} ({kich_thuoc:.1f} MB)")


def main():
    """Hàm chính - chia dữ liệu."""
    # Kiểm tra nếu đã chia rồi
    if all(os.path.exists(f) for f in [config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE]):
        logger.info("Đã có file train/val/test. Bỏ qua.")
        logger.info("Để chia lại, xóa các file trong data/processed/ hoặc set FORCE_RESPLIT=1")
        if not os.environ.get("FORCE_RESPLIT"):
            return

    # Tải dữ liệu đã làm sạch
    if not os.path.exists(config.RATINGS_CLEAN_FILE):
        logger.error("Chưa có ratings_clean.parquet. Chạy 02_preprocess.py trước!")
        sys.exit(1)

    logger.info("Tải ratings_clean.parquet")
    df_ratings = pd.read_parquet(config.RATINGS_CLEAN_FILE)
    logger.info(f"  Tổng ratings: {len(df_ratings):,}")

    # Chia tập
    df_train, df_val, df_test = temporal_split(df_ratings)

    # Kiểm tra data leakage
    kiem_tra_data_leakage(df_train, df_val, df_test)

    # Thống kê cold-start
    stats = thong_ke_tap(df_train, df_val, df_test)

    # Lưu file
    luu_tap_du_lieu(df_train, df_val, df_test)

    logger.info("\n Chia tập hoàn tất.")
    logger.info(" Bước tiếp theo: chạy src/05_train_baseline.py")


if __name__ == "__main__":
    main()
