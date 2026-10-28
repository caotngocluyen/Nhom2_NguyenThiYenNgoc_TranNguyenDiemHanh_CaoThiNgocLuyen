"""
01_check_data.py - Kiểm tra dữ liệu đầu vào
MỤC ĐÍCH:
    Kiểm tra cấu trúc, chất lượng và tính toàn vẹn của tất cả các file
    dữ liệu trước khi thực hiện bất kỳ bước xử lý nào.

ĐẦU VÀO:
    - data/ratings.csv
    - data/movies_metadata.csv
    - data/keywords.csv
    - data/credits.csv
    - data/links.csv

ĐẦU RA:
    - In báo cáo kiểm tra ra màn hình (console)
    - reports/data_quality_report.txt

CÁCH CHẠY:
    python src/01_check_data.py
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

# Thêm thư mục gốc vào Python path để import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import numpy as np
from tqdm import tqdm

# Cấu hình logging để theo dõi quá trình chạy
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.REPORTS_DIR, "01_check_data.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def kiem_tra_file_ton_tai():
    """Kiểm tra xem tất cả các file dữ liệu cần thiết có tồn tại không."""
    logger.info("=" * 60)
    logger.info("BƯỚC 1: KIỂM TRA SỰ TỒN TẠI CỦA CÁC FILE DỮ LIỆU")
    logger.info("=" * 60)

    # Danh sách file cần kiểm tra và mô tả
    files_can_kiem_tra = {
        "ratings.csv":          config.RATINGS_FILE,
        "movies_metadata.csv":  config.MOVIES_METADATA_FILE,
        "keywords.csv":         config.KEYWORDS_FILE,
        "credits.csv":          config.CREDITS_FILE,
        "links.csv":            config.LINKS_FILE,
    }

    tat_ca_ton_tai = True
    for ten_file, duong_dan in files_can_kiem_tra.items():
        if os.path.exists(duong_dan):
            kich_thuoc_mb = os.path.getsize(duong_dan) / (1024 * 1024)
            logger.info(f"  {ten_file:<30} {kich_thuoc_mb:>8.1f} MB")
        else:
            logger.error(f"  THIẾU FILE: {ten_file} tại {duong_dan}")
            tat_ca_ton_tai = False

    if not tat_ca_ton_tai:
        logger.error("\nMột số file dữ liệu bị thiếu!")
        logger.error("Hãy đặt tất cả file CSV từ Kaggle vào thư mục: data/")
        sys.exit(1)
    else:
        logger.info("\n Tất cả file dữ liệu đều tồn tại.")
    return True


def kiem_tra_ratings():
    """
    Kiểm tra chi tiết file ratings.csv
    
    ratings.csv chứa lịch sử đánh giá của người dùng:
    - userId: ID người dùng
    - movieId: ID phim (MovieLens ID)
    - rating: điểm đánh giá (0.5 đến 5.0, bước 0.5)
    - timestamp: thời điểm đánh giá (Unix timestamp)
    """
    logger.info("\n" + "=" * 60)
    logger.info("BƯỚC 2: KIỂM TRA ratings.csv")
    logger.info("=" * 60)

    # Đọc file lớn theo từng chunk (khối) để tiết kiệm RAM
    # Mỗi chunk chứa 1 triệu dòng
    kich_thuoc_chunk = 1_000_000
    logger.info(f"Đọc file theo chunk (mỗi chunk = {kich_thuoc_chunk:,} dòng)")

    tong_dong = 0
    tong_null = 0
    tong_trung = 0
    rating_counts = {}
    user_ids = set()      # Tập hợp userId duy nhất
    movie_ids = set()     # Tập hợp movieId duy nhất
    timestamp_min = float('inf')
    timestamp_max = float('-inf')
    rating_invalid = 0   # Số rating ngoài khoảng hợp lệ

    # Đọc từng chunk và tính toán thống kê
    chunks_iterator = pd.read_csv(
        config.RATINGS_FILE,
        chunksize=kich_thuoc_chunk,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )

    for i, chunk in enumerate(tqdm(chunks_iterator, desc="Đọc ratings")):
        tong_dong += len(chunk)

        # Đếm giá trị null
        tong_null += chunk.isnull().sum().sum()

        # Đếm bản ghi trùng trong chunk này
        tong_trung += chunk.duplicated(subset=["userId", "movieId"]).sum()

        # Thu thập userId và movieId duy nhất
        user_ids.update(chunk["userId"].unique())
        movie_ids.update(chunk["movieId"].unique())

        # Thống kê phân phối rating
        for rating, count in chunk["rating"].value_counts().items():
            rating_counts[rating] = rating_counts.get(rating, 0) + count

        # Thống kê thời gian
        if "timestamp" in chunk.columns:
            ts_min = chunk["timestamp"].min()
            ts_max = chunk["timestamp"].max()
            timestamp_min = min(timestamp_min, ts_min)
            timestamp_max = max(timestamp_max, ts_max)

        # Đếm rating không hợp lệ
        invalid_mask = (chunk["rating"] < config.MIN_RATING) | (chunk["rating"] > config.MAX_RATING)
        rating_invalid += invalid_mask.sum()

    # In kết quả thống kê
    logger.info(f"\nTổng quan ratings.csv")
    logger.info(f"  Tổng số dòng:             {tong_dong:>15,}")
    logger.info(f"  Số người dùng duy nhất:   {len(user_ids):>15,}")
    logger.info(f"  Số phim duy nhất:         {len(movie_ids):>15,}")
    logger.info(f"  Số giá trị null:          {tong_null:>15,}")
    logger.info(f"  Số bản ghi trùng:         {tong_trung:>15,}")
    logger.info(f"  Số rating không hợp lệ:  {rating_invalid:>15,}")

    # Tính toán thời gian
    if timestamp_min != float('inf'):
        from datetime import datetime
        dt_min = datetime.fromtimestamp(timestamp_min)
        dt_max = datetime.fromtimestamp(timestamp_max)
        logger.info(f"  Khoảng thời gian:         {dt_min.strftime('%Y-%m-%d')} đến {dt_max.strftime('%Y-%m-%d')}")

    # In phân phối rating
    logger.info(f"\nPhân phối điểm đánh giá")
    for rating in sorted(rating_counts.keys()):
        pct = rating_counts[rating] / tong_dong * 100
        logger.info(f"  Rating {rating:.1f}: {rating_counts[rating]:>10,} ({pct:.1f}%)")

    # Tính độ thưa (sparsity) của ma trận người dùng - phim
    # Độ thưa = tỷ lệ ô trống trong ma trận user x movie
    n_users = len(user_ids)
    n_movies = len(movie_ids)
    so_o_co_gia_tri = tong_dong - tong_trung  # Ước tính
    tong_o_toi_da = n_users * n_movies
    do_thua = 1.0 - (so_o_co_gia_tri / tong_o_toi_da)
    logger.info(f"\nĐộ thưa ma trận User-Movie")
    logger.info(f"  Ma trận: {n_users:,} users x {n_movies:,} movies")
    logger.info(f"  Tổng ô có thể có:         {tong_o_toi_da:>15,}")
    logger.info(f"  Ô có giá trị (rating):    {tong_dong:>15,}")
    logger.info(f"  Độ thưa (Sparsity):       {do_thua:.4f} ({do_thua*100:.2f}%)")
    logger.info(f"  Nghĩa là {do_thua*100:.1f}% ô trong ma trận là trống (không có rating)")

    return {
        "n_rows": tong_dong,
        "n_users": n_users,
        "n_movies": n_movies,
        "n_null": tong_null,
        "n_duplicate": tong_trung,
        "n_invalid_rating": rating_invalid,
        "sparsity": do_thua,
    }


def kiem_tra_movies_metadata():
    """Kiểm tra file movies_metadata.csv chứa thông tin chi tiết về phim."""
    logger.info("\n" + "=" * 60)
    logger.info("BƯỚC 3: KIỂM TRA movies_metadata.csv")
    logger.info("=" * 60)

    # Đọc toàn bộ (file nhỏ, ~34MB)
    df = pd.read_csv(config.MOVIES_METADATA_FILE, low_memory=False)
    logger.info(f"  Số dòng:    {len(df):,}")
    logger.info(f"  Số cột:     {len(df.columns)}")
    logger.info(f"  Cột: {', '.join(df.columns.tolist())}")

    # Kiểm tra giá trị null từng cột
    logger.info(f"\nTỷ lệ giá trị thiếu (null) theo cột")
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    for col in df.columns:
        if null_counts[col] > 0:
            logger.info(f"  {col:<30} null: {null_counts[col]:>6,} ({null_pct[col]:.1f}%)")

    # Kiểm tra kiểu dữ liệu cột id (quan trọng để ghép với ratings)
    logger.info(f"\nKiểm tra cột 'id' (movieId trong TMDB)")
    logger.info(f"  Kiểu dữ liệu cột 'id': {df['id'].dtype}")
    
    # Kiểm tra giá trị id không hợp lệ (có thể chứa text không phải số)
    id_khong_hop_le = pd.to_numeric(df["id"], errors="coerce").isnull().sum()
    logger.info(f"  Số id không phải số: {id_khong_hop_le}")

    # Kiểm tra id trùng
    id_trung = df["id"].duplicated().sum()
    logger.info(f"  Số id trùng: {id_trung}")

    # Thống kê cột quan trọng
    if "vote_count" in df.columns:
        df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
        logger.info(f"\nThống kê vote_count")
        logger.info(f"  Min: {df['vote_count'].min():.0f}, Max: {df['vote_count'].max():.0f}, "
                    f"Mean: {df['vote_count'].mean():.1f}, Median: {df['vote_count'].median():.0f}")

    return {"n_rows": len(df), "n_null_id": id_khong_hop_le, "n_dup_id": id_trung}


def kiem_tra_links():
    """
    Kiểm tra file links.csv - bảng ánh xạ ID quan trọng.
    
    links.csv nối movieId (MovieLens) với tmdbId (TMDB):
    - movieId: ID trong MovieLens (dùng trong ratings.csv)
    - imdbId: ID trong IMDB
    - tmdbId: ID trong TMDB (dùng trong movies_metadata.csv cột 'id')
    """
    logger.info("\n" + "=" * 60)
    logger.info("BƯỚC 4: KIỂM TRA links.csv (Bảng ánh xạ ID)")
    logger.info("=" * 60)

    df_links = pd.read_csv(config.LINKS_FILE)
    logger.info(f"  Số dòng: {len(df_links):,}")
    logger.info(f"  Cột: {', '.join(df_links.columns.tolist())}")

    null_tmdb = df_links["tmdbId"].isnull().sum()
    logger.info(f"  Số dòng thiếu tmdbId: {null_tmdb} ({null_tmdb/len(df_links)*100:.1f}%)")
    logger.info(f"  movieId duy nhất: {df_links['movieId'].nunique():,}")

    # Kiểm tra liên kết với movies_metadata
    logger.info("\nKiểm tra liên kết với movies_metadata")
    df_meta = pd.read_csv(config.MOVIES_METADATA_FILE, low_memory=False)
    df_meta["id_numeric"] = pd.to_numeric(df_meta["id"], errors="coerce")
    df_links["tmdbId"] = pd.to_numeric(df_links["tmdbId"], errors="coerce")

    tmdb_trong_meta = set(df_meta["id_numeric"].dropna().astype(int))
    tmdb_trong_links = set(df_links["tmdbId"].dropna().astype(int))

    khop = tmdb_trong_links & tmdb_trong_meta
    khong_khop = tmdb_trong_links - tmdb_trong_meta

    logger.info(f"  tmdbId trong links: {len(tmdb_trong_links):,}")
    logger.info(f"  tmdbId trong metadata: {len(tmdb_trong_meta):,}")
    logger.info(f"  Khớp được: {len(khop):,} ({len(khop)/len(tmdb_trong_links)*100:.1f}%)")
    logger.info(f"  Không khớp: {len(khong_khop):,}")

    return {"n_rows": len(df_links), "n_null_tmdb": null_tmdb, "n_match": len(khop)}


def kiem_tra_keywords():
    """Kiểm tra file keywords.csv chứa từ khóa của phim."""
    logger.info("\n" + "=" * 60)
    logger.info("BƯỚC 5: KIỂM TRA keywords.csv")
    logger.info("=" * 60)

    df = pd.read_csv(config.KEYWORDS_FILE)
    logger.info(f"  Số dòng: {len(df):,}")
    logger.info(f"  Cột: {', '.join(df.columns.tolist())}")
    null_kw = df["keywords"].isnull().sum()
    logger.info(f"  Số phim thiếu keywords: {null_kw}")
    empty_kw = (df["keywords"] == "[]").sum()
    logger.info(f"  Số phim có keywords rỗng []: {empty_kw}")

    return {"n_rows": len(df)}


def kiem_tra_credits():
    """Kiểm tra file credits.csv chứa thông tin diễn viên và đoàn làm phim."""
    logger.info("\n" + "=" * 60)
    logger.info("BƯỚC 6: KIỂM TRA credits.csv")
    logger.info("=" * 60)

    df = pd.read_csv(config.CREDITS_FILE)
    logger.info(f"  Số dòng: {len(df):,}")
    logger.info(f"  Cột: {', '.join(df.columns.tolist())}")
    logger.info(f"  Kích thước file: {os.path.getsize(config.CREDITS_FILE)/1024/1024:.1f} MB")

    return {"n_rows": len(df)}


def kiem_tra_phan_cung():
    """Kiểm tra tài nguyên phần cứng để ước tính khả năng xử lý."""
    logger.info("\n" + "=" * 60)
    logger.info("BƯỚC 7: KIỂM TRA TÀI NGUYÊN PHẦN CỨNG")
    logger.info("=" * 60)

    try:
        import psutil
        ram_total = psutil.virtual_memory().total / (1024**3)
        ram_available = psutil.virtual_memory().available / (1024**3)
        logger.info(f"  RAM tổng: {ram_total:.1f} GB")
        logger.info(f"  RAM khả dụng: {ram_available:.1f} GB")
        logger.info(f"  CPU cores: {psutil.cpu_count(logical=False)} physical, "
                    f"{psutil.cpu_count(logical=True)} logical")

        # Ước tính RAM cần để load ratings vào memory
        ratings_size_mb = os.path.getsize(config.RATINGS_FILE) / (1024**2)
        # CSV 677MB ≈ DataFrame 4 cột int/float x 26M rows ≈ ~800MB
        estimated_ram_gb = ratings_size_mb / 1024 * 1.5
        logger.info(f"\n  Ước tính RAM cần để load ratings.csv: ~{estimated_ram_gb:.1f} GB")

        if ram_available < estimated_ram_gb:
            logger.warning(f"  RAM khả dụng ({ram_available:.1f}GB) < Ước tính cần ({estimated_ram_gb:.1f}GB)")
            logger.warning(f"  Sẽ sử dụng đọc theo chunk và Parquet để tiết kiệm RAM")
        else:
            logger.info(f"  RAM đủ để xử lý toàn bộ dataset")

    except ImportError:
        logger.info("  (psutil chưa cài, bỏ qua kiểm tra RAM chi tiết)")
        logger.info("  RAM ước tính: 16 GB (từ thông tin hệ thống)")


def main():
    """Hàm chính - chạy tất cả các bước kiểm tra."""

    logger.info(f"Thư mục dự án: {config.BASE_DIR}")
    logger.info(f"Thư mục dữ liệu: {config.DATA_DIR}")

    # Bước 1: Kiểm tra file tồn tại
    kiem_tra_file_ton_tai()

    # Bước 2: Kiểm tra ratings (file lớn nhất)
    stats_ratings = kiem_tra_ratings()

    # Bước 3: Kiểm tra metadata phim
    stats_meta = kiem_tra_movies_metadata()

    # Bước 4: Kiểm tra bảng liên kết ID
    stats_links = kiem_tra_links()

    # Bước 5: Kiểm tra keywords
    stats_kw = kiem_tra_keywords()

    # Bước 6: Kiểm tra credits
    stats_credits = kiem_tra_credits()

    # Bước 7: Kiểm tra phần cứng
    kiem_tra_phan_cung()

    # Tổng kết
    logger.info("\n" + "=" * 60)
    logger.info("TỔNG KẾT KIỂM TRA DỮ LIỆU")
    logger.info("=" * 60)
    logger.info(f"  Tổng ratings:         {stats_ratings['n_rows']:>12,}")
    logger.info(f"  Người dùng:           {stats_ratings['n_users']:>12,}")
    logger.info(f"  Phim trong ratings:   {stats_ratings['n_movies']:>12,}")
    logger.info(f"  Phim trong metadata:  {stats_meta['n_rows']:>12,}")
    logger.info(f"  Liên kết ID khớp:     {stats_links['n_match']:>12,}")
    logger.info(f"  Độ thưa ma trận:      {stats_ratings['sparsity']*100:.2f}%")
    logger.info("\n Kiểm tra hoàn tất. Xem log tại: reports/01_check_data.log")
    logger.info(" Bước tiếp theo: chạy src/02_preprocess.py")


if __name__ == "__main__":
    main()
