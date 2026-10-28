"""
02_preprocess.py - Tiền xử lý và làm sạch dữ liệu
MỤC ĐÍCH:
    Làm sạch, chuẩn hóa và lưu dữ liệu đã xử lý để các bước sau dùng lại.
    Thống kê rõ số dòng trước/sau mỗi bước lọc.

ĐẦU VÀO:
    - data/ratings.csv
    - data/movies_metadata.csv
    - data/keywords.csv
    - data/credits.csv
    - data/links.csv

ĐẦU RA:
    - data/processed/ratings_clean.parquet  (ratings đã làm sạch)
    - data/processed/movies_clean.parquet   (metadata phim đã làm sạch)
    - data/processed/id_map.parquet         (bảng ánh xạ movieId <-> tmdbId <-> index)

CÁCH CHẠY:
    python src/02_preprocess.py
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
import ast
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.REPORTS_DIR, "02_preprocess.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def lam_sach_ratings():
    """
    Làm sạch ratings.csv - bước quan trọng nhất do file rất lớn (~26M dòng).
    
    Các bước làm sạch:
    1. Loại bỏ dòng có giá trị null
    2. Loại bỏ rating ngoài khoảng hợp lệ (0.5 - 5.0)
    3. Loại bỏ bản ghi trùng (cùng userId + movieId)
    4. Chuyển kiểu dữ liệu phù hợp để tiết kiệm RAM
    5. Lưu dạng Parquet để đọc nhanh hơn
    """
    logger.info("=" * 60)
    logger.info("LÀM SẠCH ratings.csv")
    logger.info("=" * 60)

    kich_thuoc_chunk = 1_000_000  # Mỗi chunk = 1 triệu dòng

    # Dùng list để gom chunks lại
    chunks_sach = []
    
    tong_doc = 0       # Tổng dòng đã đọc
    tong_null = 0      # Số dòng loại do null
    tong_invalid = 0   # Số dòng loại do rating sai
    tong_trung = 0     # Số dòng loại do trùng

    logger.info(f"Đọc và làm sạch theo chunk ({kich_thuoc_chunk:,} dòng/chunk)")
    logger.info("Lưu ý: Xử lý 26M dòng mất khoảng 3-7 phút")

    chunkwise_iter = pd.read_csv(
        config.RATINGS_FILE,
        chunksize=kich_thuoc_chunk,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
        }
    )

    for i, chunk in enumerate(tqdm(chunkwise_iter, desc="Làm sạch ratings")):
        n_truoc = len(chunk)
        tong_doc += n_truoc

        #  Bước A: Loại bỏ dòng có giá trị null 
        chunk = chunk.dropna(subset=["userId", "movieId", "rating"])
        n_sau_null = len(chunk)
        tong_null += (n_truoc - n_sau_null)

        #  Bước B: Loại bỏ rating ngoài khoảng hợp lệ 
        mask_hop_le = (chunk["rating"] >= config.MIN_RATING) & (chunk["rating"] <= config.MAX_RATING)
        chunk = chunk[mask_hop_le]
        n_sau_rating = len(chunk)
        tong_invalid += (n_sau_null - n_sau_rating)

        #  Bước C: Loại bỏ bản ghi trùng trong chunk 
        chunk = chunk.drop_duplicates(subset=["userId", "movieId"])
        n_sau_dup_chunk = len(chunk)
        tong_trung += (n_sau_rating - n_sau_dup_chunk)

        #  Bước D: Chuyển kiểu dữ liệu tối ưu 
        chunk = chunk.astype({
            "userId":  "int32",
            "movieId": "int32",
            "rating":  "float32",
        })
        
        # Giữ lại timestamp nếu có (dùng cho temporal split)
        if "timestamp" in chunk.columns:
            chunk["timestamp"] = chunk["timestamp"].astype("int64")

        chunks_sach.append(chunk)

    # Ghép tất cả chunks lại thành một DataFrame
    logger.info("Gộp các chunk lại")
    df_sach = pd.concat(chunks_sach, ignore_index=True)
    del chunks_sach

    # Đảm bảo không trùng lặp toàn cục (nếu có trùng giữa các chunk)
    n_truoc_dedup = len(df_sach)
    df_sach = df_sach.drop_duplicates(subset=["userId", "movieId"])
    tong_trung += (n_truoc_dedup - len(df_sach))

    # Thống kê kết quả làm sạch
    logger.info(f"\n Thống kê làm sạch ratings ")
    logger.info(f"  Dòng ban đầu:            {tong_doc:>12,}")
    logger.info(f"  Loại do null:            {tong_null:>12,}")
    logger.info(f"  Loại do rating sai:      {tong_invalid:>12,}")
    logger.info(f"  Loại do trùng:           {tong_trung:>12,}")
    logger.info(f"  Dòng còn lại:            {len(df_sach):>12,}")
    logger.info(f"  Tỷ lệ giữ lại:          {len(df_sach)/tong_doc*100:.2f}%")

    #  Lưu dạng Parquet 
    # Parquet nhanh hơn CSV 5-10x khi đọc và nén tốt hơn
    logger.info(f"\nLưu vào: {config.RATINGS_CLEAN_FILE}")
    df_sach.to_parquet(config.RATINGS_CLEAN_FILE, index=False, engine="pyarrow")
    
    kich_thuoc_parquet = os.path.getsize(config.RATINGS_CLEAN_FILE) / (1024**2)
    logger.info(f"  Kích thước file Parquet: {kich_thuoc_parquet:.1f} MB")
    logger.info(f"  (CSV gốc: {os.path.getsize(config.RATINGS_FILE)/1024/1024:.1f} MB)")

    return df_sach


def trich_xuat_ast_list(value_str, key="name"):
    """
    Hàm tiện ích: Trích xuất danh sách từ chuỗi JSON/AST trong CSV.
    
    Ví dụ: '[{"id": 1, "name": "Drama"}, {"id": 2, "name": "Action"}]'
    --> ["Drama", "Action"]
    
    Tham số:
        value_str: Chuỗi JSON-like
        key: Tên trường cần lấy (mặc định là "name")
    
    Trả về:
        Danh sách chuỗi (list of strings)
    """
    try:
        items = ast.literal_eval(value_str)
        if isinstance(items, list):
            return [item[key] for item in items if isinstance(item, dict) and key in item]
        return []
    except (ValueError, SyntaxError, TypeError):
        return []


def lam_sach_movies_metadata(df_ratings_sach):
    """
    Làm sạch movies_metadata.csv và ghép với keywords, credits, links.
    
    Quy trình:
    1. Đọc movies_metadata.csv, chuyển cột 'id' thành số nguyên
    2. Đọc links.csv để lấy bảng ánh xạ movieId (MovieLens) <-> tmdbId
    3. Ghép metadata theo tmdbId (qua links)
    4. Trích xuất thể loại, từ khóa, diễn viên, đạo diễn dưới dạng text
    5. Chỉ giữ các phim có trong ratings (để tránh metadata vô dụng)

    Đầu vào:
        df_ratings_sach: DataFrame ratings đã làm sạch (để biết phim nào có rating)
    """
    logger.info("\n" + "=" * 60)
    logger.info("LÀM SẠCH METADATA PHIM")
    logger.info("=" * 60)

    # 1. Đọc movies_metadata.csv

    logger.info("Đọc movies_metadata.csv")
    df_meta = pd.read_csv(config.MOVIES_METADATA_FILE, low_memory=False)
    logger.info(f"  Số dòng: {len(df_meta):,}")

    # Chuyển cột 'id' thành số nguyên (có thể chứa text lỗi)
    df_meta["tmdb_id"] = pd.to_numeric(df_meta["id"], errors="coerce")
    n_id_loi = df_meta["tmdb_id"].isnull().sum()
    logger.info(f"  Số dòng có id không hợp lệ (không phải số): {n_id_loi}")
    
    df_meta = df_meta.dropna(subset=["tmdb_id"])
    df_meta["tmdb_id"] = df_meta["tmdb_id"].astype("int64")
    
    # Loại bỏ tmdb_id trùng (giữ dòng đầu tiên)
    n_truoc_dup = len(df_meta)
    df_meta = df_meta.drop_duplicates(subset=["tmdb_id"])
    logger.info(f"  Loại trùng tmdb_id: {n_truoc_dup - len(df_meta)}")


    # 2. Đọc links.csv (ánh xạ movieId <-> tmdbId)

    logger.info("\nĐọc links.csv (bảng ánh xạ ID)")
    df_links = pd.read_csv(config.LINKS_FILE)
    df_links["tmdbId"] = pd.to_numeric(df_links["tmdbId"], errors="coerce")
    df_links = df_links.dropna(subset=["tmdbId"])
    df_links["tmdbId"] = df_links["tmdbId"].astype("int64")
    df_links["movieId"] = df_links["movieId"].astype("int32")
    logger.info(f"  Số phim trong links: {len(df_links):,}")

    # 3. Ghép metadata với links (movieId <-> tmdbId)

    logger.info("\nGhép metadata với links theo tmdbId")
    # links: movieId, tmdbId
    # meta: tmdb_id (= tmdbId trong links)
    df_merged = df_links.merge(
        df_meta,
        left_on="tmdbId",
        right_on="tmdb_id",
        how="inner"
    )
    logger.info(f"  Số phim sau ghép (links × metadata): {len(df_merged):,}")

    # 4. Chỉ giữ phim có trong ratings đã làm sạch
    movie_ids_co_rating = set(df_ratings_sach["movieId"].unique())
    df_merged = df_merged[df_merged["movieId"].isin(movie_ids_co_rating)]
    logger.info(f"  Số phim có rating: {len(df_merged):,}")
    logger.info(f"  Số phim trong ratings nhưng thiếu metadata: "
                f"{len(movie_ids_co_rating) - len(df_merged):,}")

  
    # 5. Đọc keywords.csv và ghép

    logger.info("\nĐọc và ghép keywords.csv")
    df_kw = pd.read_csv(config.KEYWORDS_FILE)
    df_kw["id"] = pd.to_numeric(df_kw["id"], errors="coerce").astype("Int64")
    
    df_merged = df_merged.merge(
        df_kw.rename(columns={"id": "tmdb_id", "keywords": "keywords_raw"}),
        on="tmdb_id",
        how="left"
    )


    # 6. Đọc credits.csv và trích xuất đạo diễn + diễn viên chính
    logger.info("Đọc và ghép credits.csv")
    df_credits = pd.read_csv(config.CREDITS_FILE)
    df_credits["id"] = pd.to_numeric(df_credits["id"], errors="coerce").astype("Int64")

    # Trích xuất tên đạo diễn
    def lay_dao_dien(crew_str):
        try:
            crew = ast.literal_eval(crew_str)
            directors = [p["name"] for p in crew 
                         if isinstance(p, dict) and p.get("job") == "Director"]
            return " ".join(directors[:2])  # Lấy tối đa 2 đạo diễn
        except:
            return ""

    # Trích xuất tên diễn viên chính (top 5)
    def lay_dien_vien_chinh(cast_str):
        try:
            cast = ast.literal_eval(cast_str)
            return " ".join([p["name"] for p in cast[:5] if isinstance(p, dict)])
        except:
            return ""

    logger.info("  Trích xuất thông tin đạo diễn và diễn viên")
    df_credits["dao_dien"] = df_credits["crew"].apply(lay_dao_dien)
    df_credits["dien_vien"] = df_credits["cast"].apply(lay_dien_vien_chinh)
    
    df_merged = df_merged.merge(
        df_credits[["id", "dao_dien", "dien_vien"]].rename(columns={"id": "tmdb_id"}),
        on="tmdb_id",
        how="left"
    )

    # 7. Xây dựng đặc trưng nội dung (content features) dạng text
    logger.info("\nXây dựng đặc trưng nội dung (content features)")

    def trich_xuat_genres(genres_str):
        """Trích xuất danh sách thể loại từ chuỗi JSON."""
        genres = trich_xuat_ast_list(genres_str, "name")
        return " ".join(genres)

    def trich_xuat_keywords(kw_str):
        """Trích xuất từ khóa, thay dấu cách bằng dấu gạch dưới."""
        if pd.isna(kw_str):
            return ""
        kws = trich_xuat_ast_list(kw_str, "name")
        # Thay dấu cách trong tên từ khóa bằng _ để TF-IDF xử lý là 1 token
        return " ".join([kw.replace(" ", "_") for kw in kws[:20]])

    # Xử lý từng cột
    df_merged["genres_text"] = df_merged.get("genres", pd.Series(dtype=str)).fillna("[]").apply(trich_xuat_genres)
    df_merged["keywords_text"] = df_merged.get("keywords_raw", pd.Series(dtype=str)).fillna("[]").apply(trich_xuat_keywords)
    df_merged["overview"] = df_merged.get("overview", pd.Series(dtype=str)).fillna("")
    df_merged["dao_dien"] = df_merged.get("dao_dien", pd.Series(dtype=str)).fillna("")
    df_merged["dien_vien"] = df_merged.get("dien_vien", pd.Series(dtype=str)).fillna("")

    # Ghép tất cả đặc trưng thành một chuỗi "content soup"
    # Thể loại và từ khóa được lặp lại 2 lần để tăng trọng số
    df_merged["content_soup"] = (
        df_merged["genres_text"] + " " + df_merged["genres_text"] + " " +
        df_merged["keywords_text"] + " " + df_merged["keywords_text"] + " " +
        df_merged["overview"] + " " +
        df_merged["dao_dien"] + " " +
        df_merged["dien_vien"]
    )
    df_merged["content_soup"] = df_merged["content_soup"].str.lower().str.strip()

    # 8. Chọn cột cần thiết và lưu
    cot_giu_lai = [
        "movieId", "tmdb_id", "tmdbId",
        "title", "original_title", "overview",
        "genres_text", "keywords_text",
        "dao_dien", "dien_vien",
        "content_soup",
        "vote_average", "vote_count",
        "release_date", "runtime", "popularity",
        "original_language",
    ]
    # Chỉ giữ cột tồn tại
    cot_giu_lai = [c for c in cot_giu_lai if c in df_merged.columns]
    df_final = df_merged[cot_giu_lai].copy()

    # Xử lý năm phát hành
    if "release_date" in df_final.columns:
        df_final["release_year"] = pd.to_datetime(
            df_final["release_date"], errors="coerce"
        ).dt.year.astype("Int64")

    # Chuẩn hóa kiểu dữ liệu
    df_final["movieId"] = df_final["movieId"].astype("int32")
    df_final["vote_average"] = pd.to_numeric(df_final.get("vote_average"), errors="coerce").astype("float32")
    df_final["vote_count"] = pd.to_numeric(df_final.get("vote_count"), errors="coerce").astype("Int64")

    # Lưu metadata phim đã làm sạch
    logger.info(f"\nLưu metadata phim vào: {config.MOVIES_CLEAN_FILE}")
    df_final.to_parquet(config.MOVIES_CLEAN_FILE, index=False)
    logger.info(f"  Số phim trong file: {len(df_final):,}")

    # 9. Tạo bảng ánh xạ ID
    logger.info(f"\nTạo bảng ánh xạ ID")
    
    # Tạo index nội bộ liên tục cho userId và movieId
    # (cần thiết để xây dựng ma trận sparse)
    user_ids = sorted(df_ratings_sach["movieId"].unique())  # Placeholder
    all_user_ids = sorted(df_ratings_sach["userId"].unique())
    all_movie_ids = sorted(df_ratings_sach["movieId"].unique())

    user_id_map = pd.DataFrame({
        "userId": all_user_ids,
        "user_idx": range(len(all_user_ids))
    }).astype({"userId": "int32", "user_idx": "int32"})

    movie_id_map = pd.DataFrame({
        "movieId": all_movie_ids,
        "movie_idx": range(len(all_movie_ids))
    }).astype({"movieId": "int32", "movie_idx": "int32"})

    # Ghép thêm tmdbId vào movie_id_map
    movie_id_map = movie_id_map.merge(
        df_links[["movieId", "tmdbId"]],
        on="movieId",
        how="left"
    )

    # Lưu bảng ánh xạ
    id_map = {
        "user_map": user_id_map,
        "movie_map": movie_id_map
    }
    
    user_id_map.to_parquet(
        os.path.join(config.DATA_PROCESSED_DIR, "user_id_map.parquet"), index=False
    )
    movie_id_map.to_parquet(
        os.path.join(config.DATA_PROCESSED_DIR, "movie_id_map.parquet"), index=False
    )
    logger.info(f"  Số người dùng: {len(user_id_map):,}")
    logger.info(f"  Số phim: {len(movie_id_map):,}")
    logger.info(f"  Đã lưu: user_id_map.parquet và movie_id_map.parquet")

    return df_final


def main():
    """Hàm chính - chạy toàn bộ tiền xử lý."""


    # Kiểm tra nếu đã xử lý rồi thì hỏi có muốn chạy lại không
    if os.path.exists(config.RATINGS_CLEAN_FILE) and os.path.exists(config.MOVIES_CLEAN_FILE):
        logger.info("\nPhát hiện dữ liệu đã xử lý trước đó:")
        logger.info(f"  {config.RATINGS_CLEAN_FILE}")
        logger.info(f"  {config.MOVIES_CLEAN_FILE}")
        logger.info("Để chạy lại, xóa các file này hoặc đặt FORCE_REPROCESS=True trong config.")
        
        # Kiểm tra biến môi trường để cho phép chạy lại
        if not os.environ.get("FORCE_REPROCESS"):
            logger.info("Bỏ qua tiền xử lý. Dùng dữ liệu đã có.")
            return

    # Bước 1: Làm sạch ratings
    import time
    t0 = time.time()
    df_ratings = lam_sach_ratings()
    logger.info(f"Thời gian làm sạch ratings: {(time.time()-t0)/60:.1f} phút")

    # Bước 2: Làm sạch metadata phim
    t0 = time.time()
    df_movies = lam_sach_movies_metadata(df_ratings)
    logger.info(f"Thời gian làm sạch metadata: {(time.time()-t0)/60:.1f} phút")

    # Tổng kết
    logger.info("\n" + "=" * 60)
    logger.info("TIỀN XỬ LÝ HOÀN TẤT")
    logger.info("=" * 60)
    logger.info(f"  Ratings đã làm sạch:   {len(df_ratings):,} dòng")
    logger.info(f"  Phim có metadata:       {len(df_movies):,} phim")
    logger.info(f"\nFile đã lưu:")
    logger.info(f"  - {config.RATINGS_CLEAN_FILE}")
    logger.info(f"  - {config.MOVIES_CLEAN_FILE}")
    logger.info(f"\n Bước tiếp theo: chạy src/03_eda.py")


if __name__ == "__main__":
    main()
