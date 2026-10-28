"""
09_tune_models.py - Tối ưu tham số (Hyperparameter Tuning)

MỤC ĐÍCH:
    Tìm bộ tham số tốt nhất cho các mô hình bằng Grid Search trên tập validation.

    
    HYPERPARAMETER TUNING LÀ GÌ?
    
    Hyperparameter (siêu tham số) là tham số mà ta phải chọn TRƯỚC khi huấn luyện,
    ví dụ: số latent factors, learning rate, regularization
    
    Grid Search: Thử tất cả tổ hợp tham số trong một "lưới" định nghĩa trước.
    Random Search: Thử ngẫu nhiên một số tổ hợp (nhanh hơn Grid Search).
    
    Quy tắc quan trọng:
    - Chọn tham số tốt nhất DỰA TRÊN VALIDATION SET
    - Không được dùng TEST SET để chọn tham số (tránh data leakage)
    - Test set chỉ dùng 1 lần duy nhất để báo cáo kết quả cuối

    Chiến lược tuning trong file này:
    1. Tuning SVD (n_factors, n_epochs, lr_all, reg_all) - Giới hạn để phù hợp thời gian
    2. Tuning alpha của Hybrid model
    3. Lưu cấu hình tốt nhất và kết quả mỗi lần thử

ĐẦU VÀO:
    - data/processed/train.parquet
    - data/processed/val.parquet
    - models/tfidf_vectorizer.pkl, tfidf_matrix.npz (đã có)

ĐẦU RA:
    - models/best_svd_model.pkl
    - models/best_hybrid_model.pkl
    - outputs/tuning_results.parquet
    - reports/tuning_report.txt

CÁCH CHẠY:
    python src/09_tune_models.py
    
    Lưu ý: Quá trình này mất nhiều thời gian (30-90 phút tùy tham số).
    Kết quả được lưu sau mỗi lần thử để có thể dừng và tiếp tục.
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
import joblib
import time
import json
from itertools import product
from tqdm import tqdm

from surprise import SVD, Dataset, Reader
from train_content_module import ContentBasedRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.REPORTS_DIR, "09_tuning.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# File lưu kết quả tuning (để có thể tiếp tục nếu bị dừng giữa chừng)
TUNING_RESULTS_PATH = os.path.join(config.OUTPUTS_DIR, "tuning_results.parquet")


def tinh_rmse(y_true, y_pred):
    """Tính RMSE nhanh."""
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def tai_ket_qua_truoc():
    """Tải kết quả tuning đã có từ lần chạy trước (nếu có)."""
    if os.path.exists(TUNING_RESULTS_PATH):
        df = pd.read_parquet(TUNING_RESULTS_PATH)
        logger.info(f"Tải {len(df)} kết quả tuning trước đó.")
        return df.to_dict("records")
    return []


def luu_ket_qua(ket_qua_list):
    """Lưu kết quả tuning ngay sau mỗi lần thử."""
    df = pd.DataFrame(ket_qua_list)
    df.to_parquet(TUNING_RESULTS_PATH, index=False)


def chuan_bi_du_lieu_surprise(df_train):
    """Chuẩn bị dữ liệu train cho surprise."""
    reader  = Reader(rating_scale=(config.MIN_RATING, config.MAX_RATING))
    data    = Dataset.load_from_df(df_train[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    return trainset


def danh_gia_svd_tren_val(svd_model, df_val, n_sample=10000):
    """
    Đánh giá SVD trên một mẫu của val set.
    
    Dùng mẫu để tiết kiệm thời gian đánh giá trong quá trình tuning.
    Kết quả cuối cùng sẽ đánh giá trên toàn bộ val trong bước 10.
    """
    np.random.seed(config.RANDOM_SEED)
    if len(df_val) > n_sample:
        df_sample = df_val.sample(n_sample, random_state=config.RANDOM_SEED)
    else:
        df_sample = df_val

    preds = [
        svd_model.predict(str(row["userId"]), str(row["movieId"])).est
        for _, row in df_sample.iterrows()
    ]
    return tinh_rmse(df_sample["rating"].values, preds)


def tune_svd(trainset, df_val, ket_qua_truoc):
    """
    Grid Search cho SVD trên validation set.
    
    Do thời gian hạn chế, ta dùng lưới nhỏ:
    - n_factors: [50, 100, 150] -> ảnh hưởng lớn nhất
    - n_epochs:  [15, 20]
    - lr_all:    [0.003, 0.005]
    - reg_all:   [0.02, 0.05]
    
    Tổng cộng: 3 × 2 × 2 × 2 = 24 tổ hợp
    Mỗi tổ hợp mất ~10-20 phút -> Toàn bộ: 4-8 giờ
    
    -> Dùng Random Search chọn 8 tổ hợp tốt nhất để có thể chạy trong 1-2 giờ.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TUNING SVD - Random Search")
    logger.info("=" * 60)

    # Định nghĩa không gian tham số
    param_grid = config.SVD_PARAM_GRID
    
    # Tạo tất cả tổ hợp tham số
    all_combinations = [
        dict(zip(param_grid.keys(), combo))
        for combo in product(*param_grid.values())
    ]
    
    # Random Search: chỉ thử một số tổ hợp
    np.random.seed(config.RANDOM_SEED)
    n_trials = min(3, len(all_combinations))  # 3 lần thử có kiểm soát để hoàn thành trong 15-20 phút
    selected = np.random.choice(len(all_combinations), n_trials, replace=False)
    trial_params = [all_combinations[i] for i in selected]

    logger.info(f"  Tổng tổ hợp có thể: {len(all_combinations)}")
    logger.info(f"  Số lần thử (Random Search): {n_trials}")

    # Kiểm tra tổ hợp đã thử trước đó
    da_thu = {
        json.dumps({k: v for k, v in r.items() if k in param_grid}, sort_keys=True)
        for r in ket_qua_truoc
        if r.get("model_type") == "SVD"
    }

    svd_ket_qua = [r for r in ket_qua_truoc if r.get("model_type") == "SVD"]
    best_rmse = min((r["rmse_val"] for r in svd_ket_qua), default=float("inf"))
    best_params = {}
    best_model = None

    for trial_idx, params in enumerate(trial_params, 1):
        params_key = json.dumps(params, sort_keys=True)
        
        if params_key in da_thu:
            logger.info(f"  [Trial {trial_idx}/{n_trials}] Bỏ qua (đã thử): {params}")
            continue

        logger.info(f"\n  [Trial {trial_idx}/{n_trials}] Thử: {params}")
        t0 = time.time()

        # Huấn luyện SVD với tham số này
        mo_hinh = SVD(
            n_factors=params["n_factors"],
            n_epochs=params["n_epochs"],
            lr_all=params["lr_all"],
            reg_all=params["reg_all"],
            random_state=config.RANDOM_SEED,
            verbose=False
        )
        mo_hinh.fit(trainset)
        thoi_gian = time.time() - t0

        # Đánh giá trên val mẫu
        rmse = danh_gia_svd_tren_val(mo_hinh, df_val)

        result = {
            "model_type": "SVD",
            "n_factors": params["n_factors"],
            "n_epochs":  params["n_epochs"],
            "lr_all":    params["lr_all"],
            "reg_all":   params["reg_all"],
            "rmse_val":  rmse,
            "train_time_s": thoi_gian,
        }
        svd_ket_qua.append(result)
        ket_qua_truoc.append(result)
        luu_ket_qua(ket_qua_truoc)  # Lưu ngay sau mỗi lần thử

        logger.info(f"    RMSE: {rmse:.4f} | Thời gian: {thoi_gian:.0f}s")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = mo_hinh
            logger.info(f"     Tham số tốt nhất mới! RMSE = {best_rmse:.4f}")

    # Tổng kết SVD tuning
    logger.info(f"\nKết quả SVD Tuning")
    svd_df = pd.DataFrame(svd_ket_qua).sort_values("rmse_val")
    logger.info(f"\n{svd_df[['n_factors', 'n_epochs', 'lr_all', 'reg_all', 'rmse_val']].to_string()}")
    logger.info(f"\nTham số SVD tốt nhất: {best_params}")
    logger.info(f"RMSE tốt nhất: {best_rmse:.4f}")

    return best_model, best_params, best_rmse


def tune_hybrid_alpha(svd_model, cb_model, df_train, df_val):
    """
    Tuning trọng số alpha cho Hybrid model.
    
    alpha điều chỉnh tỷ lệ đóng góp của CF và Content-Based:
    - alpha = 0.0: Hoàn toàn Content-Based
    - alpha = 0.5: Cân bằng
    - alpha = 1.0: Hoàn toàn CF (SVD)
    
    Ta thử các giá trị alpha từ 0.1 đến 0.9 và chọn alpha cho RMSE nhỏ nhất.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TUNING HYBRID ALPHA")
    logger.info("=" * 60)

    # Lấy mẫu val để đánh giá nhanh
    n_sample = min(3000, len(df_val))
    df_val_sample = df_val.sample(n_sample, random_state=config.RANDOM_SEED)
    y_true = df_val_sample["rating"].values.astype(np.float32)

    # Xây dựng user histories chỉ cho các user có trong mẫu val (tối ưu tốc độ & RAM)
    logger.info("Xây dựng user histories cho val sample")
    val_sample_users = set(df_val_sample["userId"])
    df_train_sub = df_train[df_train["userId"].isin(val_sample_users)]
    user_histories = df_train_sub.groupby("userId").apply(
        lambda x: dict(zip(x["movieId"], x["rating"]))
    ).to_dict()
    global_mean = float(df_train["rating"].mean())
    logger.info(f"  Số user có history trong val sample: {len(user_histories):,}")

    # Tính trước dự đoán CF và CB cho mẫu val (để không phải tính lại cho mỗi alpha)
    logger.info("Tính dự đoán CF và CB cho mẫu val")
    cf_preds = []
    cb_preds = []

    for _, row in tqdm(df_val_sample.iterrows(), total=len(df_val_sample), desc="Pre-compute"):
        uid, mid = row["userId"], row["movieId"]
        
        # CF prediction
        try:
            cf_p = svd_model.predict(str(uid), str(mid)).est
        except:
            cf_p = global_mean
        cf_preds.append(cf_p)
        
        # CB prediction
        history = user_histories.get(uid, {})
        cb_p = cb_model.predict_rating(history, mid)
        cb_preds.append(cb_p)

    cf_preds = np.array(cf_preds, dtype=np.float32)
    cb_preds = np.array(cb_preds, dtype=np.float32)

    # Cả CF và CB đều dự đoán trên thang điểm rating [0.5, 5.0]
    # Kết hợp tuyến tính trực tiếp theo trọng số alpha
    alpha_results = []
    logger.info(f"\nThử {len(config.HYBRID_ALPHA_RANGE)} giá trị alpha")

    for alpha in config.HYBRID_ALPHA_RANGE:
        hybrid_preds = alpha * cf_preds + (1.0 - alpha) * cb_preds
        hybrid_preds = np.clip(hybrid_preds, config.MIN_RATING, config.MAX_RATING)
        
        rmse = tinh_rmse(y_true, hybrid_preds)
        mae  = float(np.mean(np.abs(y_true - hybrid_preds)))
        
        alpha_results.append({
            "model_type": "Hybrid_Alpha",
            "alpha": alpha,
            "rmse_val": rmse,
            "mae_val": mae,
        })
        logger.info(f"  alpha={alpha:.1f}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Tìm alpha tốt nhất
    best_result = min(alpha_results, key=lambda x: x["rmse_val"])
    best_alpha = best_result["alpha"]
    best_rmse  = best_result["rmse_val"]

    logger.info(f"\nAlpha tốt nhất: {best_alpha} (RMSE={best_rmse:.4f})")
    logger.info(f"  So sánh:")
    logger.info(f"    Thuần CF (alpha=1.0):  RMSE={tinh_rmse(y_true, cf_preds):.4f}")
    logger.info(f"    Thuần CB (alpha=0.0):  RMSE={tinh_rmse(y_true, cb_preds):.4f}")
    logger.info(f"    Best Hybrid:           RMSE={best_rmse:.4f}")

    return best_alpha, alpha_results


def main():
    """Hàm chính."""
    logger.info("""
    Quy tắc tuning:
    1. Chọn tham số DỰA TRÊN VALIDATION SET (không dùng test)
    2. Lưu kết quả sau mỗi lần thử để tránh mất công khi máy tắt
    3. Sau khi có tham số tốt nhất -> huấn luyện lại trên toàn bộ train
    4. Chỉ đánh giá trên test 1 lần duy nhất trong bước 10
    """)

    # Tải dữ liệu
    df_train = pd.read_parquet(config.TRAIN_FILE)
    df_val   = pd.read_parquet(config.VAL_FILE)
    logger.info(f"Train: {len(df_train):,} | Val: {len(df_val):,}")

    # Tải kết quả tuning đã có
    ket_qua_truoc = tai_ket_qua_truoc()

    
    # Phần 1: Tuning SVD
    
    trainset = chuan_bi_du_lieu_surprise(df_train)
    best_svd, best_svd_params, best_svd_rmse = tune_svd(trainset, df_val, ket_qua_truoc)

    if best_svd is not None:
        # Lưu model SVD tốt nhất
        svd_path = os.path.join(config.MODELS_DIR, "best_svd_model.pkl")
        joblib.dump(best_svd, svd_path)
        logger.info(f"Best SVD model lưu tại: {svd_path}")
    else:
        # Load model mặc định nếu đã có
        svd_path = os.path.join(config.MODELS_DIR, "svd_model.pkl")
        if os.path.exists(svd_path):
            best_svd = joblib.load(svd_path)
            logger.info(f"Dùng SVD model mặc định: {svd_path}")
        else:
            logger.error("Không có SVD model! Chạy 06_train_cf.py trước.")
            sys.exit(1)

    
    # Phần 2: Tuning Hybrid alpha
    
    # Tải Content-Based model
    try:
        cb_model = ContentBasedRecommender.load(config.MODELS_DIR)
        logger.info("\nContent-Based model loaded.")
    except Exception as e:
        logger.error(f"Không load được CB model: {e}")
        logger.error("Chạy 07_train_content.py trước!")
        sys.exit(1)

    best_alpha, alpha_results = tune_hybrid_alpha(best_svd, cb_model, df_train, df_val)
    ket_qua_truoc.extend(alpha_results)
    luu_ket_qua(ket_qua_truoc)

    
    # Lưu cấu hình tốt nhất
    
    best_config = {
        "svd_params": best_svd_params,
        "svd_rmse_val": best_svd_rmse,
        "hybrid_alpha": best_alpha,
    }

    config_path = os.path.join(config.MODELS_DIR, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"\nCấu hình tốt nhất lưu tại: {config_path}")

    
    # Báo cáo tuning
    
    df_results = pd.read_parquet(TUNING_RESULTS_PATH)
    report_path = os.path.join(config.REPORTS_DIR, "tuning_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("HYPERPARAMETER TUNING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(" SVD TUNING \n")
        svd_df = df_results[df_results["model_type"] == "SVD"].sort_values("rmse_val")
        f.write(svd_df.to_string(index=False) + "\n\n")
        
        f.write(" HYBRID ALPHA TUNING \n")
        alpha_df = df_results[df_results["model_type"] == "Hybrid_Alpha"].sort_values("alpha")
        if len(alpha_df) > 0:
            f.write(alpha_df[["alpha", "rmse_val", "mae_val"]].to_string(index=False) + "\n\n")
        
        f.write(f" BEST CONFIGURATION \n")
        f.write(f"SVD: {best_svd_params}\n")
        f.write(f"SVD RMSE val: {best_svd_rmse:.4f}\n")
        f.write(f"Hybrid alpha: {best_alpha}\n")

    logger.info(f"Báo cáo tuning: {report_path}")
    logger.info("\nTuning hoàn tất.")
    logger.info("Bước tiếp theo: chạy src/10_evaluate.py")


if __name__ == "__main__":
    main()
