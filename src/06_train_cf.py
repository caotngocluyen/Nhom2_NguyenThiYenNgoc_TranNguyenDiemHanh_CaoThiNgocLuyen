import os
import sys
import logging
import warnings
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from surprise import SVD, Dataset, Reader

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log_file = os.path.join(config.REPORTS_DIR, "06_cf.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def prepare_surprise_dataset(df_train, df_val):
    """Chuan bi tap du lieu Surprise voi Reader."""
    reader = Reader(rating_scale=(config.MIN_RATING, config.MAX_RATING))
    
    # Su dung kieu so nguyen int de toi uu bo nho ma tran
    train_data = df_train[["userId", "movieId", "rating"]].copy()
    train_data["userId"] = train_data["userId"].astype(int)
    train_data["movieId"] = train_data["movieId"].astype(int)

    data = Dataset.load_from_df(train_data, reader)
    trainset = data.build_full_trainset()
    return trainset, reader


def train_svd_model(trainset, n_factors=50, n_epochs=15, lr_all=0.005, reg_all=0.02):
    """Huan luyen mo hinh SVD Matrix Factorization."""
    logger.info("Khoi tao va huan luyen SVD (factors=%d, epochs=%d, lr=%.4f, reg=%.4f)",
                n_factors, n_epochs, lr_all, reg_all)
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=config.RANDOM_SEED,
        verbose=True
    )
    t0 = time.time()
    model.fit(trainset)
    elapsed = time.time() - t0
    logger.info("Thoi gian huan luyen SVD: %.1f giay", elapsed)
    return model, elapsed


def evaluate_on_val(model, df_val, sample_size=10000):
    """Danh gia sai so tren tap validation."""
    n_eval = min(sample_size, len(df_val))
    df_eval = df_val.sample(n_eval, random_state=config.RANDOM_SEED)

    y_true = []
    y_pred = []
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Danh gia val"):
        uid = int(row["userId"])
        iid = int(row["movieId"])
        pred = model.predict(uid, iid)
        y_true.append(float(row["rating"]))
        y_pred.append(float(pred.est))

    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    logger.info("Ket qua SVD tren tap Validation: RMSE = %.4f, MAE = %.4f", rmse, mae)
    return rmse, mae


def main():
    train_path = config.TRAIN_IMPROVED_FILE if os.path.exists(config.TRAIN_IMPROVED_FILE) else config.TRAIN_FILE
    val_path = config.VAL_IMPROVED_FILE if os.path.exists(config.VAL_IMPROVED_FILE) else config.VAL_FILE

    logger.info("Doc tap train tu %s va tap val tu %s", train_path, val_path)
    df_train = pd.read_parquet(train_path, columns=["userId", "movieId", "rating"])
    df_val = pd.read_parquet(val_path, columns=["userId", "movieId", "rating"])

    trainset, reader = prepare_surprise_dataset(df_train, df_val)
    model, train_time = train_svd_model(
        trainset,
        n_factors=50,
        n_epochs=15,
        lr_all=0.005,
        reg_all=0.02
    )

    rmse_val, mae_val = evaluate_on_val(model, df_val)

    # Toi uu dung luong trainset: thay ur va ir bang tap hop chi so de giam dung luong ma van du bao duoc
    model.trainset.ur = set(range(model.trainset.n_users))
    model.trainset.ir = set(range(model.trainset.n_items))

    # Luu mo hinh vao models/svd_improved.pkl
    joblib.dump(model, config.SVD_IMPROVED_MODEL)
    logger.info("Da luu SVD model cai tien tai: %s", config.SVD_IMPROVED_MODEL)

    metrics = {
        "model": "SVD_CF",
        "rmse_val": rmse_val,
        "mae_val": mae_val,
        "train_time_s": train_time,
        "n_factors": 50,
        "n_epochs": 15,
        "lr_all": 0.005,
        "reg_all": 0.02
    }
    pd.DataFrame([metrics]).to_parquet(
        os.path.join(config.OUTPUTS_DIR, "cf_metrics.parquet"),
        index=False
    )

    report_path = os.path.join(config.REPORTS_DIR, "cf_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SVD COLLABORATIVE FILTERING RESULTS\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info("Da luu ket qua vao: %s", report_path)


if __name__ == "__main__":
    main()
