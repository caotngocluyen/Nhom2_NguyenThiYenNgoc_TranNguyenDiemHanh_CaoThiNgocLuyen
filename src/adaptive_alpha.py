import numpy as np
import pandas as pd


def compute_adaptive_alpha(interaction_count, min_alpha=0.25, max_alpha=0.88, threshold=40):
    """
    Tinh trong so alpha thich ung dua tren so luong tuong tac cua nguoi dung.
    
    Nguoi dung moi co it tuong tac se nhan alpha thap (uu tien Content-Based va Popularity).
    Nguoi dung ky cuu co nhieu tuong tac se nhan alpha cao (uu tien Collaborative Filtering).
    
    Tham so:
        interaction_count: So luot danh gia cua nguoi dung trong tap train.
        min_alpha: Gia tri alpha toi thieu danh cho nguoi dung moi.
        max_alpha: Gia tri alpha toi da danh cho nguoi dung tich cuc.
        threshold: Nguong so luot tuong tac de dat alpha toi da.
        
    Tra ve:
        alpha: Gia tri trong so nam trong khoang [min_alpha, max_alpha].
    """
    if interaction_count <= 1:
        return float(min_alpha)
    
    if interaction_count >= threshold:
        return float(max_alpha)
    
    # Ham tang truong logarit muot ma tu min_alpha den max_alpha
    ratio = np.log(1.0 + interaction_count) / np.log(1.0 + threshold)
    alpha = min_alpha + (max_alpha - min_alpha) * ratio
    return float(np.clip(alpha, min_alpha, max_alpha))


def build_user_interaction_map(df_train):
    """
    Dem so luot tuong tac cua tung nguoi dung tu tap du lieu huan luyen.
    
    Tham so:
        df_train: DataFrame chua cot userId.
        
    Tra ve:
        dict: Anh xa userId sang so luot tuong tac.
    """
    counts = df_train["userId"].value_counts()
    return counts.to_dict()


def build_user_adaptive_alpha_map(df_train, min_alpha=0.25, max_alpha=0.88, threshold=40):
    """
    Tao bang tra cuu alpha thich ung cho toan bo nguoi dung trong tap train.
    
    Tham so:
        df_train: DataFrame chua cot userId.
        min_alpha: Gia tri alpha toi thieu.
        max_alpha: Gia tri alpha toi da.
        threshold: Nguong tuong tac toi da.
        
    Tra ve:
        dict: Anh xa userId sang gia tri alpha ca nhan hoa.
    """
    interaction_map = build_user_interaction_map(df_train)
    alpha_map = {
        uid: compute_adaptive_alpha(count, min_alpha=min_alpha, max_alpha=max_alpha, threshold=threshold)
        for uid, count in interaction_map.items()
    }
    return alpha_map
