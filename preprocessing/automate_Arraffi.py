"""
automate_Arraffi.py
Automated Preprocessing Script for Wine Quality Dataset
Author: Ar'raffi Abqori Nur Azizi

Script ini mengotomatisasi seluruh pipeline preprocessing yang telah
dilakukan pada notebook eksperimen (Eksperimen_Arraffi.ipynb).

Tahapan:
1. Data Loading - Memuat dataset Wine Quality (red & white)
2. Data Cleaning - Menghapus duplikat dan menangani missing values
3. Feature Engineering - Membuat fitur baru dan binary target
4. Encoding & Scaling - Label encoding dan StandardScaler
5. Train-Test Split - Membagi data menjadi train dan test set
6. Save Output - Menyimpan dataset yang sudah dipreprocess
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import json

warnings.filterwarnings("ignore")


def load_data(red_path=None, white_path=None):
    """
    Memuat dataset Wine Quality (red dan white wine).
    Jika path tidak diberikan, akan mengunduh dari UCI ML Repository.
    
    Parameters:
    -----------
    red_path : str, optional
        Path ke file CSV red wine
    white_path : str, optional  
        Path ke file CSV white wine
        
    Returns:
    --------
    pd.DataFrame : Dataset gabungan red dan white wine
    """
    print("=" * 60)
    print("TAHAP 1: DATA LOADING")
    print("=" * 60)
    
    if red_path and os.path.exists(red_path):
        df_red = pd.read_csv(red_path, sep=";")
    else:
        url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df_red = pd.read_csv(url_red, sep=";")
    
    if white_path and os.path.exists(white_path):
        df_white = pd.read_csv(white_path, sep=";")
    else:
        url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        df_white = pd.read_csv(url_white, sep=";")
    
    # Tambahkan kolom wine_type
    df_red["wine_type"] = "red"
    df_white["wine_type"] = "white"
    
    # Gabungkan kedua dataset
    df = pd.concat([df_red, df_white], axis=0, ignore_index=True)
    
    print(f"  Red wine samples  : {len(df_red)}")
    print(f"  White wine samples: {len(df_white)}")
    print(f"  Total samples     : {len(df)}")
    print(f"  Features          : {list(df.columns)}")
    print()
    
    return df


def clean_data(df):
    """
    Membersihkan data: menangani missing values dan duplikat.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset mentah
        
    Returns:
    --------
    pd.DataFrame : Dataset yang sudah dibersihkan
    """
    print("=" * 60)
    print("TAHAP 2: DATA CLEANING")
    print("=" * 60)
    
    initial_shape = df.shape
    
    # Cek missing values
    missing = df.isnull().sum()
    missing_total = missing.sum()
    print(f"  Missing values: {missing_total}")
    
    if missing_total > 0:
        # Isi missing values numerik dengan median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"    Filled {col} with median: {median_val}")
    
    # Hapus duplikat
    duplicates = df.duplicated().sum()
    print(f"  Duplicates found: {duplicates}")
    
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  Removed {duplicates} duplicate rows")
    
    print(f"  Shape before: {initial_shape} -> after: {df.shape}")
    print()
    
    return df


def handle_outliers(df, columns=None, method="iqr", threshold=1.5):
    """
    Mendeteksi dan menangani outlier menggunakan metode IQR.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    columns : list, optional
        Kolom numerik yang akan diproses
    method : str
        Metode deteksi outlier ('iqr')
    threshold : float
        Threshold untuk IQR method
        
    Returns:
    --------
    pd.DataFrame : Dataset dengan outlier yang sudah ditangani
    """
    print("=" * 60)
    print("TAHAP 3: HANDLING OUTLIERS")
    print("=" * 60)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target column
        if "quality" in columns:
            columns.remove("quality")
    
    initial_len = len(df)
    outlier_info = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_info[col] = outliers
        
        # Clip outliers instead of removing (to preserve data)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    total_clipped = sum(outlier_info.values())
    print(f"  Total outlier values clipped: {total_clipped}")
    for col, count in outlier_info.items():
        if count > 0:
            print(f"    {col}: {count} outliers clipped")
    print()
    
    return df


def feature_engineering(df):
    """
    Membuat fitur baru dan mengonversi target menjadi binary classification.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset yang sudah dibersihkan
        
    Returns:
    --------
    pd.DataFrame : Dataset dengan fitur baru
    """
    print("=" * 60)
    print("TAHAP 4: FEATURE ENGINEERING")
    print("=" * 60)
    
    # 1. Binary target: quality >= 7 = "good" (1), else "bad" (0)
    df["quality_label"] = (df["quality"] >= 7).astype(int)
    print(f"  Binary target distribution:")
    print(f"    Bad wine (0) : {(df['quality_label'] == 0).sum()}")
    print(f"    Good wine (1): {(df['quality_label'] == 1).sum()}")
    
    # 2. Membuat fitur baru
    # Total acidity
    df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]
    print(f"  Created feature: total_acidity")
    
    # Ratio sulfur dioxide
    df["free_sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
    print(f"  Created feature: free_sulfur_ratio")
    
    # Alcohol to density ratio
    df["alcohol_density_ratio"] = df["alcohol"] / (df["density"] + 1e-6)
    print(f"  Created feature: alcohol_density_ratio")
    
    print(f"  Total features: {len(df.columns)}")
    print()
    
    return df


def encode_features(df):
    """
    Melakukan encoding pada fitur kategorikal.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
        
    Returns:
    --------
    pd.DataFrame : Dataset dengan fitur yang sudah di-encode
    dict : Mapping encoding untuk referensi
    """
    print("=" * 60)
    print("TAHAP 5: ENCODING")
    print("=" * 60)
    
    encoding_map = {}
    
    # Encode wine_type (red=0, white=1)
    le = LabelEncoder()
    df["wine_type_encoded"] = le.fit_transform(df["wine_type"])
    encoding_map["wine_type"] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  wine_type encoding: {encoding_map['wine_type']}")
    
    # Drop kolom original yang tidak diperlukan
    df = df.drop(columns=["wine_type", "quality"])
    print(f"  Dropped columns: wine_type, quality")
    print(f"  Remaining columns: {list(df.columns)}")
    print()
    
    return df, encoding_map


def scale_features(X_train, X_test, feature_names):
    """
    Melakukan standardisasi pada fitur numerik.
    
    Parameters:
    -----------
    X_train : np.array
        Training features
    X_test : np.array
        Testing features
    feature_names : list
        Nama fitur
        
    Returns:
    --------
    pd.DataFrame, pd.DataFrame, StandardScaler : Scaled train, test, dan scaler
    """
    print("=" * 60)
    print("TAHAP 6: SCALING")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    print(f"  Scaler fitted on {len(feature_names)} features")
    print(f"  Train shape: {X_train_scaled.shape}")
    print(f"  Test shape : {X_test_scaled.shape}")
    print()
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(df, target_col="quality_label", test_size=0.2, random_state=42):
    """
    Membagi data menjadi train dan test set.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset yang sudah dipreprocess
    target_col : str
        Nama kolom target
    test_size : float
        Proporsi test set
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple : X_train, X_test, y_train, y_test, feature_names
    """
    print("=" * 60)
    print("TAHAP 7: TRAIN-TEST SPLIT")
    print("=" * 60)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Total samples: {len(df)}")
    print(f"  Train samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"  Test samples : {len(X_test)} ({test_size*100:.0f}%)")
    print(f"  Train target distribution: {dict(y_train.value_counts())}")
    print(f"  Test target distribution : {dict(y_test.value_counts())}")
    print()
    
    return X_train, X_test, y_train, y_test, feature_names


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir, encoding_map=None):
    """
    Menyimpan dataset yang sudah dipreprocess.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Scaled features
    y_train, y_test : pd.Series
        Target variables
    output_dir : str
        Direktori output
    encoding_map : dict, optional
        Mapping encoding
    """
    print("=" * 60)
    print("TAHAP 8: SAVING PREPROCESSED DATA")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Gabungkan X dan y untuk penyimpanan
    train_df = X_train.copy()
    train_df["quality_label"] = y_train.values
    
    test_df = X_test.copy()
    test_df["quality_label"] = y_test.values
    
    # Simpan sebagai CSV
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  Saved train data: {train_path} ({train_df.shape})")
    print(f"  Saved test data : {test_path} ({test_df.shape})")
    
    # Simpan encoding map
    if encoding_map:
        encoding_path = os.path.join(output_dir, "encoding_map.json")
        # Convert numpy types to Python types for JSON serialization
        encoding_map_serializable = {}
        for key, value in encoding_map.items():
            if isinstance(value, dict):
                encoding_map_serializable[key] = {
                    str(k): int(v) for k, v in value.items()
                }
            else:
                encoding_map_serializable[key] = value
        
        with open(encoding_path, "w") as f:
            json.dump(encoding_map_serializable, f, indent=2)
        print(f"  Saved encoding map: {encoding_path}")
    
    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


def run_preprocessing(raw_data_dir=None, output_dir=None):
    """
    Menjalankan seluruh pipeline preprocessing secara otomatis.
    
    Parameters:
    -----------
    raw_data_dir : str, optional
        Direktori raw data
    output_dir : str, optional
        Direktori output untuk data yang sudah dipreprocess
        
    Returns:
    --------
    tuple : X_train, X_test, y_train, y_test (data siap dilatih)
    """
    # Set default directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if raw_data_dir is None:
        raw_data_dir = os.path.join(os.path.dirname(script_dir), "winequality_raw")
    
    if output_dir is None:
        output_dir = os.path.join(script_dir, "winequality_preprocessing")
    
    red_path = os.path.join(raw_data_dir, "winequality-red.csv")
    white_path = os.path.join(raw_data_dir, "winequality-white.csv")
    
    print("\n" + "=" * 60)
    print("AUTOMATED PREPROCESSING PIPELINE")
    print(f"Author: Ar'raffi Abqori Nur Azizi")
    print(f"Dataset: Wine Quality (UCI ML Repository)")
    print("=" * 60 + "\n")
    
    # Pipeline
    df = load_data(red_path, white_path)
    df = clean_data(df)
    df = handle_outliers(df)
    df = feature_engineering(df)
    df, encoding_map = encode_features(df)
    X_train, X_test, y_train, y_test, feature_names = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_names)
    save_preprocessed_data(X_train_scaled, X_test_scaled, y_train, y_test, output_dir, encoding_map)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_preprocessing()
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")
