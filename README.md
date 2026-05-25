# Eksperimen_SML_Arraffi

Repositori eksperimen Machine Learning untuk Proyek Akhir kelas "Membangun Sistem Machine Learning" - Dicoding.

## Author
**Ar'raffi Abqori Nur Azizi**

## Dataset
**Wine Quality** — UCI Machine Learning Repository
- Red Wine: 1599 samples
- White Wine: 4898 samples
- Task: Binary Classification (Good/Bad Wine)

## Struktur Repository
```
Eksperimen_SML_Arraffi/
├── .github/workflows/
│   └── preprocessing.yml        # GitHub Actions untuk automated preprocessing
├── winequality_raw/
│   ├── winequality-red.csv      # Raw red wine data
│   └── winequality-white.csv    # Raw white wine data
└── preprocessing/
    ├── Eksperimen_Arraffi.ipynb  # Notebook eksperimen (EDA + Preprocessing)
    ├── automate_Arraffi.py      # Script otomasi preprocessing
    └── winequality_preprocessing/
        ├── train.csv            # Data latih (80%)
        ├── test.csv             # Data uji (20%)
        └── encoding_map.json   # Mapping encoding kategorikal
```

## Tahapan Eksperimen
1. **Data Loading** — Memuat dataset Wine Quality (red + white)
2. **EDA** — Exploratory Data Analysis (distribusi, korelasi, outlier)
3. **Data Cleaning** — Hapus duplikat, handle missing values
4. **Outlier Handling** — IQR clipping
5. **Feature Engineering** — Binary target, fitur baru
6. **Encoding** — LabelEncoder untuk wine_type
7. **Train-Test Split** — 80/20 dengan stratifikasi
8. **Scaling** — StandardScaler

## Cara Menjalankan
```bash
# Jalankan preprocessing otomatis
cd preprocessing
python automate_Arraffi.py
```

## GitHub Actions
Workflow otomatis akan menjalankan preprocessing setiap kali ada push atau pull request ke branch main.
