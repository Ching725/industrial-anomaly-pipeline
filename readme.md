# industrial-anomaly-pipeline
End-to-end anomaly detection pipeline for industrial data (SECOM and Steel Faults), featuring preprocessing, SMOTEENN, ensemble models, and SHAP explainability.


![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/github/license/Ching725/industrial-anomaly-pipeline)
![Status](https://img.shields.io/badge/status-WIP-yellow)
![Issues](https://img.shields.io/github/issues/Ching725/industrial-anomaly-pipeline)
![Last Commit](https://img.shields.io/github/last-commit/Ching725/industrial-anomaly-pipeline)

> A modular and scalable pipeline for industrial fault prediction using SECOM and Steel Faults datasets.  
> Supports preprocessing, model training, hybrid anomaly scoring, and SHAP-based interpretability.

---

## 🧱 Project Structure

```bash
industrial-anomaly-pipeline/
│
├── dataset/                # 原始資料（SECOM、Steel Faults）
├── notebooks/              # 各任務分析流程（EDA、模型比較、SHAP）
├── preprocessing/          # 前處理模組
├── models/                 # 模型訓練與參數調整模組
├── evaluation/             # 評估與異常分數計算模組
├── run_pipeline/           # Pipeline 執行腳本
├── utils/                  # 工具程式（儲存、記錄器、SHAP 工具）
├── artifacts/              # 輸出結果（模型、分數、報告）
└── README.md
```
---

## 🗃️ Datasets

| Dataset | Description |
|--------|-------------|
| **SECOM** | Semiconductor sensor data with 590+ features (imbalanced binary classification) |
| **Steel Plates Faults** | Steel manufacturing multi-class defect classification dataset |

---

## ⚙️ Features

- ✅ 多任務支援（SECOM / Steel Faults）
- 🔍 資料清洗 + 特徵工程模組化（遺失值處理、IQR/Z-Score 剪裁、標準化）
- ⚖️ 支援 SMOTEENN 重取樣平衡類別
- 📊 LightGBM、CatBoost、Voting Ensemble 模型訓練
- 🧠 SHAP 解釋模型特徵影響力
- 🧪 Isolation Forest + SHAP 整合異常分數（Hybrid Score）
- 📈 產出混淆矩陣圖、SHAP Summary Plot、分數報告

---

## 🚀 Quick Start

```bash
# 安裝必要套件
pip install -r requirements.txt

# 執行 SECOM 任務的 pipeline
python run_pipeline/run_pipeline_secom.py

# 執行 Steel Faults 任務的 pipeline
python run_pipeline/run_pipeline_steel.py
```

⸻

## 📘 Notebook 分析流程總覽（SECOM 與 Steel Faults）

---

### 🏭 Steel Faults 分析流程

- [`01_EDA_and_Preprocessing.ipynb`](notebooks/steel/01_EDA_and_Preprocessing.ipynb)  
  探索性資料分析（EDA）：檢視欄位分佈、相關性與類別不平衡問題。處理遺失值與異常值、執行標準化與特徵選擇，並使用 SMOTE 或 SMOTEENN 平衡資料。

- [`02_Traditional_ML_Models.ipynb`](notebooks/steel/02_Traditional_ML_Models.ipynb)  
  訓練傳統機器學習模型（隨機森林、XGBoost、LightGBM），並建立投票法（VotingClassifier）整合模型。評估指標包含分類報告、Macro-F1、ROC AUC 及混淆矩陣。

- [`03_Model_Interpretability_SHAP.ipynb`](notebooks/steel/03_Model_Interpretability_SHAP.ipynb)  
  使用 SHAP（Shapley 加法解釋法）視覺化模型的重要特徵，包括 summary plot、bar plot 與類別別解釋圖，幫助理解模型判斷依據。

- [`04_Enhance_Modeling_Strategy.ipynb`](notebooks/steel/04_Enhance_Modeling_Strategy.ipynb)  
  針對小樣本類別進行 Recall 提升策略，包含二階段分類（先判斷是否有瑕疵，再分類瑕疵種類）、不同類別平衡技術、自訂 loss function 與錯誤分析。


---

### 🔬 SECOM 製程異常預測流程

- [`01_data_exploration.ipynb`](notebooks/secom/01_data_exploration.ipynb)  
  載入 SECOM 資料與標籤，進行欄位分佈觀察、缺失值處理、標準化與特徵剪裁。

- [`02_Traditional_ML_Models.ipynb`](notebooks/secom/02_Traditional_ML_Models.ipynb)  
  建立多種分類模型（LogisticRegression、LightGBM、CatBoost），搭配 SMOTEENN 處理類別不平衡，依據 Recall 表現篩選模型。

- [`03_Enhance_lightgbm_pipeline.ipynb`](notebooks/secom/03_Enhance_lightgbm_pipeline.ipynb)  
  強化 LightGBM 建模效果：使用 class_weight、CalibratedClassifierCV 校正預測機率，加入 PCA 降維並優化 threshold。搭配 PR AUC / ROC AUC / F1 曲線視覺化。

- [`04_anomaly_scores.ipynb`](notebooks/secom/04_anomaly_scores.ipynb)  
  建立 AutoEncoder / IsolationForest 的異常分數，並整合 SHAP 分析與 PU Learning 建立 hybrid 分數與異常標記。

- [`05_realtime_simulation.ipynb`](notebooks/secom/05_realtime_simulation.ipynb)  
  模擬即時樣本流入與預測流程，顯示異常率變化趨勢圖，並即時顯示 SHAP 解釋與雙重異常標記結果。

- [`00_main_pipeline_overview.ipynb`](notebooks/secom/00_main_pipeline_overview.ipynb)  
  SECOM 任務的流程總覽筆記本，整合所有模組、模型、分數與異常偵測邏輯，作為總結入口。


⸻

## 📦 Output Example
- `artifacts/models_bundle/`：儲存最佳模型與前處理器（scaler/imputer）
- `artifacts/figures/`：儲存confusion matrix 


⸻

## 🧩 Supported Models
- ✅ Logistic Regression  
- ✅ LightGBM / CatBoost  
- ✅ Voting Ensemble  
- ✅ Isolation Forest  
- ✅ SHAP Explainer  

⸻

## 🗓️ Roadmap / TODO
- 整合 Streamlit dashboard（含 SHAP 圖與模型切換）  
- 將 pipeline 容器化  
- 新增 API 推論端點與 Swagger 文件  

⸻

👩‍💻 Author

Ching Yeh
AI Engineer / Data Scientist
📫 [LinkedIn](https://www.linkedin.com/in/chingyeh725) | 📝 [Medium Blog](https://medium.com/@amy2598877)

⸻
