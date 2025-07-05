# ❤️ AuxiliaryHeart: AI-Powered Heart Disease Detection from ECG

> **“What if you had an app that could analyze your ECG and warn you before real danger happens?”**

---

## 🚨 Problem Statement

Bangladesh faces a **critical heart health crisis**:

- 🩺 **Doctor-to-population ratio**: 1 doctor per 1,847 people.
- ❤️ **Heart diseases cause over 21%** of total deaths in Bangladesh.
- 💔 **Coronary Heart Disease** alone caused **108,528 deaths** (15.16% of total) as of WHO 2020 report.

---

## 💡 Our Solution

**An AI-based application** that:

- 🧠 Takes your **ECG input**
- ⚠️ **Detects signs of Coronary Artery Disease** and **Heart Attack**
- 📊 Provides accurate health analysis
- 🧾 Explains results using **LLM (Large Language Model)**

---

## 🧪 Datasets Used

### 📁 PTB-XL ECG Dataset:
- 5469 Myocardial Infarction
- 9514 Normal Patients
- 5235 ST/T Change
- 4898 Conduction Disturbance
- 2649 Hypertrophy

### 📁 GU-ECG Database:
- 74 Coronary Artery Disease ECGs (MI-prone)

---

## 🔬 Key ECG Features

### 🧠 ST Segment & T Wave
- **ST Level**: Baseline elevation/depression
- **ST Slope**: Angle indicating ischemia
- **T Area**: Area under T wave
- **T Extremum**: Peak T value
- **T Width & Symmetry**: Duration & shape for tissue damage detection

### ❤️ P Wave
- **P Duration & Amplitude**
- **P-R Segment Level**

### ⚡ QRS Complex
- **QRS Duration**: Ventricular depolarization time
- **QRS Area & Slope**
- **R Peak Amplitude**

### ⏱️ Intervals
- **RR Interval**: Time between R peaks
- **PR & QT Intervals**: Conduction and recovery indicators

---

## 🧮 Data Types

- 📈 **Continuous Numeric**: All wave-based features (T, P, QRS, ST, intervals)
- 🔢 **Binary**: Gender, heart disease presence

---

## ⚙️ Project Pipeline

1️⃣ Data Collection
2️⃣ Preprocessing
3️⃣ Data Split
4️⃣ Model Selection
5️⃣ Training
6️⃣ Evaluation
7️⃣ Deployment
8️⃣ Documentation


---

## 🤖 Machine Learning Model

- 🧠 Supervised Learning  
- 📈 Support Vector Machine (SVM)  
- 🩺 Binary classification (disease/no disease)

---

## 🧱 Software Architecture

flowchart LR
    A[Frontend (Flutter)] --> B[Backend (Dart + Python)]
    B --> C[ECG Feature Extraction]
    C --> D[ML Model (SVM)]
    D --> E[LLM Model]
    E --> F[User Explanation & Feedback]
🧠 Research Inspiration
📄 Detection of Acute Coronary Syndrome using ECG, ICECCE 2020 — DOI:10.1109/ICECCE49384.2020.9179337

📄 CAD Classification using 1D CNN, arXiv 2024 — arXiv:2406.16895

🌟 Why Our App is Unique?
⚡ Edge Computing for faster processing

🤖 Explainable AI via LLMs

🫀 Helps users understand their heart health visually & textually

👨‍💻 Contributors
MD. Saiful Islam — 2132105642

Ishtiak Ahmed Moyen — 2131580642

S.M. Karimul — 22126886542

Raiyan Ahmed — 2221931042
