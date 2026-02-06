# Experiment 2 – Binary Classification using Naïve Bayes, KNN, and SVM

This repository contains **Experiment 2** from the *Machine Learning Algorithms Laboratory*.  
The experiment focuses on implementing and evaluating **Naïve Bayes**, **K-Nearest Neighbors (KNN)**, and **Support Vector Machine (SVM)** classifiers for a **binary email spam classification problem**.

---

## 📌 Experiment Details

- **Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai  
- **Affiliation:** Anna University  
- **Degree & Branch:** B.E. Computer Science & Engineering  
- **Semester:** VI  
- **Subject Code & Name:** UCS2612 – Machine Learning Algorithms Laboratory  
- **Academic Year:** 2025–2026 (Even Semester)  
- **Batch:** 2023–2027  

---

## 🎯 Aim

To classify emails as **Spam** or **Ham** using:
- Naïve Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

and evaluate their performance using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC–AUC
- K-Fold Cross-Validation

---

## 🧰 Libraries Used

- **Pandas** – Data manipulation  
- **NumPy** – Numerical operations  
- **Scikit-learn** – Model building, preprocessing, evaluation  
- **Matplotlib** – Visualization  
- **Seaborn** – Statistical visualization  

---

## 📂 Dataset Used

- **Spambase Dataset**
- Binary classification:
  - `0` → Ham (Not Spam)
  - `1` → Spam

---

## 🤖 Machine Learning Models Used

- **Naïve Bayes**
  - GaussianNB
  - MultinomialNB
  - BernoulliNB
- **K-Nearest Neighbors (KNN)**
  - k = 1, 3, 5, 7
  - KDTree and BallTree
- **Support Vector Machine (SVM)**
  - Linear kernel
  - Polynomial kernel
  - RBF kernel
  - Sigmoid kernel

---

## 🧪 Experiment Workflow

### 1️⃣ Data Loading
- Load dataset using Pandas
- Check for missing values
- Separate features and labels

### 2️⃣ Data Preprocessing
- Feature normalization using `StandardScaler`
- Train–test split with stratification

### 3️⃣ Exploratory Data Analysis
- Class distribution bar chart
- Feature distribution histograms

### 4️⃣ Model Training
- Train Naïve Bayes variants
- Train KNN with different `k` values
- Compare KDTree vs BallTree
- Train SVM with multiple kernels

### 5️⃣ Model Evaluation
- Classification report
- Confusion matrix
- ROC curve and AUC score
- Training time comparison

### 6️⃣ K-Fold Cross-Validation
- 5-Fold cross-validation
- Compare average accuracy across models

---

## 📊 Performance Metrics Used

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC–AUC  
- Training Time  

---

## 📈 Output Visualizations

- Class distribution bar chart  
- Feature distribution histograms  
- Confusion matrices for all models  
- ROC curves with AUC values  
- KNN tree comparison plots  
- SVM kernel-wise performance table  
- 5-Fold cross-validation results  

---

## 🔍 Observations

### ✅ Best Classifier
- **SVM (Linear Kernel)** achieved the **highest average accuracy: 0.9274**

### ✅ Best Naïve Bayes Variant
- **Bernoulli Naïve Bayes**
  - Accuracy: 0.8863
  - Highest AUC among NB variants

### ✅ KNN Performance
- Accuracy improved as `k` increased
- Best results at `k = 7`
- KDTree and BallTree gave similar accuracy
- BallTree trained slightly faster

### ✅ Best SVM Kernel
- **Linear kernel** performed best overall
- RBF kernel was a close second
- Polynomial kernel performed poorly

### ✅ Hyperparameter Influence
- KNN accuracy highly dependent on `k`
- SVM performance strongly dependent on kernel choice

---

## 🧠 Learning Outcomes

From this experiment, we learned:

- Practical implementation of Naïve Bayes, KNN, and SVM
- Importance of feature scaling and preprocessing
- Effect of hyperparameters on model performance
- Use of evaluation metrics beyond accuracy
- Visualization using confusion matrices and ROC curves
- Importance of K-fold cross-validation
- Comparative analysis of multiple classifiers

---
