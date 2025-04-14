# 🏥 Healthcare Cost Prediction Using Machine Learning

This end-to-end machine learning project predicts hospital charges for patients using real-world healthcare data from New York State. The goal is to improve cost transparency by giving patients and providers an intelligent, data-driven estimate of inpatient healthcare expenses.

---

## 📌 Objective

Hospital bills can be unpredictable, especially in the U.S. healthcare system. By analyzing patient attributes such as length of stay, diagnosis category, and severity of illness, this project builds predictive models that estimate total hospital charges.

This project walks through the **entire data science pipeline** — from raw data preprocessing to model building, evaluation, and user deployment through a Shiny web app.

---

## 🔍 Key Highlights

- ✅ Processed over **2.3 million patient discharge records** from New York State (2017)
- ✅ Built a modular pipeline with **cleaned data, encoded variables, and feature selection**
- ✅ Trained and compared **five regression models** for cost prediction
- ✅ Deployed a **Shiny for Python app** that allows real-time user interaction and predictions
- ✅ Analyzed **feature importance** to understand what drives healthcare cost

---

## 📁 Repository Structure

```bash
📦 Healthcare-Cost-Prediction-ML/
├── DataProcessing/         # EDA, preprocessing, and train-test split notebooks
├── Models/                 # All trained models (XGBoost, RF, LGBM, NN, DecisionTree)
├── ShinyApp/               # Deployed interface with model files and mappings
├── Project_Report/         # Final written report
└── README.md               # You are here
```

---

## 📊 Models Built & Evaluated

The following regression models were trained and tested:

| Model           | R² Score | RMSE  |
|----------------|----------|-------|
| XGBoost        | 0.8499   | 0.39  |
| Neural Network | 0.8436   | 0.40  |
| LightGBM       | 0.8173   | 0.43  |
| Random Forest  | 0.7637   | 0.49  |
| Decision Tree  | 0.6257   | 0.61  |

- **Best Model:** XGBoost
- **Top Feature:** Length of Stay

---

## ⚙️ Technologies Used

- **Languages & Tools:** Python, Jupyter Notebook
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, Matplotlib, Seaborn
- **Interface:** [Shiny for Python](https://shiny.posit.co/py/) for real-time prediction app
- **Deployment Components:** JSON mappings, Joblib model persistence

---

## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Healthcare-Cost-Prediction-ML.git
cd Healthcare-Cost-Prediction-ML
```

### 2. Set Up the Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, manually install:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn shiny
```

### 3. Run Preprocessing

Navigate to the `DataProcessing/` folder and run these notebooks:
- `ExploratoryDataAnalysis.ipynb`
- `Preprocessing.ipynb`
- `Split.ipynb`

These notebooks:
- Clean and prepare the raw dataset
- Encode features and apply dimensionality reduction
- Split the data into train/test sets

### 4. Train Models

Go to the `Models/` folder and run any of the model training notebooks. For example:
- `XGBoost.ipynb`
- `NeuralNetwork.ipynb`

You can explore model comparisons and select the best based on R² or RMSE.

### 5. Launch the Shiny App

Navigate to the `ShinyApp/` folder and run `ShinyApp.ipynb`.

In the last cell:
- It will output a link (`your url is:`) to launch the app in your browser
- Paste the generated password from the previous cell when prompted

This app allows users to input patient characteristics and receive a **live cost estimate**.

---

## 📁 Dataset

The dataset used is publicly available and contains inpatient discharge summaries from New York State for 2017.

🔗 [Download Dataset Here](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/y93g-8xk5)

⚠️ **Note:** Due to file size, the raw dataset is not included in this repository.

---

## 👨‍💻 Author

**Faizan Bhutto**  
M.S. in Data Science  
DePaul University  
📫 [LinkedIn](https://www.linkedin.com/in/faizanbhutto) | [Portfolio](https://faizanbhutto.com/)

---

## 📜 License

This project is open-source and available for educational or non-commercial use.
