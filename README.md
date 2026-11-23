# 📈 Stock Price Prediction Using Machine Learning

## 🎓 University AI Course Lab Final Project

This project was developed as the **Final Project** for our **University Artificial Intelligence (AI) Course Lab**.
The project demonstrates the practical implementation of machine learning techniques for predicting stock prices using historical market data.

The main goal of this project is to apply AI concepts in a real-world financial prediction system while gaining hands-on experience in model training, data visualization, and web application deployment.

---



## 📌 Project Overview

This repository contains a machine learning-based stock price prediction system designed to forecast S&P 500 closing prices for up to **30 business days**.

The Streamlit application lets users select from 503 S&P 500 symbols, review recent price history, and generate forecasts using a Random Forest model trained on historical OHLCV data.

The project includes:

* 📊 Data preprocessing and cleaning
* 🤖 Random Forest model training with lag and rolling-average features
* 📈 Stock trend visualization
* 🌐 Interactive Streamlit web application
* 🔮 Future stock price prediction

---

## ✨ Features

* Predict stock prices for the next 30 days
* Visualize historical stock price trends
* Interactive and responsive Streamlit UI
* Real-time prediction visualization
* Machine learning-based forecasting system
* Easy-to-use web interface

---

## 🛠️ Technologies Used

The project utilizes the following technologies and libraries:

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Plotly**

---

## ⚙️ Setup and Installation

To run this project locally, follow the steps below:

### 1️⃣ Clone the Repository

```bash id="9rkqgb"
git clone https://github.com/MdFarhadHossenImon/Stock-Price-Prediction-500-Comp.git

cd Stock-Price-Prediction-500-Comp
```

---

### 2️⃣ Create a Virtual Environment

```bash id="b6ikf8"
python -m venv env
```

Activate the environment:

#### Windows

```bash id="f5gxyo"
env\Scripts\activate
```

#### Linux / Mac

```bash id="0ydzxh"
source env/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash id="r5wk94"
pip install -r requirements.txt
```

---

### 4️⃣ Download the Dataset

Install [Git LFS](https://git-lfs.com/), then run:

```bash
git lfs install
git lfs pull
```

---

### 5️⃣ Run the Streamlit Application

```bash id="z7j62o"
streamlit run main.py
```

---

## 📂 Project Structure

```plaintext id="2ghgk4"
Stock-Price-Prediction-500-Comp/
│
├── new dataset stock/
│   ├── sp500_stocks.csv     # Daily OHLCV history (tracked with Git LFS)
│   └── sp500_companies.csv # Company names and metadata
├── model.py                 # Machine learning model training script
├── main.py                  # Streamlit web application
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── .gitattributes           # Git LFS tracking rules
└── .gitignore               # Ignored local files
```

---

## 📊 Dataset

The application uses `new dataset stock/sp500_stocks.csv`, a daily OHLCV dataset covering 2000 through 2026 for 503 S&P 500 symbols. The approximately 290 MB stock file is stored with Git LFS; `sp500_companies.csv` provides company names for the selector.

The app loads the selected symbol's price history, creates lag and rolling-average features, trains a Random Forest regressor, and reports validation mean absolute error. Forecasts are generated recursively for business days.

---

## 🤖 Machine Learning Workflow

The project follows these major steps:

1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Model Training using a Random Forest regressor
5. Prediction Generation
6. Data Visualization
7. Deployment with Streamlit

---

## 🔮 Future Scope

Future improvements that can be added to the project:

* 📡 Integration with live stock market APIs
* 📊 Advanced chart visualizations
* 📈 Multiple stock comparison system
* 🤖 More accurate AI prediction models
* ☁️ Cloud deployment support
* 📱 Mobile responsive optimization

---

## 👨‍💻 Developer Information

### Md Farhad Hossen Imon

🔗 LinkedIn: https://www.linkedin.com/in/md-farhad-hossen-imon-403a25259/
💻 GitHub: https://github.com/MdFarhadHossenImon

### Md. Iftaker Hossain

🔗 LinkedIn: https://www.linkedin.com/in/ifhhamim/
💻 GitHub: https://github.com/ifhhamim


## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub and sharing your feedback.