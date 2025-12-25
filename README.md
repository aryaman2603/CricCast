# 🏏 CricCast: IPL Score Predictor

**End-to-End Machine Learning Pipeline | XGBoost | FastAPI | Docker | CI/CD**

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)
![Build](https://img.shields.io/badge/GitHub_Actions-Passing-brightgreen)

## 📖 Overview
**CricCast** is a production-grade Machine Learning system that predicts the final score of an IPL (Indian Premier League) cricket match in real-time. Unlike simple run-rate calculators, this model considers critical context: wickets lost, venue scoring history, and recent batting momentum.

The project demonstrates a **complete MLOps lifecycle**: from raw data ingestion and feature engineering to model serving via a containerized API with automated testing.

##  Architecture
The system consists of four key stages:
1.  **ETL Pipeline:** Parses 1,100+ JSON match files into a structured CSV dataset.
2.  **Feature Engineering:** Calculates advanced metrics like *rolling run-rate*, *wickets-in-hand*, and *venue scoring avg*.
3.  **Model Training:** Uses **XGBoost Regressor** optimized for regression tasks.
4.  **Deployment:** Serves predictions via a **FastAPI** backend, packaged in **Docker**, and verified via **GitHub Actions**.

## Key Features
* **Context-Aware:** Understands that `100/0` is vastly different from `100/5` at the same over.
* **More weight to recent matches:** Matches of the last 3 years have been given more weight as compared to older matches, to keep in mind the increase in scoring rates since the impact player rule. 
* **Venue Logic:** Adjusts predictions based on ground history (e.g., high-scoring Wankhede vs. low-scoring Chepauk).
* **Production Ready:** Fully Dockerized for "write once, run anywhere" deployment.
* **CI/CD Pipeline:** Automated build and test workflow ensures no broken code reaches production.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.13 |
| **ML Core** | Scikit-Learn, XGBoost, Pandas |
| **API Framework** | FastAPI, Uvicorn, Pydantic |
| **Containerization** | Docker (Linux/Debian) |
| **CI/CD** | GitHub Actions |
| **Versioning** | Git, DVC (Data Version Control) |

---

## ⚡ Quick Start (Run with Docker)
The easiest way to run the application is using the pre-built Docker container.

**1. Clone the repository**
```bash
git clone [https://github.com/aryaman2603/criccast.git](https://github.com/aryaman2603/criccast.git)
cd criccast

**2. Build the image**
```bash
docker build -t criccast:v1

**3. Run the container**
```bash
docker run -p 8000:8000 criccast:v1

