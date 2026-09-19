# Zepto Data & AI Platform — Capstone Project

**Certificate Program in Artificial Intelligence and Machine Learning**  
*End-to-End Enterprise Data Engineering, Predictive Analytics & Grounded GenAI Platform*

---

## Executive Summary

The **Zepto Data & AI Platform** is an enterprise-grade, single-repository multi-capability platform built for Zepto's analytics and machine learning guild. It connects three core engineering capabilities:

1. **Data Engineering Pipeline (`/data_pipeline`) — 25 Marks**: Scrapes live catalog data from `books.toscrape.com`, cleans and enriches product records, applies the project-baseline fixed currency conversion rate ($1\text{ GBP} = 105.50\text{ INR}$), stores data into a normalized 2-table SQLite relational database, and executes analytical SQL queries verified against in-memory pandas operations.
2. **Analytics & Machine Learning Pipeline (`/analytics`) — 50 Marks**: Profiles and models customer-style data end-to-end on the Titanic dataset with a committed offline fallback (`titanic.csv`), defensive missing-value handling, outlier detection, bivariate & multivariate data storytelling, stratified train-test splitting, leakage-free `ColumnTransformer` preprocessing, classifier evaluation (Logistic Regression, Decision Tree, Random Forest), class imbalance mitigation (SMOTE vs `class_weight`), `GridSearchCV` hyperparameter tuning with Out-of-Bag (OOB) scoring, multivariate fare regression with heteroscedasticity analysis, and a bundled `joblib` pipeline export.
3. **Grounded GenAI Support Assistant (`/support_assistant`) — 25 Marks**: A multi-stage RAG assistant that embeds Zepto's official policy documents (`all-MiniLM-L6-v2`), orchestrates query intent routing via a LangGraph state machine, retrieves grounded context from a local vector store, enforces structured JSON responses via Pydantic (`answer`, `sources`, `confidence`), serves queries over a FastAPI `POST /ask` endpoint, and packages the service into a locally buildable `Dockerfile`.

---

## Repository Structure

```
zepto_data_ai_platform/
│
├── .gitignore
├── requirements.txt                   # Master consolidated requirements
├── README.md                          # Master Capstone documentation (this file)
│
├── data_pipeline/                     # MODULE 1: Data Engineering Pipeline (25 Marks)
│   ├── scraper.py                     # BeautifulSoup catalog scraper (144 books across 4 categories)
│   ├── pipeline.py                    # Cleaning, currency conversion, SQLite loading, SQL & Pandas queries
│   ├── data_pipeline.ipynb            # Step-by-step interactive notebook walkthrough
│   ├── books.db                       # Normalized SQLite relational database
│   ├── requirements.txt               # Module-specific requirements
│   └── README.md                      # Detailed data engineering report
│
├── analytics/                         # MODULE 2: Analytics & Predictive Modeling (50 Marks)
│   ├── 01_eda.ipynb                   # Part A: Profiling, cleaning, IQR outliers, multivariate data story
│   ├── 02_modeling.ipynb              # Part B: Modeling, imbalance comparison, GridSearchCV, fare regression
│   ├── run_analytics.py               # Standalone end-to-end analytics & ML script
│   ├── titanic.csv                    # Committed offline fallback dataset
│   ├── best_model_pipeline.joblib     # Serialized complete scikit-learn pipeline (Preprocessor + Model)
│   ├── plots/                         # Generated high-resolution visualization artifacts
│   │   ├── univariate_age_fare.png
│   │   ├── correlation_matrix.png
│   │   ├── multivariate_data_story.png
│   │   ├── decision_tree.png
│   │   ├── roc_curves.png
│   │   └── fare_residuals.png
│   ├── requirements.txt               # Module-specific requirements
│   └── README.md                      # Comprehensive analysis report & deployment recommendation
│
└── support_assistant/                 # MODULE 3: Grounded GenAI Support Assistant (25 Marks)
    ├── docs/                          # Official Zepto policy corpus (doc_01.txt to doc_08.txt)
    │   ├── doc_01.txt                 # Delivery Policy
    │   ├── doc_02.txt                 # Returns & Refunds
    │   ├── doc_03.txt                 # Membership Tiers (Basic, Pass, Pass+)
    │   ├── doc_04.txt                 # Live Order Tracking
    │   ├── doc_05.txt                 # Order Cancellation Policy
    │   ├── doc_06.txt                 # Damaged or Missing Items
    │   ├── doc_07.txt                 # Gift Cards Policy
    │   └── doc_08.txt                 # Customer Support Hours
    ├── config.py                      # Configurations and MOCK_LLM toggle
    ├── ingestion.py                   # Ingestion, all-MiniLM-L6-v2 embeddings, and VectorStore
    ├── prompt.py                      # Structured skeleton prompt with negative constraints & few-shots
    ├── graph.py                       # LangGraph StateGraph (classify_intent, retrieve_and_answer, direct_answer)
    ├── app.py                         # FastAPI web application serving POST /ask
    ├── test_assistant.py              # Test suite verifying endpoints and intent routing
    ├── Dockerfile                     # Containerization recipe for FastAPI microservice
    ├── chroma_db/                     # Persistent vector index
    ├── requirements.txt               # Module-specific requirements
    └── README.md                      # RAG architecture description & API transcripts
```

---

## Installation & Setup

### Option A: Consolidated Installation (Recommended)
You can install all dependencies across the entire repository using the root `requirements.txt`:

```bash
# 1. Clone the repository
git clone <repository_url>
cd zepto_data_ai_platform

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Option B: Module-Level Installation
Each module contains its own dedicated `requirements.txt` if you prefer modular installations:
- Data Pipeline: `pip install -r data_pipeline/requirements.txt`
- Analytics: `pip install -r analytics/requirements.txt`
- Support Assistant: `pip install -r support_assistant/requirements.txt`

---

## How to Run Each Module End to End

### 1. Running Module 1: Data Pipeline
```bash
# Execute standalone pipeline (scrapes, cleans, creates books.db, runs queries, asserts merge equivalence):
python data_pipeline/pipeline.py

# Or launch Jupyter and run data_pipeline/data_pipeline.ipynb:
jupyter notebook data_pipeline/data_pipeline.ipynb
```

### 2. Running Module 2: Analytics & ML Pipeline
```bash
# Execute full analytics and modeling pipeline (profiles data, produces plots, trains models, exports joblib):
python analytics/run_analytics.py

# Or run the ordered Jupyter notebooks:
# Step 1: Part A EDA & Visual Story
jupyter notebook analytics/01_eda.ipynb
# Step 2: Part B Predictive Modeling & Evaluation
jupyter notebook analytics/02_modeling.ipynb
```

### 3. Running Module 3: GenAI Support Assistant
```bash
# Run local verification test suite:
python support_assistant/test_assistant.py

# Launch the FastAPI server locally:
python support_assistant/app.py
# The API will be live at: http://localhost:7860
# Interactive Swagger Documentation: http://localhost:7860/docs

# Build and Run using Docker:
docker build -t zepto-support-assistant -f support_assistant/Dockerfile support_assistant/
docker run -p 7860:7860 zepto-support-assistant
```

---

## Module-by-Module Design Decisions & Technical Highlights

### Module 1: Data Pipeline
- **Fixed-Rate Currency Conversion**: Currency is converted from GBP to INR using the required constant:
  $$\mathbf{1\text{ GBP} = 105.50\text{ INR}}$$
- **Scraping Scope**: Scraped 144 books across 4 distinct categories (`Travel`, `Mystery`, `Historical Fiction`, `Sequential Art`), surpassing the required threshold ($\ge 60$ books, $\ge 3$ categories).
- **Data Cleaning & Defensive Imputation**: Cleaned price strings with regex, converted text ratings (`One`..`Five`) to integers (1–5), parsed availability to boolean (`in_stock`), and implemented median price imputation for unparsable numeric fields to ensure robust execution.
- **Relational Schema (3NF)**:
  - `categories(category_id INTEGER PRIMARY KEY AUTOINCREMENT, category_name TEXT UNIQUE NOT NULL)`
  - `books(book_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, price_gbp REAL NOT NULL, price_inr REAL NOT NULL, rating INTEGER NOT NULL, in_stock INTEGER NOT NULL, category_id INTEGER NOT NULL REFERENCES categories(category_id))`
- **SQL & Pandas Equivalence**: Executed 5 SQL queries covering `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, and a relational `JOIN`. Replicated the SQL join and aggregations via `pd.merge()` on in-memory DataFrames, confirming 100% equivalence via `pd.testing.assert_frame_equal`.

### Module 2: Analytics & Predictive Modeling
- **Offline Fallback Guarantee**: The dataset was fetched once via `sns.load_dataset('titanic')` and immediately saved as `analytics/titanic.csv`. All downstream EDA, modeling, and grading steps load directly from this committed CSV without requiring network access.
- **Rule-Based Missing Handling**:
  - `embarked`: 0.22% missing ($<5\%$) $\rightarrow$ dropped missing rows.
  - `age`: 19.87% missing ($5\%-30\%$) $\rightarrow$ imputed with median ($28.0$).
  - `deck`: 77.22% missing ($>70\%$) $\rightarrow$ dropped column entirely to avoid synthetic bias.
- **Outliers & Skewness**:
  - `age`: 65 IQR outliers ($7.31\%$).
  - `fare`: 114 IQR outliers ($12.82\%$).
  - Skewness: $\text{Mean } (\$32.10) > \text{Median } (\$14.45) > \text{Mode } (\$8.05) \implies$ **Strongly Right-Skewed**.
- **Bivariate & Multivariate Story**:
  - Female survival was $74.04\%$ vs Male $18.89\%$. 1st Class was $62.62\%$ vs 3rd Class $24.24\%$.
  - 6x6 correlation matrix on exact numeric features (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`), highlighting strong negative correlation between `pclass` and `fare` ($r = -0.548$) and family co-travel between `sibsp` and `parch` ($r = +0.415$).
  - 4 multivariate charts illustrate the intersection of gender, social standing, age, and family size on survival probability.
- **Model Evaluation & Imbalance Mitigation**:
  - Stratified 80/20 train-test split (`stratify=y`) preserving the 61.6% to 38.4% target class balance.
  - Leakage-free `ColumnTransformer` fit **strictly on training data**.
  - Evaluated Logistic Regression, Decision Tree (`plot_tree` rendered), and Random Forest.
  - Compared baseline vs `class_weight='balanced'` vs `SMOTE` (train-fold only), demonstrating substantial recall gains ($0.6087 \rightarrow 0.7246$).
  - Tuned Random Forest via `GridSearchCV` with `RandomForestClassifier(oob_score=True, ...)` reporting best parameters (`max_depth: 4`, `max_features: 'sqrt'`, `n_estimators: 150`) and an Out-of-Bag (OOB) score of **0.8301**.
  - Multivariate linear regression on `fare` ($\text{MAE}=\$18.87, \text{RMSE}=\$30.92, R^2=0.3822, \text{Adj } R^2=0.3531$) with residual analysis proving **heteroscedasticity**.
  - Exported complete end-to-end pipeline to `analytics/best_model_pipeline.joblib` and verified inference reload on raw data.

#### Master Model Comparison Table

| Model Type | Model / Estimator | Accuracy | Precision | Recall | F1 Score | AUC-ROC | MAE ($) | RMSE ($) | R² | Adj R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classification** | Logistic Regression | **0.7989** | 0.7797 | **0.6667** | **0.7188** | **0.8436** | *N/A* | *N/A* | *N/A* | *N/A* |
| **Classification** | Decision Tree (depth=4) | 0.7933 | 0.8333 | 0.5797 | 0.6838 | 0.8281 | *N/A* | *N/A* | *N/A* | *N/A* |
| **Classification** | Random Forest (Tuned) | **0.7989** | **0.8667** | 0.5652 | 0.6842 | 0.8404 | *N/A* | *N/A* | *N/A* | *N/A* |
| **Regression** | Multivariate Linear Reg (Fare) | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | **18.87** | **30.92** | **0.3822** | **0.3531** |

> **Final Deployment Recommendation:**
> **Logistic Regression** is recommended for standard operational deployment due to its optimal F1 score (**0.7188**), highest AUC-ROC (**0.8436**), balanced recall (**0.6667**), and rapid inference latency with direct explainability. In high-stakes environments where false positive survivor predictions must be strictly minimized, the **Tuned Random Forest** is the preferred alternative with a top precision of **0.8667** and an OOB score of **0.8301**.

---

### Module 3: Support Assistant
- **Deterministic Offline Mock Baseline (`MOCK_LLM=1` — Default)**:
  - Runs 100% offline without API keys, accounts, or paid tiers.
  - Keyword intent router accurately routes policy queries containing terms like `delivery`, `refund`, `return`, `cancel`, `membership`, `tracking`, etc. to `policy_question` and unrelated queries to `general_question`.
  - Executes real vector retrieval against the 8 indexed policy documents in `ChromaDB` / `LocalVectorStore` and formats grounded canned responses with source document IDs (`doc_01`..`doc_08`).
- **Structured Prompt Design (`prompt.py`)**:
  - Implements the strict **role–context–task–format–length** skeleton.
  - Contains an explicit negative constraint prohibiting external hallucination.
  - Embeds positive policy few-shot and out-of-scope refusal few-shot examples.
- **LangGraph State Machine (`graph.py`)**:
  - TypedDict State (`query`, `intent`, `retrieved_docs`, `answer`, `sources`, `confidence`).
  - 3 Nodes: `classify_intent`, `retrieve_and_answer`, `direct_answer`.
  - Conditional edge router directing graph execution.
- **Pydantic Validation & FastAPI Microservice (`app.py`)**:
  - Enforces `AgentResponse` schema (`answer: str`, `sources: List[str]`, `confidence: float`).
  - Serves `POST /ask` with input validation, CORS middleware, and health check endpoints.
  - Fully containerized with a production-ready `Dockerfile`.

---

## Git Workflow History

This repository strictly followed professional branch and merge workflow standards:
1. Initial scaffolding committed on `main`.
2. Dedicated feature branch created: `feature/zepto-platform-pipeline`.
3. Sequentially developed and committed all 3 modules (`data_pipeline`, `analytics`, `support_assistant`).
4. Merged back into `main` with a formal non-fast-forward merge commit (`df14bef`).

```
*   df14bef (HEAD -> main) Merge branch 'feature/zepto-platform-pipeline' into main
|\  
| * e44309c (feature/zepto-platform-pipeline) feat(support_assistant): Implement RAG ingestion, LangGraph intent router, FastAPI app, Pydantic validation, tests, and Dockerfile
| * 76b6f33 feat(analytics): Complete EDA, visualizations, predictive modeling, regression side-task, and joblib pipeline export
| * 550d460 feat(data_pipeline): Implement scraper, SQLite normalization, SQL analytics, and pandas merge verification
|/  
* ae05239 Initial commit: Project scaffolding, requirements, and policy corpus
```

---

## Academic Integrity & Compliance Statement

- **100% Free & Open-Source**: No paid APIs or subscriptions are required to run, test, or evaluate any component of this platform.
- **Full Textual Submission**: All deliverables, code files (`.py`, `.ipynb`), datasets, databases, serializations, and comprehensive markdown documentations are self-contained within this single repository.
