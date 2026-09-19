# Module 1 — Data Pipeline (/data_pipeline)

## Overview & Architecture
This module implements a production-grade data engineering pipeline that scrapes live catalog data from `books.toscrape.com`, cleans and enriches the records, normalizes the data into a relational SQLite database schema, and runs analytical SQL queries verified against in-memory pandas operations.

```
books.toscrape.com (HTML)
          │
          ▼
   [BeautifulSoup & Requests]
          │
          ▼
Raw Book Dictionaries (title, price, star_rating, availability, category)
          │
          ▼
   [Cleaning & Enrichment]
   ├── Price extraction & float casting
   ├── Fixed currency conversion (1 GBP = 105.50 INR)
   ├── Rating parsing (text -> int 1..5)
   ├── Availability parsing (text -> boolean)
   └── Median-imputation for unexpected missing values
          │
          ▼
   [Normalized SQLite DB: books.db]
   ├── categories (category_id PK, category_name UNIQUE)
   └── books (book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)
          │
          ▼
   [SQL Analytics & Pandas Verification]
   ├── 5 distinct SQL queries (WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, JOIN)
   └── pd.read_sql vs pd.merge equivalence assertion
```

---

## Key Design Decisions & Constants

1. **Fixed Baseline Currency Conversion**:
   - As required by the project specifications, currency conversion uses the fixed project-defined baseline:
     $$\mathbf{1\text{ GBP} = 105.50\text{ INR}}$$
   - This constant is defined directly in `pipeline.py` and requires no external network lookup or API key.

2. **Data Scraping Scope**:
   - Scraped 4 rich categories: **Travel**, **Mystery**, **Historical Fiction**, and **Sequential Art**.
   - Total records captured: **144 books** (exceeding the $\ge 60$ books requirement).

3. **Data Cleaning & Missing Value Strategy**:
   - `price_gbp`: Regular expression extraction `(\d+\.?\d*)` to strip currency symbols and non-numeric artifacts.
   - `price_inr`: Computed as `round(price_gbp * 105.50, 2)`.
   - `rating`: Text mapping (`One` $\rightarrow 1$, `Two` $\rightarrow 2$, `Three` $\rightarrow 3$, `Four` $\rightarrow 4$, `Five` $\rightarrow 5$).
   - `in_stock`: Boolean check (`True` if "in stock" is present in availability string).
   - **Imputation Strategy**: If any price field fails to parse, it is imputed with the **median price** of the catalog to prevent pipeline failure on noisy data while preserving distributional stability.

4. **Normalized Relational Schema**:
   - Two-table design enforcing 3NF with Primary Key / Foreign Key constraints:
     - `categories`: `(category_id INTEGER PRIMARY KEY AUTOINCREMENT, category_name TEXT UNIQUE NOT NULL)`
     - `books`: `(book_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, price_gbp REAL NOT NULL, price_inr REAL NOT NULL, rating INTEGER NOT NULL, in_stock INTEGER NOT NULL, category_id INTEGER NOT NULL, FOREIGN KEY(category_id) REFERENCES categories(category_id))`

---

## Executed SQL Queries & Results

### Query 1: `SELECT`, `WHERE`, `ORDER BY` — Top Rated Books in Stock
```sql
SELECT title, price_gbp, price_inr, rating
FROM books
WHERE rating >= 4 AND in_stock = 1
ORDER BY rating DESC, price_gbp ASC;
```
*Output (First 5 Rows):*
| title | price_gbp | price_inr | rating |
| :--- | :--- | :--- | :--- |
| Fruits Basket, Vol. 2 (Fruits Basket #2) | £11.64 | ₹1228.02 | 5 |
| Superman Vol. 1: Before Truth (Superman by Gene Luen Yang #1) | £11.89 | ₹1254.40 | 5 |
| The Girl You Lost | £12.29 | ₹1296.59 | 5 |
| Princess Jellyfish 2-in-1 Omnibus, Vol. 01 | £13.61 | ₹1435.86 | 5 |
| Roller Girl | £14.10 | ₹1487.55 | 5 |

### Query 2: `ORDER BY`, `LIMIT` — Top 5 Most Expensive Books
```sql
SELECT title, price_gbp, price_inr, rating
FROM books
ORDER BY price_inr DESC
LIMIT 5;
```
*Output:*
| title | price_gbp | price_inr | rating |
| :--- | :--- | :--- | :--- |
| Boar Island (Anna Pigeon #19) | £59.48 | ₹6275.14 | 3 |
| The No. 1 Ladies' Detective Agency (No. 1 Ladies' Detective Agency #1) | £57.70 | ₹6087.35 | 4 |
| El Deafo | £57.62 | ₹6078.91 | 5 |
| Ajin: Demi-Human, Volume 1 (Ajin: Demi-Human #1) | £57.06 | ₹6019.83 | 4 |
| A Year in Provence (Provence #1) | £56.88 | ₹6000.84 | 4 |

### Query 3: `DISTINCT` — Unique Star Ratings in Catalog
```sql
SELECT DISTINCT rating
FROM books
ORDER BY rating ASC;
```
*Output:* `[1, 2, 3, 4, 5]`

### Query 4: `BETWEEN` — Mid-Range Budget Books (INR 2,000 to INR 4,000)
```sql
SELECT title, price_inr, rating
FROM books
WHERE price_inr BETWEEN 2000.0 AND 4000.0
ORDER BY price_inr ASC;
```
*Total Matching Rows:* **57 books**

### Query 5: Relational `JOIN`, `GROUP BY`, `ORDER BY` — Category Pricing & Catalog Depth
```sql
SELECT 
    c.category_name,
    COUNT(b.book_id) AS total_books,
    ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
    ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
    MAX(b.rating) AS max_rating
FROM categories c
INNER JOIN books b ON c.category_id = b.category_id
GROUP BY c.category_id, c.category_name
ORDER BY total_books DESC, avg_price_inr DESC;
```
*Output:*
| category_name | total_books | avg_price_gbp | avg_price_inr | max_rating |
| :--- | :--- | :--- | :--- | :--- |
| Sequential Art | 75 | £34.57 | ₹3647.37 | 5 |
| Mystery | 32 | £31.72 | ₹3346.36 | 5 |
| Historical Fiction | 26 | £33.64 | ₹3549.47 | 5 |
| Travel | 11 | £39.79 | ₹4198.32 | 5 |

---

## SQL vs Pandas Merge Equivalence Verification

We read the relational tables into in-memory pandas DataFrames and executed `pd.merge()` followed by `groupby()` aggregation:

```python
merged = pd.merge(df_categories, df_books, on="category_id", how="inner")
df_pandas = merged.groupby(["category_id", "category_name"]).agg(
    total_books=("book_id", "count"),
    avg_price_gbp=("price_gbp", lambda x: round(x.mean(), 2)),
    avg_price_inr=("price_inr", lambda x: round(x.mean(), 2)),
    max_rating=("rating", "max")
).reset_index().drop(columns=["category_id"])

df_pandas = df_pandas.sort_values(by=["total_books", "avg_price_inr"], ascending=[False, False]).reset_index(drop=True)
pd.testing.assert_frame_equal(df_sql, df_pandas, check_dtype=False)
```

**Result:** `pd.testing.assert_frame_equal` passed with zero discrepancy, proving absolute parity between relational engine execution and DataFrame manipulations.

---

## How to Run

1. **Run standalone pipeline**:
   ```bash
   python data_pipeline/pipeline.py
   ```

2. **Run notebook**:
   Open and execute `data_pipeline/data_pipeline.ipynb` in Jupyter.
