# Discuss with your CSV (Groq)

An interactive data analysis tool that lets you chat with your CSV files using natural language!

## Features

- **Upload Any CSV**: Works with your data files, handles different encodings automatically
- **Instant Auto-Insights**: 
  - Quick statistical summaries
  - Data quality checks
  - Distribution analysis
  - Automatic visualization of key patterns

- **Natural Language to SQL**: 
  - Ask questions in plain English like "What's the total sales by region?"
  - Get instant SQL queries and results
  - Safe querying with SELECT-only operations
  - Visual representations of results

- **Smart Data Handling**:
  - Automatic date/time detection
  - Handles large files (auto-samples if > 100k rows)
  - Smart type inference
  - Missing value analysis

## Getting Started

1) `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2) `pip install -r requirements.txt`
3) Set your key: `export GROQ_API_KEY="your_key_here"` 
4) `streamlit run streamlit_app.py`

## Example Questions You Can Ask

- "Show me the top 10 categories by average price"
- "What's the monthly trend of sales?"
- "Compare performance across different regions"
- "Which products have the highest growth rate?"
- "Summarize the data distribution by category"
