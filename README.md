# ETF Screener

This is a small project I built to combine two things I care about:  
my passion for investing and my interest in coding.  

It's been a while since I started investing in ETFs and I decided to try creating something that could be useful in my day-to-day investing, helping me evaluate and filter ETFs quickly.

> Note: this project is still a work in progress.  
> I'm building it step by step while learning and experimenting, so not all the features described below are fully working yet.

---

##  What it does
- Downloads historical ETF prices (via Yahoo Finance / `yfinance`)
- Computes **key performance and risk metrics**:
  - CAGR (annualized return)
  - Annual volatility
  - Maximum drawdown
  - Sharpe and Sortino ratios
- Screens ETFs with **realistic filters**:
  - TER (total expense ratio)
  - Volatility threshold
  - Maximum drawdown
  - Minimum average daily volume (Liquidity)
- Exports a clean **CSV ranking table**
- Creates simple **charts**:
  - Risk vs Return scatterplot
  - Equity curve vs benchmark

---

##  How to run it

```bash
# 1. Clone the repo
git clone https://github.com/davidedele/ETF_Screener.git
cd ETF_Screener

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the CLI
python cli.py all \
  --tickers VWCE --tickers CSPX --tickers IWDA \
  --start 2018-01-01 --end 2025-01-01 \
  --ter-csv data/reference/ter.csv

# 5. Results
The ranked CSV table and charts will be saved in `data/output/`.
