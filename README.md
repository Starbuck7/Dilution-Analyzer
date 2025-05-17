# Stock Analyzer: Cash Runway, Market Cap, and ATM Offering

A Streamlit app that analyzes public companies by:
- Fetching market cap using Yahoo Finance
- Calculating cash runway using the latest SEC 10-Q or 10-K filing
- Detecting ATM offering capacity from SEC filings (424B5, 8-K, S-3)

## Deploy on Streamlit Cloud

1. Fork or clone this repo
2. Push to GitHub
3. Deploy at [streamlit.io/cloud](https://streamlit.io/cloud)

**Note:** The SEC requires a valid `User-Agent` header for API access. Update it in `app.py` with your name/email.

## Example Input
- `SYTA`
- `GNS`
- `SNTI`

Enjoy!
