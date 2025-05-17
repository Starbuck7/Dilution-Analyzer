import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re

# -------------------- Config --------------------
USER_AGENT = {"User-Agent": "Your Name (youremail@example.com)"}

# -------------------- Utility Functions --------------------
def get_cik_from_ticker(ticker):
    url = f"https://www.sec.gov/files/company_tickers.json"
    res = requests.get(url, headers=USER_AGENT).json()
    for item in res.values():
        if item['ticker'].upper() == ticker.upper():
            return str(item['cik_str']).zfill(10)
    return None

# -------------------- Module 1: Market Cap --------------------
def get_market_cap(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info.get("marketCap", None)

# -------------------- Module 2: Cash Runway --------------------
def get_cash_and_burn(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=USER_AGENT).json()
    filings = res.get("filings", {}).get("recent", {})
    for i, form in enumerate(filings.get("form", [])):
        if form in ["10-Q", "10-K"]:
            accession = filings["accessionNumber"][i].replace("-", "")
            doc = filings["primaryDocument"][i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
            cash_match = re.search(r"cash and cash equivalents[^$]*\$?([0-9.]+)", text, re.IGNORECASE)
            burn_match = re.search(r"monthly burn rate[^$]*\$?([0-9.]+)", text, re.IGNORECASE)
            if cash_match and burn_match:
                return float(cash_match.group(1)), float(burn_match.group(1))
    return None, None

def calculate_cash_runway(cash, burn):
    if cash and burn:
        return cash / burn
    return None

# -------------------- Module 3: ATM Offering Capacity --------------------
def get_atm_offering(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=USER_AGENT).json()
    filings = res.get("filings", {}).get("recent", {})
    for i, form in enumerate(filings.get("form", [])):
        if form == "424B5":
            accession = filings["accessionNumber"][i].replace("-", "")
            doc = filings["primaryDocument"][i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text()
            match = re.search(r"at-the-market[^$]*\$?([0-9,.]+)", text, re.IGNORECASE)
            if match:
                amount = float(match.group(1).replace(",", ""))
                return amount, html_url
    return None, None

# -------------------- Module 4: Authorized vs Outstanding Shares --------------------
def get_authorized_shares(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=USER_AGENT).json()
    filings = res.get("filings", {}).get("recent", {})
    for i, form in enumerate(filings.get("form", [])):
        if form == "DEF 14A":
            accession = filings["accessionNumber"][i].replace("-", "")
            doc = filings["primaryDocument"][i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
            match = re.search(r"authorized\s+shares[^0-9]+([0-9]+)", text, re.IGNORECASE)
            if match:
                return int(match.group(1))
    return None

def get_outstanding_shares(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=USER_AGENT).json()
    filings = res.get("filings", {}).get("recent", {})
    for i, form in enumerate(filings.get("form", [])):
        if form in ["10-Q", "10-K"]:
            accession = filings["accessionNumber"][i].replace("-", "")
            doc = filings["primaryDocument"][i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
            match = re.search(r"outstanding\s+shares[^0-9]+([0-9]+)", text, re.IGNORECASE)
            if match:
                return int(match.group(1))
    return None

# -------------------- Module 5: Convertibles and Warrants --------------------
def get_convertibles_and_warrants(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=USER_AGENT).json()
    filings = res.get("filings", {}).get("recent", {})
    instruments = []
    for i, form in enumerate(filings.get("form", [])):
        if form in ["S-1", "S-3", "424B5"]:
            accession = filings["accessionNumber"][i].replace("-", "")
            doc = filings["primaryDocument"][i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
            matches = re.findall(r"(convertible\s+(?:note|debenture|preferred\s+stock)|warrant[s]?)", text, re.IGNORECASE)
            for match in matches:
                instruments.append(match[0])
            if instruments:
                return list(set(instruments)), html_url
    return None, None

# -------------------- Module 6: Historical Capital Raises --------------------
def get_historical_capital_raises(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=USER_AGENT)
    if res.status_code != 200:
        return None
    data = res.json()
    filings = data.get("filings", {}).get("recent", {})
    capital_raises = []
    for i, form in enumerate(filings.get("form", [])):
        if form in ["S-1", "S-3", "424B5", "8-K"]:
            accession = filings["accessionNumber"][i].replace("-", "")
            doc = filings["primaryDocument"][i]
            filing_date = filings["filingDate"][i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html_res = requests.get(html_url, headers=USER_AGENT)
            if html_res.status_code != 200:
                continue
            soup = BeautifulSoup(html_res.text, "lxml")
            text = soup.get_text().replace(",", "")
            match = re.search(r"offering of.*?\$?([0-9.]+)\s?(million|thousand)?", text, re.IGNORECASE)
            if match:
                amount = float(match.group(1))
                unit = match.group(2)
                if unit == "million":
                    amount *= 1_000_000
                elif unit == "thousand":
                    amount *= 1_000
                capital_raises.append({
                    "form": form,
                    "date": filing_date,
                    "amount": amount,
                    "url": html_url
                })
    return capital_raises

# -------------------- Module 7: Get Public Float ---------------------

def get_public_float(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=USER_AGENT).json()
    filings = res.get("filings", {}).get("recent", {})
    for i, form in enumerate(filings.get("form", [])):
        if form == "10-K":  # Public float is often in 10-K
            accession = filings["accessionNumber"][i].replace("-", "")
            doc = filings["primaryDocument"][i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
            match = re.search(r"public float.*?\$?([0-9.]+)\s?(million|billion)?", text, re.IGNORECASE)
            if match:
                amount = float(match.group(1))
                unit = match.group(2)
                if unit == "billion":
                    amount *= 1_000_000_000
                elif unit == "million":
                    amount *= 1_000_000
                return amount
    return None

# -------------------- Module 8: Offering Ability --------------------
def get_shelf_registered_shares(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=USER_AGENT).json()
    filings = res.get("filings", {}).get("recent", {})
    for i, form in enumerate(filings.get("form", [])):
        if form in ["S-3", "S-1"]:
            accession = filings["accessionNumber"][i].replace("-", "")
            doc = filings["primaryDocument"][i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
            match = re.search(r"register[^$]*?\$?([0-9.]+)\s?(million|billion)?", text, re.IGNORECASE)
            if match:
                amount = float(match.group(1))
                unit = match.group(2)
                if unit == "billion":
                    amount *= 1_000_000_000
                elif unit == "million":
                    amount *= 1_000_000
                return amount
    return None

def estimate_offering_ability(cik):
    authorized = get_authorized_shares(cik)
    outstanding = get_outstanding_shares(cik)
    atm_capacity, _ = get_atm_offering(cik)
    public_float = get_public_float(cik)
    shelf_registered = get_shelf_registered_shares(cik)

    available_shares = (authorized - outstanding) if authorized and outstanding else 0
    max_baby_shelf = (public_float / 3) if public_float else 0
    offering_ceiling = min(max_baby_shelf, (shelf_registered or 0) + (atm_capacity or 0))

    return {
        "Available Shares": available_shares,
        "ATM Capacity": atm_capacity,
        "Shelf Registered Shares": shelf_registered,
        "Max Baby Shelf": max_baby_shelf,
        "Offering Ceiling": offering_ceiling
    }


# -------------------- Streamlit App --------------------
# [Streamlit code continues unchanged...]
