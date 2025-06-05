import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import warnings
import logging
import os
import yfinance as yf
import time
import glob
from bs4 import XMLParsedAsHTMLWarning
from datetime import datetime, timedelta
from yahoo_fin import stock_info as si
from sec_edgar_downloader import Downloader
from functools import lru_cache
 
# -------------------- Config --------------------
USER_AGENT = {"User-Agent": "DilutionAnalyzerBot/1.0 (ASHLEYMCGAVERN@YAHOO.COM)"}

# --- Setup ---
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Utility: Fetch SEC JSON ---
def fetch_sec_json(cik, headers=None):
    """
    Fetches the SEC JSON data for a given CIK.
    Returns the JSON dict if successful, otherwise None.
    """
    if headers is None:
        headers = USER_AGENT
    url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
    try:
        res = requests.get(url, headers=USER_AGENT)
        if res.status_code == 200:
            return res.json()
        else:
            logger.warning(f"SEC API returned status {res.status_code} for CIK {cik}")
    except Exception as e:
        logger.error(f"Failed to fetch SEC JSON for CIK {cik}: {e}")
    return None


def scrape_sec_filings_html(cik, forms=None, max_results=10):
    """
    Scrape the SEC EDGAR browse HTML page for ALL filings for a given CIK.
    Returns a list of dicts with form, accession number, doc link, and filing date.
    """
    cik = str(cik).lstrip('0')  # SEC wants no leading zeros for HTML query
    url = f"https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude"
    resp = requests.get(url, headers=USER_AGENT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    filings = []
    for row in soup.select('table.table tbody tr'):
        cols = row.find_all('td')
        if len(cols) < 5:
            continue
        form = cols[0].text.strip()
        if form not in forms:
            continue
        accession = cols[2].text.strip()
        date_filed = cols[3].text.strip()
        details_link = cols[1].find('a')
        doc_link = "https://www.sec.gov" + details_link['href'] if details_link else None
        filings.append({
            "form": form,
            "accession": accession,
            "date_filed": date_filed,
            "doc_link": doc_link,
        })
        if len(filings) >= max_results:
            break
    return filings

# -------------------- Utility: Improved CIK Lookup --------------------
@lru_cache(maxsize=100)
def get_cik_from_ticker(ticker):
    resp = requests.get("https://www.sec.gov/include/ticker.txt", headers=USER_AGENT)
    resp.raise_for_status()
    mapping = dict(line.split('\t') for line in resp.text.strip().split('\n'))
    cik = mapping.get(ticker.lower())
    if cik:
        return cik.zfill(10)
    raise ValueError(f"CIK not found for ticker: {ticker}")
 
# -------------------- Module 1: Market Cap --------------------
def get_market_cap(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        print("yfinance info:", info)  # Add this
        market_cap = info.get('marketCap')
        if market_cap:
            logger.info(f"Market cap for {ticker} from yfinance: {market_cap}")
            return market_cap
    except Exception as e:
        logger.warning(f"yfinance failed for {ticker} with error: {e}")
    # ...rest unchanged...

    # Try yahoo_fin.get_quote_table
    try:
        quote_table = si.get_quote_table(ticker, dict_result=True)
        market_cap_str = quote_table.get('Market Cap')
        if market_cap_str:
            return _parse_market_cap_str(market_cap_str)
    except Exception as e:
        logger.warning(f"yahoo_fin.get_quote_table failed for {ticker} with error: {e}")

    # Try yahoo_fin.get_stats_valuation
    try:
        val_table = si.get_stats_valuation(ticker)
        if not val_table.empty:
            mc_row = val_table[val_table.iloc[:, 0] == 'Market Cap (intraday)']
            if not mc_row.empty:
                market_cap_str = mc_row.iloc[0, 1]
                return _parse_market_cap_str(market_cap_str)
    except Exception as e:
        logger.warning(f"yahoo_fin.get_stats_valuation failed for {ticker} with error: {e}")

    logger.error(f"Could not retrieve market cap for {ticker}")
    return None

def _parse_market_cap_str(market_cap_str):
    try:
        multipliers = {'B': 1e9, 'M': 1e6, 'K': 1e3}
        if market_cap_str[-1] in multipliers:
            return int(float(market_cap_str[:-1]) * multipliers[market_cap_str[-1]])
        else:
            return int(market_cap_str.replace(',', ''))
    except Exception as e:
        logger.warning(f"Failed to parse market cap string '{market_cap_str}': {e}")
        return None

# -------------------- Module 2: Cash Runway --------------------

def get_latest_10q_10k(cik):
    # First try JSON feed
    url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
    resp = requests.get(url, headers=USER_AGENT)
    if resp.status_code == 200:
        data = resp.json()
        filings = data.get("filings", {}).get("recent", {})
        for i, form in enumerate(filings.get("form", [])):
            if form in ("10-Q", "10-K"):
            try:
                accession = filings["accessionNumber"][i].replace("-", "")
                file_name = filings["primaryDocument"][i]
                period = filings.get("periodOfReport", [None])[i]
                return {
                    "form": form,
                    "accession": accession,
                    "file_name": file_name,
                    "period": period,
                    "source": "recent"
                }
            except IndexError:
                continue
    # Try "files" for historical filings
    for f in data.get("filings", {}).get("files", []):
        f_url = "https://data.sec.gov" + f["name"]
        f_resp = requests.get(f_url, headers=USER_AGENT)
        f_resp.raise_for_status()
        f_data = f_resp.json()
        for i, form in enumerate(f_data.get("form", [])):
            if form in form_types:
                try:
                    accession = f_data["accessionNumber"][i].replace("-", "")
                    file_name = f_data["primaryDocument"][i]
                    period = f_data.get("periodOfReport", [None])[i]
                    return {
                        "form": form,
                        "accession": accession,
                        "file_name": file_name,
                        "period": period,
                        "source": "historical"
                    }
                except IndexError:
                    continue
    html_results = scrape_sec_filings_html(cik)
    if not html_results:
        raise ValueError("No 10-Q or 10-K filings found in JSON or HTML.")
    return html_results[0]


def fetch_filing_html(cik, accession, file_name):
    url = SEC_ARCHIVES_URL.format(cik=cik, accession_nodash=accession, file_name=file_name)
    resp = requests.get(url, headers=USER_AGENT)
    resp.raise_for_status()
    return resp.text
 
def extract_cash_and_burn(html):
    soup = BeautifulSoup(html, "lxml")

    # Find the period (look for period end date string)
    text = soup.get_text(separator="\n")
    period_match = re.search(r"(For the (three|six|nine|twelve)[- ]month[s]? ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4}))", text, re.IGNORECASE)
    if period_match:
        period_str = period_match.group(1)
        months_map = {"three": 3, "six": 6, "nine": 9, "twelve": 12}
        for k, v in months_map.items():
            if k in period_match.group(2).lower():
                period_months = v
                break
        else:
            period_months = None
    else:
        period_str = None
        period_months = None

    # Find cash/cash equivalents
    cash_val = None
    cash_rows = soup.find_all(string=re.compile(r"cash and cash equivalents", re.I))
    for row in cash_rows:
        # try to grab a nearby $ value
        parent = row.find_parent(["tr", "td", "th"])
        if parent:
            val_match = re.search(r"\$?[\(\-]?\d[\d,]*\.?\d*", parent.get_text())
            if val_match:
                cash_val = int(val_match.group().replace("$", "").replace(",", "").replace("(", "-").replace(")", ""))
                break

    # Find net cash used in operating activities (from cash flow statement)
    net_cash_used = None
    op_cash_rows = soup.find_all(string=re.compile(r"net cash used in operating activities", re.I))
    for row in op_cash_rows:
        parent = row.find_parent(["tr", "td", "th"])
        if parent:
            val_match = re.search(r"\$?[\(\-]?\d[\d,]*\.?\d*", parent.get_text())
            if val_match:
                net_cash_used = int(val_match.group().replace("$", "").replace(",", "").replace("(", "-").replace(")", ""))
                break

    return period_str, period_months, cash_val, net_cash_used


def calculate_burn_and_runway(cash, net_cash_used, period_months):
    if not (cash and net_cash_used and period_months):
        return None, None
    # net_cash_used is negative (cash outflow), so take its absolute and divide by period
    burn_rate = abs(net_cash_used) / period_months
    runway = cash / burn_rate if burn_rate else None
    return round(burn_rate, 2), round(runway, 2) if runway else None


def get_cash_runway_for_ticker(ticker):
    try:
        cik = get_cik_from_ticker(ticker)
        filing = get_latest_filing(cik)
        html = fetch_filing_html(cik, filing["accession"], filing["file_name"])
        period_str, period_months, cash, net_cash_used = extract_cash_and_burn(html)
        burn_rate, runway = calculate_burn_and_runway(cash, net_cash_used, period_months)
        return {
            "period_string": period_str,
            "period_months": period_months,
            "cash": cash,
            "net_cash_used": net_cash_used,
            "burn_rate": burn_rate,
            "runway_months": runway,
            "cik": cik,
            "accession": filing["accession"],
            "file_name": filing["file_name"],
            "form": filing["form"]
        }
    except Exception as e:
        return {"error": str(e)}


# --- Module 3: ATM Offering Capacity ---
def get_atm_offering(cik, lookback=10):
    try:
        data = fetch_sec_json(cik)
        if not data:
            return None, None
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])
        urls = []

        for i, form in enumerate(forms):
            if form in ["S-1", "S-3", "8-K", "424B5"] and i < lookback:
                accession = accessions[i].replace("-", "")
                doc = docs[i]
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
                urls.append((form, url))

        total_atm_usd = None
        sold_usd = 0

        for form, url in urls:
            html = requests.get(url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
            if form in ["S-1", "S-3"]:
                match = re.search(r"(?:at[-\s]the[-\s]market|ATM)[^$]{0,40}\$([\d\.]+)\s*(million|billion)?", text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    unit = match.group(2)
                    if unit:
                        val *= 1_000_000 if unit.lower() == "million" else 1_000_000_000
                    total_atm_usd = val
            elif form in ["8-K", "424B5"]:
                sold_matches = re.findall(r"under\s+the\s+ATM[^$]{0,100}\$([\d\.]+)\s*(million|billion)?", text, re.IGNORECASE)
                for match in sold_matches:
                    val = float(match[0])
                    unit = match[1]
                    if unit:
                        val *= 1_000_000 if unit.lower() == "million" else 1_000_000_000
                    sold_usd += val

        if total_atm_usd is not None:
            remaining = max(total_atm_usd - sold_usd, 0)
            return remaining, url  # return the most recent ATM-related URL
            
        html_results = scrape_sec_filings_html(cik)
        if not html_results:
            raise ValueError("No S-1, S-3 or 8-K filings found in JSON or HTML.")
        return html_results[0] 

    except Exception as e:
        logger.error(f"Error in get_atm_offering: {e}")
        return None, None


# -------------------- Module 4: Authorized vs Outstanding Shares & Float --------------------
def get_authorized_shares(cik):
    try:
        data = fetch_sec_json(cik)
        if not data:
            return None
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if form not in ["10-K", "10-Q", "DEF 14A"]:
                continue
            if i >= len(accessions) or i >= len(docs):
                continue
            accession = accessions[i].replace("-", "")
            doc = docs[i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
            patterns = [
                r"(?i)(?:total\s+)?authorized\s+(?:number\s+of\s+)?shares[^0-9]{0,20}([0-9]{5,})",
                r"(?i)number\s+of\s+authorized\s+shares[^0-9]{0,20}([0-9]{5,})",
                r"(?i)authorized\s+capital\s+stock[^0-9]{0,20}([0-9]{5,})",
                r"(?i)authorized[^0-9]{0,10}([0-9]{5,})\s+shares"
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return int(match.group(1))
        html_results = scrape_sec_filings_html(cik)
        if not html_results:
            raise ValueError("No filings found in JSON or HTML.")
        return html_results[0]
    
    except Exception as e:
        print(f"Error in get_authorized_shares: {e}")
        return None

def get_outstanding_shares(cik):
    try:
        data = fetch_sec_json(cik)
        if not data:
            return None
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])
        dates = filings.get("filingDate", [])
        for i, form in enumerate(forms):
            if form not in ["10-Q", "10-K", "DEF 14A"]:
                continue
            if i >= len(accessions) or i >= len(docs):
                continue
            accession = accessions[i].replace("-", "")
            doc = docs[i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
            patterns = [
                r"(?i)(?:common\s+stock\s+)?outstanding\s+(?:shares|stock)[^0-9]{0,20}([0-9]{5,})",
                r"(?i)shares\s+issued\s+and\s+outstanding[^0-9]{0,20}([0-9]{5,})",
                r"(?i)common\s+shares\s+outstanding[^0-9]{0,20}([0-9]{5,})",
                r"(?i)total\s+shares\s+outstanding[^0-9]{0,20}([0-9]{5,})",
                r"(?i)outstanding[^0-9]{0,10}([0-9]{5,})\s+shares",
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return int(match.group(1))
        html_results = scrape_sec_filings_html(cik)
        if not html_results:
            raise ValueError("No filings found in JSON or HTML.")
        return html_results[0]
    except Exception as e:
        print(f"Error in get_outstanding_shares: {e}")
        return None

def get_public_float(cik):
    try:
        data = fetch_sec_json(cik)
        if not data:
            return None
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if form == "10-K":  # Public float is often in 10-K
                accession = accessions[i].replace("-", "")
                doc = docs[i]
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
        html_results = scrape_sec_filings_html(cik)
        if not html_results:
            raise ValueError("No filings found in JSON or HTML.")
        return html_results[0]
    except Exception as e:
        logger.error(f"Error getting public float: {e}")
        return None
       
# -------------------- Module 5: Convertibles and Warrants --------------------
def get_convertibles_and_warrants_with_amounts(cik):
    try:
        data = fetch_sec_json(cik)
        if not data:
            return [], []
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])
        results = []
        for i, form in enumerate(forms):
            if form in ["10-Q", "10-K", "8-K"]:
                if i >= len(accessions) or i >= len(docs):
                    continue
                accession = accessions[i].replace("-", "")
                doc = docs[i]
                html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
                html = requests.get(html_url, headers=USER_AGENT).text
                text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
                # Look for convertible debt/notes
                conv = re.findall(r"convertible.*?(?:notes?|debentures?).{0,100}?\$([-\d,\.]+)", text, re.IGNORECASE)
                # Look for warrants
                warrants = re.findall(r"warrants?.{0,30}?(?:to purchase)?\s*([-\d,\.]+)\s*(?:shares|stock)?", text, re.IGNORECASE)
                result = {}
                if conv:
                    try:
                        result["convertible"] = float(conv[0].replace(",", ""))
                    except:
                        result["convertible"] = conv[0]
                if warrants:
                    try:
                        result["warrants"] = float(warrants[0].replace(",", ""))
                    except:
                        result["warrants"] = warrants[0]
                if result:
                    results.append((result, html_url))
        return results  # List of (dict, url)
        results = scrape_sec_filings_html(cik, forms=form_types)
        if not results:
            raise ValueError(f"No {', '.join(form_types)} filings found via JSON or HTML.")
        return results[0]  # Most recent filing of specified type(s)
    except Exception as e:
        print(f"Error in get_convertibles_and_warrants_with_amounts: {e}")
        return []


# -------------------- Module 6: Historical Capital Raises --------------------
def summarize_recent_cap_raises(raises, months=18):
    """Summarize capital raises in the past 'months' months."""
    if not raises:
        return 0, 0.0, []
    cutoff = datetime.now() - timedelta(days=months*30)
    recent = [
        entry for entry in raises
        if "date" in entry and datetime.strptime(entry["date"], "%Y-%m-%d") > cutoff
    ]
    total_amt = sum(entry["amount"] for entry in recent)
    return len(recent), total_amt, recent
    
def get_historical_capital_raises(cik):
    try:
        data = fetch_sec_json(cik)
        if not data:
            return None
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])
        dates = filings.get("filingDate", [])
        capital_raises = []
        for i, form in enumerate(forms):
            if form in ["8-K", "424B5", "S-1", "S-3"]:
                if i >= len(accessions) or i >= len(docs) or i >= len(dates):
                    continue  # Skip incomplete entries
                accession = accessions[i].replace("-", "")
                doc = docs[i]
                date = dates[i]
                html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
                html = requests.get(html_url, headers=USER_AGENT).text
                text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
                patterns = [
                    r"gross\s+proceeds\s+of\s+\$?([0-9,]+(?:\.[0-9]{1,2})?)",
                    r"raised\s+\$?([0-9,]+(?:\.[0-9]{1,2})?)",
                    r"proceeds\s+of\s+\$?([0-9,]+(?:\.[0-9]{1,2})?)",
                    r"offering\s+of\s+\$?([0-9,]+(?:\.[0-9]{1,2})?)"
                ]
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            amount = float(match.group(1).replace(",", ""))
                            capital_raises.append({
                                "date": date,
                                "form": form,
                                "amount": amount,
                                "url": html_url
                            })
                            break  # Stop after the first match
                        except:
                            continue
        return capital_raises if capital_raises else None
    except Exception as e:
        print(f"Error in get_historical_capital_raises: {e}")
        return None


# -------------------- Module 7: Offering Ability --------------------
def get_shelf_registered_shares(cik, num_filings=10):
    try:
        cik_str = str(cik).zfill(10)
        filing_types = ['S-3', 'S-1', '424B3']
        all_text = ""
        for form_type in filing_types:
            dl.get(form_type, cik_str)
            path = f"/tmp/sec/sec-edgar-filings/{cik_str}/{form_type.lower()}"
            for subdir, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".txt"):
                        with open(os.path.join(subdir, file), "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                            all_text += "\n" + text
        dollar_matches = re.findall(
            r"offer(?:ing)?(?: and sell)? (?:up to|of up to)?\s*\$([\d,.]+)\s*(million|billion)?",
            all_text, re.IGNORECASE)
        amounts = []
        for match in dollar_matches:
            num_str, magnitude = match
            num = float(num_str.replace(",", "").strip())
            if magnitude:
                if magnitude.lower() == "million":
                    num *= 1_000_000
                elif magnitude.lower() == "billion":
                    num *= 1_000_000_000
            amounts.append(num)
        share_matches = re.findall(
            r"offer(?:ing)?(?: and sell)? (?:up to|of up to)?\s*([\d,.]+)\s*(shares|common stock)?",
            all_text, re.IGNORECASE)
        for match in share_matches:
            num_str, _ = match
            num = float(num_str.replace(",", "").strip())
            amounts.append(num * 1.00)
        if amounts:
            return max(amounts)
        return None
    except Exception as e:
        print(f"Error extracting shelf registered shares: {e}")
        return None

def estimate_offering_ability(cik):
    try:
        authorized = get_authorized_shares(cik)
        outstanding = get_outstanding_shares(cik)
        available_shares = max(authorized - outstanding, 0) if authorized and outstanding else None
        atm_usd, _ = get_atm_offering(cik)
        shelf_usd = get_shelf_registered_shares(cik)
        float_shares = get_public_float(cik)
        values = {
            "Available Shares": available_shares,
            "ATM Capacity": atm_usd,
            "Shelf Registered Shares": shelf_usd,
            "Public Float": float_shares
        }
        if not any(v for v in values.values() if v):
            return {"Status": "Unable to determine offering ability (missing data)"}
        return values
    except Exception as e:
        print(f"Error estimating offering ability: {e}")
        return {"Status": f"Error: {e}"}


# -------------------- Module 8: Dilution Pressure Score --------------------
def get_atm_capacity_score(atm_capacity_usd, market_cap):
    if not atm_capacity_usd or not market_cap:
        return 10  # neutral if missing
    ratio = atm_capacity_usd / market_cap
    if ratio > 0.75:
        return 25
    elif ratio > 0.5:
        return 20
    elif ratio > 0.25:
        return 15
    elif ratio > 0.1:
        return 10
    else:
        return 5

def get_authorized_vs_outstanding_score(authorized, outstanding, market_cap):
    if not authorized or not outstanding or not market_cap:
        return 5
    available = authorized - outstanding
    est_value = available * 0.5  # assume $0.50 dilution price
    ratio = est_value / market_cap
    if ratio > 1:
        return 25
    elif ratio > 0.5:
        return 20
    elif ratio > 0.25:
        return 10
    else:
        return 5

def get_convertibles_score(instruments, market_cap):
    if not instruments or not market_cap:
        return 5
    # Estimate rough value (you can make this smarter later with parsing)
    convertibles_usd = 2_000_000 if len(instruments) > 0 else 0
    ratio = convertibles_usd / market_cap
    if ratio > 0.5:
        return 20
    elif ratio > 0.25:
        return 10
    elif ratio > 0.1:
        return 5
    else:
        return 0

def get_capital_raises_score(num_raises):
    if num_raises >= 4:
        return 15
    elif num_raises == 3:
        return 10
    elif num_raises == 2:
        return 7
    elif num_raises == 1:
        return 4
    else:
        return 0

def get_cash_runway_score(runway_months):
    if runway_months is None:
        return 5
    elif runway_months < 3:
        return 15
    elif runway_months < 6:
        return 10
    elif runway_months < 12:
        return 5
    else:
        return 0

def calculate_dilution_pressure_score(
    atm_capacity_usd, authorized_shares, outstanding_shares,
    convertibles, capital_raises_past_year, cash_runway, market_cap
):
    """Calculates a dilution risk score based on financial pressures."""
    
    def safe_value(value, default=0):
        """Prevents crashes by handling missing values gracefully."""
        return value if value is not None else default

    # ✅ Ensures missing values won’t break calculations
    total_score = 0
    total_score += get_atm_capacity_score(safe_value(atm_capacity_usd), safe_value(market_cap))
    total_score += get_authorized_vs_outstanding_score(safe_value(authorized_shares), safe_value(outstanding_shares), safe_value(market_cap))
    total_score += get_convertibles_score(safe_value(convertibles), safe_value(market_cap))
    total_score += get_capital_raises_score(safe_value(capital_raises_past_year))
    total_score += get_cash_runway_score(safe_value(cash_runway))

    return min(total_score, 100)  # ✅ Caps score at 100


   
# -------------------- Streamlit App --------------------
st.title("Stock Analysis Dashboard")
st.markdown("Analyze dilution and financial health based on SEC filings.")

# Input ticker
ticker = st.text_input("Enter a stock ticker (e.g., SYTA)", "").strip().upper()

if ticker:
    cik = get_cik_from_ticker(ticker)
    if not cik:
        st.error("CIK not found for this ticker.")
    else:
        st.success(f"CIK found: {cik}")

     #Module 1: Market Cap
        market_cap = get_market_cap(ticker)
        st.subheader("1. Market Cap")
        st.write(f"Market Cap: ${market_cap:,.0f}" if market_cap is not None else "Market Cap: Not available")

        # Module 2: Cash Runway
        st.header("Module 2: Cash Runway")
    with st.spinner("Analyzing cash runway..."):
        result = get_cash_runway_for_ticker(ticker)
    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.success(f"Latest {result['form']} for {ticker} (CIK {result['cik']}):")
        st.write(f"- Accession: {result['accession']}")
        st.write(f"- File: {result['file_name']}")
        if result['period_string']:
            st.write(f"**Reporting period:** {result['period_string']} ({result['period_months']} months)")
        if result['cash'] is not None:
            st.write(f"**Cash and cash equivalents:** ${result['cash']:,}")
        if result['net_cash_used'] is not None:
            st.write(f"**Net cash used in operating activities:** ${result['net_cash_used']:,}")
        if result['burn_rate'] is not None:
            st.write(f"**Burn rate:** ${result['burn_rate']:,} per month")
        if result['runway_months'] is not None:
            st.write(f"### 🚦 Estimated Cash Runway: **{result['runway_months']} months**")
        else:
            st.warning("Could not estimate cash runway (missing data).")
            
        #Module 3: ATM Offering Capacity
        atm, atm_url = get_atm_offering(cik, lookback=10)
        st.subheader("3. ATM Offering Capacity")
        if atm:
            st.write(f"${atm:,.0f}")
            if atm_url:
                st.markdown(f"[Source Document]({atm_url})")
        else:
            st.write("No ATM filing found.")

        #Module 4: Authorized vs Outstanding Shares & Float
        float_val = get_public_float(cik)
        authorized = get_authorized_shares(cik)
        outstanding = get_outstanding_shares(cik)
        st.subheader("4. Authorized vs Outstanding Shares")
        st.write(f"Public Float: {float_val:,}" if float_val else "Public Float: Not found")
        st.write(f"Authorized Shares: {authorized:,}" if authorized else "Authorized Shares: Not found")
        st.write(f"Outstanding Shares: {outstanding:,}" if outstanding else "Outstanding Shares: Not found")

        #Moduele 5: Convertibles & Warrants
        convertible_results = get_convertibles_and_warrants_with_amounts(cik)
        instruments = convertible_results if convertible_results else []  # Always define instruments
        st.subheader("5. Convertibles and Warrants")
        if convertible_results:
            for r, url in convertible_results:
                st.write(", ".join(f"{k}: {v}" for k, v in r.items()))
                st.markdown(f"[Source Document]({url})")
        else:
            st.write("No convertible instruments or warrants detected.")

        #Module 6: Historical Capital Raises
        raises = get_historical_capital_raises(cik)
        st.subheader("6. Historical Capital Raises")

        if raises:
            count_18mo, total_18mo, raises_18mo = summarize_recent_cap_raises(raises, months=18)
            st.write(f"**Raises in last 18 months:** {count_18mo}  |  **Total Raised:** ${total_18mo:,.0f}")
            st.write("**All Raises:**")
            for entry in raises:
                st.write(f"- {entry['form']} on {entry['date']}: ${entry['amount']:,.0f}")
                if entry.get('url'):
                    st.markdown(f"[Filing]({entry['url']})")
        else:
            st.write("No historical raises found.")
            
        #Module 7: Offering Ability
        offering_data = estimate_offering_ability(cik)
        st.subheader("7. Offering Ability")
        for k, v in offering_data.items():
            try:
                if v is not None and isinstance(v, (int, float)):
                    st.write(f"{k}: {v:,.0f}")
                else:
                    st.write(f"{k}: {v}")
            except Exception as ex:
                st.write(f"{k}: [error displaying value] ({ex})")

      # 8. Dilution Pressure Score
        st.subheader("8. Dilution Pressure Score")
        st.caption("Combines cash runway, ATM capacity, dilution ability, and more to assess dilution risk.")

        try:
            # Make sure all variables exist
            instruments = instruments if 'instruments' in locals() else []
            raises = raises if 'raises' in locals() else []
            available_dilution_shares = (authorized - outstanding) if authorized and outstanding else 0
            convertible_total_usd = 2_000_000 if instruments else 0  # Or a better estimate!
            num_raises_past_year = 0
            if raises and isinstance(raises, list):
                num_raises_past_year = len([
                    entry for entry in raises
                    if "date" in entry and datetime.strptime(entry["date"], "%Y-%m-%d") > datetime.now() - timedelta(days=365)
                ])
            score = calculate_dilution_pressure_score(
                atm_capacity_usd=atm,
                authorized_shares=authorized,
                outstanding_shares=outstanding,
                convertibles=convertible_total_usd,
                capital_raises_past_year=num_raises_past_year,
                cash_runway=runway if 'runway' in locals() else None,
                market_cap=market_cap
            )
            if score is not None:
                st.metric("Score (0-100)", f"{score}")
                if score > 70:
                    st.warning("⚠️ High Dilution Risk")
                elif score > 40:
                    st.info("🟡 Moderate Dilution Risk")
                else:
                    st.success("🟢 Low Dilution Risk")
            else:
                st.write("Insufficient data to calculate score.")
        except Exception as e:
            st.error(f"Error calculating score: {e}")
