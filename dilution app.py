import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import warnings
import logging
import os
import yfinance as yf
from bs4 import XMLParsedAsHTMLWarning
from datetime import datetime, timedelta
from yahoo_fin import stock_info as si
from sec_edgar_downloader import Downloader
from functools import lru_cache
dl = Downloader(email_address="ashleymcgavern@yahoo.com", company_name="Dilution Analyzer")
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  

dir_path = os.path.join(os.getcwd(), "sec-edgar-filings")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)  # ✅ Create directory if missing
    logger.warning(f"Directory {dir_path} was missing and has been created.")
else:
    print("DIR:", os.listdir(dir_path))

# -------------------- Config --------------------
USER_AGENT = {"User-Agent": "DilutionAnalyzerBot/1.0"}

# -------------------- Utility: Improved CIK Lookup --------------------
@lru_cache(maxsize=100)
def get_cik_from_ticker(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        res = requests.get(url, headers={"User-Agent": "DilutionAnalyzerBot/1.0"})

        # ✅ Check if response is empty
        if res.status_code != 200 or not res.text.strip():
            raise Exception(f"SEC API returned an invalid response for {ticker}")

        data = res.json()  # ✅ Ensure the response is valid JSON
        for item in data.values():
            if item['ticker'].upper() == ticker.upper():
                return str(item['cik_str']).zfill(10)

    except Exception as e:
        logger.error(f"Error fetching CIK for {ticker}: {e}")
    
    return None


    # Ensure ticker is defined before proceeding
    if ticker:
        cik = get_cik_from_ticker(ticker)
        if cik:
            try:
                dl.get("10-Q", ticker)
                dl.get("10-K", ticker)
            except Exception as e:
                logger.error(f"{ticker} - Failed to download filings: {e}")


# -------------------- Module 1: Market Cap --------------------
def get_market_cap(ticker):
    # Try yfinance
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        market_cap = info.get('marketCap')
        if market_cap:
            logger.info(f"Market cap for {ticker} from yfinance: {market_cap}")
            return market_cap
    except Exception as e:
        logger.warning(f"yfinance failed for {ticker} with error: {e}")

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
def parse_dollar_amount(text):
    """Extracts dollar amounts while handling variations in SEC formatting."""
    match = re.search(r'\$?\(?([\d,\.]+)\)?', text)
    if match:
        amount = match.group(1).replace(",", "")
        try:
            return float(amount)
        except ValueError:
            return None
    return None

def extract_operating_cash_flow(text):
    """Extracts burn rate with more robust regex and better context detection."""
    pattern = r'net cash used in operating activities[^$\d]{0,50}\$?([\d,\.]+)'
    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))

    for match in matches:
        value_str = match.group(1)
        value = parse_dollar_amount(value_str)

        # 🛠 Improved method to detect period more reliably
        window_start = max(0, match.start() - 300)
        window_text = text[window_start:match.start()].lower()

        months_map = {"three": 3, "six": 6, "nine": 9, "twelve": 12}
        for period in months_map:
            if period in window_text:
                return value, months_map[period]

    return None, None  # ✅ Returns None safely if extraction fails

def extract_cash_position(text):
    """Extracts cash balance with improved regex scope."""
    pattern = r'cash and cash equivalents(?:[^$\d]{0,40})\$?([\d,\.]+)'
    match = re.search(pattern, text, re.IGNORECASE)
   
    if match:
        return parse_dollar_amount(match.group(1))
    return None

def get_cash_and_burn_dl(ticker, downloader):
    """Extracts cash position and burn rate from SEC filings."""
    try:
        base_path = os.path.join(os.getcwd(), "sec-edgar-filings")
        for form_type in ["10-Q", "10-K"]:
            form_path = os.path.join(base_path, ticker, form_type)
            if not os.path.exists(form_path):
                continue

            subdirs = sorted(
                [os.path.join(form_path, d) for d in os.listdir(form_path)],
                key=os.path.getmtime, reverse=True
            )
            for subdir in subdirs:
                files = [f for f in os.listdir(subdir) if f.endswith((".txt", ".htm", ".html"))]
                if not files:
                    continue
                with open(os.path.join(subdir, files[0]), "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().lower()

                cash = extract_cash_position(text)
                burn_raw, months = extract_operating_cash_flow(text)
                monthly_burn = burn_raw / months if burn_raw and months else None

                # 🛠 **Improved Error Handling** - Logs missing values
                if cash is None or monthly_burn is None:
                    logger.warning(f"{ticker} - Cash or Burn rate extraction failed.")
                    return None, None

                logger.info(f"{ticker}: Cash: {cash}, Burn: {monthly_burn}")
                return cash, monthly_burn

    except Exception as e:
        logger.error(f"{ticker} - Error in get_cash_and_burn_dl: {e}")
        return None, None

def calculate_cash_runway(cash, burn):
    """Calculates how many months of runway a company has left."""
    if cash is None or burn is None or burn == 0:
        return None
    return round(cash / burn, 1)  # ✅ Keeps precision while avoiding div-by-zero


# -------------------- Module 3: ATM Offering Capacity --------------------
def get_atm_offering(cik, lookback=10):
    try:
        cik_str = str(cik).zfill(10)
        base_url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
        headers = {"User-Agent": "Ashley (ashleymcgavern@yahoo.com)"}

        # Get recent filings metadata
        res = requests.get(base_url, headers=headers)
        if res.status_code != 200:
            raise Exception(f"Failed to fetch submissions for CIK {cik_str}")
        filings = res.json().get("filings", {}).get("recent", {})

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
            html = requests.get(url, headers=headers).text
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
        else:
            return None, None

    except Exception as e:
        logger.error(f"Error in get_atm_offering: {e}")
        return None, None

# -------------------- Module 4: Authorized vs Outstanding Shares & Float --------------------
def get_authorized_shares(cik):
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=USER_AGENT)
        if res.status_code != 200:
            return None

        filings = res.json().get("filings", {}).get("recent", {})
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
        return None
    except Exception as e:
        print(f"Error in get_authorized_shares: {e}")
        return None


def get_outstanding_shares(cik):
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=USER_AGENT)
        if res.status_code != 200:
            return None

        filings = res.json().get("filings", {}).get("recent", {})
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
        return None
    except Exception as e:
        print(f"Error in get_outstanding_shares: {e}")
        return None

        
def get_public_float(cik):
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=USER_AGENT)
        if res.status_code != 200:
            return None

        filings = res.json().get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])
        
        for i, form in enumerate(forms):
            if form != "DEF 14A":
                continue
            if i >= len(accessions) or i >= len(docs):
                continue
            
            accession = accessions[i].replace("-", "")
            doc = docs[i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
            
            # Try to extract insider ownership
            insider_match = re.search(r"beneficially\s+owned\s+by\s+officers\s+and\s+directors[^0-9]+([0-9]{3,})", text, re.IGNORECASE)
            if insider_match:
                insider_shares = int(insider_match.group(1))
                break
        else:
            insider_shares = 0  # Fallback if no DEF 14A or no match

        # Get outstanding shares
        outstanding = get_outstanding_shares(cik)
        if not outstanding:
            return None

        # Estimate float
        float_shares = outstanding - insider_shares
        return max(float_shares, 0)

    except Exception as e:
        print(f"Error getting float: {e}")
        return None
       
# -------------------- Module 5: Convertibles and Warrants --------------------
def get_convertibles_and_warrants(cik):
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=USER_AGENT)
        if res.status_code != 200:
            return None, None

        filings = res.json().get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])

        instruments = []

        for i, form in enumerate(forms):
            if form in ["10-Q", "10-K", "8-K"]:
                if i >= len(accessions) or i >= len(docs):
                    continue  # Skip incomplete entries

                accession = accessions[i].replace("-", "")
                doc = docs[i]
                html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
                html = requests.get(html_url, headers=USER_AGENT).text
                text = BeautifulSoup(html, "lxml").get_text().replace(",", "")

                # Look for common convertibles/warrant patterns
                patterns = [
                    r"(?:convertible\s+(?:notes|debentures|securities)[^\.]{0,100})",
                    r"(?:warrants\s+to\s+purchase\s+[^\.;]{0,100})",
                    r"(?:preferred\s+stock\s+convertible\s+into\s+common\s+stock[^\.]{0,100})"
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        instruments.append(match.strip())

                if instruments:
                    return instruments, html_url  # Return on first good find

        return None, None
    except Exception as e:
        print(f"Error in get_convertibles_and_warrants: {e}")
        return None, None


# -------------------- Module 6: Historical Capital Raises --------------------
def get_historical_capital_raises(cik):
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=USER_AGENT)
        if res.status_code != 200:
            return None

        filings = res.json().get("filings", {}).get("recent", {})
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

                # Look for language about capital raised
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
def get_shelf_registered_shares(cik, num_filings=10):
    try:
        cik_str = str(cik).zfill(10)
        local_dl = Downloader(email_address="ashleymcgavern@yahoo.com", company_name="Dilution Analyzer")
        
        # Download recent S-3, S-1, and 424B3 filings
        filing_types = ['S-3', 'S-1', '424B3']
        all_text = ""

        for form_type in filing_types:
            dl.get(form_type, cik_str, amount=num_filings)
            path = f"/tmp/sec/sec-edgar-filings/{cik_str}/{form_type.lower()}"
            for subdir, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".txt"):
                        with open(os.path.join(subdir, file), "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                            all_text += "\n" + text

        # Try to find registered USD amounts
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

        # Fallback: Try to find registered share counts
        share_matches = re.findall(
            r"offer(?:ing)?(?: and sell)? (?:up to|of up to)?\s*([\d,.]+)\s*(shares|common stock)?",
            all_text, re.IGNORECASE)

        for match in share_matches:
            num_str, _ = match
            num = float(num_str.replace(",", "").strip())
            # Assume nominal price of $1/share if no dollar value is found
            amounts.append(num * 1.00)

        if amounts:
            return max(amounts)

        return None

    except Exception as e:
        print(f"Error extracting shelf registered shares: {e}")
        return None

def get_last_close_price(ticker):
    try:
        return si.get_live_price(ticker)
    except Exception as e:
        print(f"Error fetching last close price: {e}")
        return None

def estimate_offering_ability(cik):
    try:
        authorized = get_authorized_shares(cik)
        outstanding = get_outstanding_shares(cik)
        available_shares = max(authorized - outstanding, 0) if authorized and outstanding else None

        # ATM Capacity
        atm_usd, _ = get_atm_offering(cik)

        # Shelf Shares (estimated)
        shelf_usd = get_shelf_registered_shares(cik)

        # Public float estimate (for Baby Shelf Rule)
        float_shares = get_public_float(cik)
        float_price = get_last_close_price(ticker)
        max_baby_shelf = None
        if float_shares and float_price:
            float_value = float_shares * float_price
            if float_value < 75000000:  # If company qualifies for baby shelf rule
                max_baby_shelf = float_value / 3

        # Offering ceiling = most restrictive cap
        ceiling_usd = min([
            x for x in [atm_usd, shelf_usd, max_baby_shelf] if x is not None
        ], default=0)

        return {
            "Available Shares": available_shares,
            "ATM Capacity": atm_usd,
            "Shelf Registered Shares": shelf_usd,
            "Max Baby Shelf": max_baby_shelf,
            "Offering Ceiling": ceiling_usd
        }

    except Exception as e:
        print(f"Error estimating offering ability: {e}")
        return {}


# -------------------- Module 9: Dilution Pressure Score --------------------
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

        # Market Cap
        market_cap = get_market_cap(ticker)
        st.subheader("1. Market Cap")
        st.write(f"Market Cap: ${market_cap:,.0f}" if market_cap is not None else "Market Cap: Not available")

        # Module 2: Cash Runway
        cash, burn = get_cash_and_burn_dl(ticker, dl)
        runway = calculate_cash_runway(cash, burn)

        st.subheader("2. Cash Runway")
        if cash:
            st.write(f"Cash: ${cash:,.0f}")
        if burn:
            st.write(f"Monthly Burn Rate: ${burn:,.0f}")
        if runway:
            st.write(f"Runway: {runway:.1f} months")
        else:
            st.warning("Cash or burn rate not found.")


        # ATM Offering
        atm, atm_url = get_atm_offering(cik, lookback=10)
        st.subheader("3. ATM Offering Capacity")
        if atm:
            st.write(f"${atm:,.0f}")
            st.markdown(f"[Source Document]({atm_url})")
        else:
            st.write("No ATM filing found.")

        # Authorized vs Outstanding
        authorized = get_authorized_shares(cik)
        outstanding = get_outstanding_shares(cik)
        st.subheader("4. Authorized vs Outstanding Shares")
        st.write(f"Authorized Shares: {authorized:,}" if authorized else "Not found")
        st.write(f"Outstanding Shares: {outstanding:,}" if outstanding else "Not found")

        # Convertibles & Warrants
        instruments, cw_url = get_convertibles_and_warrants(cik)
        st.subheader("5. Convertibles and Warrants")
        if instruments:
            st.write(", ".join(set(instruments)))
            st.markdown(f"[Source Document]({cw_url})")
        else:
            st.write("No convertible instruments or warrants detected.")

        # Historical Capital Raises
        raises = get_historical_capital_raises(cik)
        st.subheader("6. Historical Capital Raises")
        if raises:
            for entry in raises:
                st.write(f"- {entry['form']} on {entry['date']}: ${entry['amount']:,.0f}")
                st.markdown(f"[Filing]({entry['url']})")
        else:
            st.write("No historical raises found.")

        # Offering Ability
        offering_data = estimate_offering_ability(cik)
        st.subheader("7. Offering Ability")
        for k, v in offering_data.items():
            st.write(f"{k}: {v:,.0f}" if isinstance(v, (int, float)) else f"{k}: {v}")

        # Gathering all values for Dilution Score
        available_dilution_shares = (authorized - outstanding) if authorized and outstanding else 0
        convertible_total_usd = 2_000_000 if instruments else 0  # ✅ Correct
        num_raises_past_year = len([
            entry for entry in raises
            if datetime.strptime(entry["date"], "%Y-%m-%d") > datetime.now() - timedelta(days=365)
        ]) if raises else 0

        # Optional red flag (for future expansion)
        red_flags_score = 0

      # Module 9: Calculate Dilution Pressure Score
try:
    convertible_total_usd = 2_000_000 if instruments else 0  # ✅ Placeholder estimate
    available_dilution_shares = (authorized - outstanding) if authorized and outstanding else 0
    num_raises_past_year = len([
        entry for entry in raises
        if datetime.strptime(entry["date"], "%Y-%m-%d") > datetime.now() - timedelta(days=365)
    ]) if raises else 0

    score = calculate_dilution_pressure_score(
        atm_capacity_usd=atm,
        authorized_shares=authorized,
        outstanding_shares=outstanding,
        convertibles=convertible_total_usd,
        capital_raises_past_year=num_raises_past_year,
        cash_runway=runway,
        market_cap=market_cap
    )

    st.subheader("8. Dilution Pressure Score")
    st.caption("Combines cash runway, ATM capacity, dilution ability, and more to assess dilution risk.")
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
    st.subheader("8. Dilution Pressure Score")
    st.error(f"Error calculating score: {e}")







