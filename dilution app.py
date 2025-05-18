import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
import warnings
from bs4 import XMLParsedAsHTMLWarning
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# -------------------- Config --------------------
USER_AGENT = {"User-Agent": "Ashley (ashleymcgavern@yahoo.com)"}

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
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=USER_AGENT)
        if res.status_code != 200:
            return None, None
        data = res.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if i >= len(accessions) or i >= len(docs):
                continue
            if form in ["10-Q", "10-K"]:
                accession = accessions[i].replace("-", "")
                doc = docs[i]
                html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
                html = requests.get(html_url, headers=USER_AGENT).text
                text = BeautifulSoup(html, "lxml").get_text().replace(",", "").lower()
                cash_match = re.search(r"cash and cash equivalents[^$0-9]{0,20}\$?([0-9.]+)", text)
                burn_match = re.search(r"(monthly burn rate|net cash used in operating activities)[^$0-9]{0,20}\$?([0-9.]+)", text)
                cash = float(cash_match.group(1)) if cash_match else None
                burn = float(burn_match.group(2)) / 3 if burn_match and "operating" in burn_match.group(1) else float(burn_match.group(2)) if burn_match else None
                return cash, burn
    except Exception as e:
        print(f"Error in get_cash_and_burn: {e}")
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
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=USER_AGENT)
        if res.status_code != 200:
            return None
        data = res.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if i >= len(accessions) or i >= len(docs):
                continue
            if form == "DEF 14A":
                accession = accessions[i].replace("-", "")
                doc = docs[i]
                html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
                html = requests.get(html_url, headers=USER_AGENT).text
                text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
                match = re.search(r"(?i)authorized\s+(?:number of\s+)?shares[^0-9]{0,20}([0-9]{3,})", text)
                if match:
                    return int(match.group(1))
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
            if i >= len(accessions) or i >= len(docs) or i >= len(dates):
                continue  # Skip incomplete entries

            if form in ["10-Q", "10-K", "DEF 14A"]:  # Expand if needed
                accession = accessions[i].replace("-", "")
                doc = docs[i]
                html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
                html = requests.get(html_url, headers=USER_AGENT).text
                text = BeautifulSoup(html, "lxml").get_text().replace(",", "")

                # Try primary pattern
                match = re.search(r"(?:common\s+stock\s+)?outstanding\s+(?:shares|stock)[^0-9]{0,20}([0-9]{3,})", text, re.IGNORECASE)
                if match:
                    return int(match.group(1))

                # Fallback: try additional phrasing patterns
                patterns = [
                    r"(?i)shares\s+issued\s+and\s+outstanding[^0-9]+([0-9,]+)",
                    r"(?i)common\s+shares\s+outstanding[^0-9]+([0-9,]+)",
                    r"(?i)total\s+shares\s+outstanding[^0-9]+([0-9,]+)",
                    r"(?i)shares\s+outstanding[^0-9]+([0-9,]+)"
                ]
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match:
                        return int(match.group(1).replace(",", ""))
        return None
    except Exception as e:
        print(f"Error in get_outstanding_shares: {e}")
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

#-------------MODULE 9: CALCULATIONG DILUTION PRESSURE SCORE ------------
def get_atm_capacity_score(available_atm_usd, market_cap):
    if not market_cap or not available_atm_usd:
        return 10  # neutral if missing

    ratio = available_atm_usd / market_cap
    if ratio > 0.75:
        return 20
    elif ratio > 0.5:
        return 15
    elif ratio > 0.25:
        return 10
    elif ratio > 0.1:
        return 5
    else:
        return 0

def get_convertibles_score(convertible_total_usd, market_cap):
    if not market_cap or not convertible_total_usd:
        return 5  # neutral

    ratio = convertible_total_usd / market_cap
    if ratio > 0.5:
        return 15
    elif ratio > 0.3:
        return 10
    elif ratio > 0.15:
        return 5
    else:
        return 0

def get_capital_raises_score(num_raises_past_year):
    if num_raises_past_year >= 4:
        return 15
    elif num_raises_past_year == 3:
        return 10
    elif num_raises_past_year == 2:
        return 7
    elif num_raises_past_year == 1:
        return 4
    else:
        return 0

def calculate_dilution_pressure_score(atm_capacity, convertibles_and_warrants, total_raises, market_cap, num_raises_past_year=0, red_flags_score=0):
    try:
        if not market_cap or market_cap == 0:
            return None

        atm_score = min(atm_capacity / market_cap * 100, 40)
        convert_score = min(convertibles_and_warrants * 5, 20)
        raise_score = min(total_raises / market_cap * 100, 30)
        frequency_score = min(num_raises_past_year * 5, 10)

        total_score = atm_score + convert_score + raise_score + frequency_score + red_flags_score
        return round(min(total_score, 100), 1)
    except Exception as e:
        print(f"Error in score calculation: {e}")
        return None

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
        st.write(f"${market_cap:,.0f}" if market_cap else "Not available")

        # Cash Runway
        cash, burn = get_cash_and_burn(cik)
        runway = calculate_cash_runway(cash, burn)
        st.subheader("2. Cash Runway")
        if cash and burn:
            st.write(f"Cash: ${cash:,.0f}")
            st.write(f"Monthly Burn: ${burn:,.0f}")
            st.write(f"Runway: {runway:.1f} months")
        else:
            st.write("Cash or burn rate not found.")

        # ATM Offering
        atm, atm_url = get_atm_offering(cik)
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
        atm_capacity = atm if atm else 0
        convertibles_and_warrants = len(instruments) if instruments else 0
        total_raises = sum(entry["amount"] for entry in raises) if raises else 0
        num_raises_past_year = len([
            entry for entry in raises if datetime.strptime(entry["date"], "%Y-%m-%d") > datetime.now() - timedelta(days=365)
        ]) if raises else 0

       # Optional red flag (for future expansion)
        red_flags_score = 0

       # Calculate dilution score
        try:
            score = calculate_dilution_pressure_score(
                atm_capacity,
                convertibles_and_warrants,
                total_raises,
                market_cap,
                num_raises_past_year,
                red_flags_score
            )

            st.subheader("8. Dilution Pressure Score")
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







