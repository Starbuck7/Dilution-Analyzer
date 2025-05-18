import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
import warnings
from bs4 import XMLParsedAsHTMLWarning

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
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
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
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
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

                match = re.search(r"(?:common\s+stock\s+)?outstanding\s+(?:shares|stock)[^0-9]{0,20}([0-9]{3,})", text, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        return None
        # Expanded regex to match various phrasings
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
    except Exception as e:
        print(f"Error in get_outstanding_shares: {e}")
        return None
        
# -------------------- Module 5: Convertibles and Warrants --------------------
def get_convertibles_and_warrants(cik):
    filings = res.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    docs = filings.get("primaryDocument", [])
    dates = filings.get("filingDate", [])

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=USER_AGENT).json()
    instruments = []
    for i, form in enumerate(forms):
        if i >= len(accessions) or i >= len(docs) or i >= len(dates):
            continue  # Skip incomplete entries

        if form in [...]:
            accession = accessions[i].replace("-", "")
            doc = docs[i]
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
    import math
    docs = filings.get("primaryDocument", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=USER_AGENT)
    if res.status_code != 200:
        return None
    data = res.json()
    filings = data.get("filings", {}).get("recent", {})
    capital_raises = []

    patterns = [
        r"offering (?:of|for).*?\$?([0-9.,]+)\s?(million|thousand|billion)?",
        r"aggregate proceeds.*?\$?([0-9.,]+)\s?(million|thousand|billion)?",
        r"gross proceeds.*?\$?([0-9.,]+)\s?(million|thousand|billion)?",
        r"sale of.*?\$?([0-9.,]+)\s?(million|thousand|billion)?",
    ]

    for i, form in enumerate(filings.get("form", [])):
        if form in ["S-1", "S-3", "424B5", "8-K"]:
            if i >= len(docs) or i >= len(accessions) or i >= len(dates):
                continue  # Skip if any required field is missing

            accession = accessions[i].replace("-", "")
            doc = docs[i]
            filing_date = dates[i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html_res = requests.get(html_url, headers=USER_AGENT)
            if html_res.status_code != 200:
                continue
            soup = BeautifulSoup(xml_data, "xml")
            text = soup.get_text().replace(",", "").lower()

            found_amount = None
            for pattern in patterns:
                match = re.search(pattern, text)
                if match and match.group(1):
                    try:
                        amount = float(match.group(1).replace(",", ""))
                        unit = match.group(2)
                        if unit == "billion":
                            amount *= 1_000_000_000
                        elif unit == "million":
                            amount *= 1_000_000
                        elif unit == "thousand":
                            amount *= 1_000
                        found_amount = amount
                        break
                    except (ValueError, TypeError):
                        continue

            if found_amount:
                capital_raises.append({
                    "form": form,
                    "date": filing_date,
                    "amount": found_amount,
                    "url": html_url
                })

    return capital_raises if capital_raises else None


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

