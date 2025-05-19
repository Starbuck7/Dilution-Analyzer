import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import warnings
from bs4 import XMLParsedAsHTMLWarning
from datetime import datetime, timedelta
from yahoo_fin import stock_info as si
from sec_edgar_downloader import Downloader

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
    try:
        table = si.get_quote_table(ticker)
        market_cap_str = table.get("Market Cap")
        if not market_cap_str:
            return None
        
        multiplier = 1
        if market_cap_str.endswith("B"):
            multiplier = 1_000_000_000
        elif market_cap_str.endswith("M"):
            multiplier = 1_000_000
        elif market_cap_str.endswith("K"):
            multiplier = 1_000

        return float(market_cap_str[:-1].replace(",", "")) * multiplier
    except Exception as e:
        print(f"[ERROR] Failed to fetch market cap for {ticker}: {e}")
        return None

# -------------------- Module 2: Cash Runway --------------------
def get_cash_and_burn(cik):
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=USER_AGENT)
        if res.status_code != 200:
            return None, None

        filings = res.json().get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])
        dates = filings.get("filingDate", [])

        for i, form in enumerate(forms):
            if form not in ["10-Q", "10-K"]:
                continue
            if i >= len(accessions) or i >= len(docs):
                continue

            accession = accessions[i].replace("-", "")
            doc = docs[i]
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(html_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")

            # Look for cash
            cash_match = re.search(
                r"(?i)(cash and cash equivalents|cash position)\s*[\$:]*\s*([0-9]{1,3}(?:[0-9\s]*[0-9]))", text)
            cash = int(cash_match.group(2).replace(" ", "")) if cash_match else None

            # Look for burn rate: sometimes referred to as "net cash used in operating activities"
            burn_match = re.search(
                r"(?i)(net cash used in operating activities)[^\$0-9\-]{0,20}[\$:]*\s*\(?-?([0-9]{1,3}(?:[0-9\s]*[0-9]))\)?", text)
            burn = int(burn_match.group(2).replace(" ", "")) / 3 if burn_match else None  # quarterly to monthly

            if cash or burn:
                return cash, abs(burn) if burn else None

        return None, None
    except Exception as e:
        print(f"Error in get_cash_and_burn: {e}")
        return None, None

def calculate_cash_runway(cash, burn):
    if cash and burn:
        return cash / burn
    return None

# -------------------- Module 3: ATM Offering Capacity --------------------
def get_atm_offering(cik):
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=USER_AGENT)
        if res.status_code != 200:
            return None, None

        filings = res.json().get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        docs = filings.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form not in ["S-3", "S-1", "8-K"]:
                continue
            if i >= len(accessions) or i >= len(docs):
                continue

            accession = accessions[i].replace("-", "")
            doc = docs[i]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            html = requests.get(filing_url, headers=USER_AGENT).text
            text = BeautifulSoup(html, "lxml").get_text().replace(",", "")

            atm_patterns = [
                r"(?i)at-the-market\s+offering\s+.*?\$([0-9]{2,})",
                r"(?i)offering\s+under\s+the\s+atm\s+program.*?\$([0-9]{2,})",
                r"(?i)may\s+sell\s+up\s+to\s+\$([0-9]{2,})\s+.*?(?:shares|stock|securities)?",
                r"(?i)maximum\s+aggregate\s+offering\s+price[^0-9]{0,20}\$([0-9]{2,})"
            ]

            for pattern in atm_patterns:
                match = re.search(pattern, text)
                if match:
                    atm_amount = int(match.group(1))
                    return atm_amount, filing_url
        return None, None
    except Exception as e:
        print(f"Error in get_atm_offering: {e}")
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
        dl = Downloader("/tmp/sec")
        
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


#-------------MODULE 9: CALCULATIONG DILUTION PRESSURE SCORE ------------
def get_atm_capacity_score(atm_capacity_usd, market_cap):
    if not market_cap or not atm_capacity_usd:
        return 10  # neutral if missing

    ratio = atm_capacity_usd / market_cap
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

def get_convertibles_score(convertibles_usd, market_cap):
    if not market_cap or not convertibles_usd:
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

def calculate_dilution_pressure_score(
    atm_capacity_usd,
    authorized_shares,
    outstanding_shares,
    convertibles_usd,
    num_raises_past_year,
    cash_runway_months,
    market_cap
):
    score = 0

    # ATM Offering Capacity Scoring
    if atm_capacity_usd and market_cap:
        ratio = atm_capacity_usd / market_cap
        if ratio > 0.75:
            score += 25
        elif ratio > 0.5:
            score += 20
        elif ratio > 0.25:
            score += 15
        elif ratio > 0.1:
            score += 10
        else:
            score += 5

    # Authorized - Outstanding (Dilution Potential)
    if authorized_shares and outstanding_shares and market_cap:
        available = authorized_shares - outstanding_shares
        est_value = available * 0.5  # Assuming $0.50 per share
        dilution_ratio = est_value / market_cap
        if dilution_ratio > 1:
            score += 25
        elif dilution_ratio > 0.5:
            score += 20
        elif dilution_ratio > 0.25:
            score += 10
        else:
            score += 5

    # Convertibles & Warrants
    if convertibles_usd and market_cap:
        convert_ratio = convertibles_usd / market_cap
        if convert_ratio > 0.5:
            score += 20
        elif convert_ratio > 0.25:
            score += 10
        elif convert_ratio > 0.1:
            score += 5

    # Capital Raises in Past Year
    if num_raises_past_year >= 4:
        score += 15
    elif num_raises_past_year == 3:
        score += 10
    elif num_raises_past_year == 2:
        score += 7
    elif num_raises_past_year == 1:
        score += 4

    # Cash Runway
    if cash_runway_months is not None:
        if cash_runway_months < 3:
            score += 15
        elif cash_runway_months < 6:
            score += 10
        elif cash_runway_months < 12:
            score += 5

    return min(score, 100)



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
        st.write(f"Market Cap: {market_cap}")
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
        available_dilution_shares = (authorized - outstanding) if authorized and outstanding else 0
        convertible_total_usd = sum(instruments.values()) if isinstance(instruments, dict) else 0
        num_raises_past_year = len([
            entry for entry in raises
            if datetime.strptime(entry["date"], "%Y-%m-%d") > datetime.now() - timedelta(days=365)
        ]) if raises else 0

        # Optional red flag (for future expansion)
        red_flags_score = 0

       # Calculate dilution score
        try:
            score = calculate_dilution_pressure_score(
                atm_capacity_usd=atm,  # from get_atm_offering()
                authorized_shares=authorized,  # from get_authorized_shares()
                outstanding_shares=outstanding,  # from get_outstanding_shares()
                convertibles_usd=convertible_total_usd,  # estimate total from instruments
                historical_capital_raises=num_raises_past_year,  # len of filtered raises
                cash_runway_months=runway,  # from calculate_cash_runway()
                market_cap=market_cap  # from get_market_cap()
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







