import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from sec_utils import (
    get_cik_from_ticker,
    fetch_filing_html,
    get_latest_filing,
    get_all_filings,
)
from bs4 import BeautifulSoup
import re

# ===================== Helpers and Patterns =====================
def _fmt(val):
    if val is None:
        return "Not found"
    elif isinstance(val, (int, float)):
        return f"{val:,.0f}"
    else:
        return str(val)

def _parse_market_cap_str(market_cap_str):
    try:
        multipliers = {'B': 1e9, 'M': 1e6, 'K': 1e3}
        if market_cap_str[-1] in multipliers:
            return int(float(market_cap_str[:-1]) * multipliers[market_cap_str[-1]])
        else:
            return int(market_cap_str.replace(',', ''))
    except Exception:
        return None

def get_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        mc = info.get('marketCap')
        if mc: return mc
    except Exception:
        pass
    # Try yahoo_fin fallback if needed (not included for brevity)
    return None

# ===================== Module 2: Cash Runway =====================
CASH_PATTERNS = [
    r"cash and cash equivalents",
    r"total cash and cash equivalents",
    r"cash & cash equivalents",
    r"cash and short-term investments",
    r"total cash",
    r"cash, cash equivalents",
]
BURN_PATTERNS = [
    r"net cash (provided by|used in) operating activities",
    r"net cash used in operating activities",
    r"net cash provided by operating activities",
    r"net cash (provided by|used in) operations",
    r"net cash (used in|provided by) operations",
    r"net cash (used in|provided by) operating",
]
PERIOD_PATTERNS = [
    r"For the (three|six|nine|twelve)[- ]month[s]? ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
    r"As of\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})"
]

def extract_numeric_cell(cell):
    match = re.search(r"\$?[\(\-]?\d[\d,]*\.?\d*", cell)
    if match:
        try:
            val = match.group().replace("$", "").replace(",", "").replace("(", "-").replace(")", "")
            return int(float(val))
        except Exception:
            return None
    return None

def extract_cash_and_burn_from_html(html):
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")

    period_str, period_months = None, None
    for pat in PERIOD_PATTERNS:
        period_match = re.search(pat, text, re.IGNORECASE)
        if period_match:
            period_str = period_match.group(0)
            if "month" in period_str.lower():
                months_map = {"three": 3, "six": 6, "nine": 9, "twelve": 12}
                for name, val in months_map.items():
                    if name in period_match.group(1).lower():
                        period_months = val
                        break
            break

    # Extract cash value (try tables first, then text)
    cash_val = None
    for label_pat in CASH_PATTERNS:
        label_cells = soup.find_all(string=re.compile(label_pat, re.I))
        for cell in label_cells:
            row = cell.find_parent("tr")
            if row:
                tds = row.find_all("td")
                for td in tds[::-1]:
                    val = extract_numeric_cell(td.get_text())
                    if val is not None:
                        cash_val = val
                        break
            if cash_val is not None:
                break
        if cash_val is not None:
            break
    if cash_val is None:
        for pat in CASH_PATTERNS:
            for line in text.split("\n"):
                if re.search(pat, line, re.I):
                    val = extract_numeric_cell(line)
                    if val is not None:
                        cash_val = val
                        break
            if cash_val is not None:
                break

    # Extract net cash used in operating activities
    net_cash_used = None
    for label_pat in BURN_PATTERNS:
        label_cells = soup.find_all(string=re.compile(label_pat, re.I))
        for cell in label_cells:
            row = cell.find_parent("tr")
            if row:
                tds = row.find_all("td")
                for td in tds[::-1]:
                    val = extract_numeric_cell(td.get_text())
                    if val is not None:
                        net_cash_used = val
                        break
            if net_cash_used is not None:
                break
        if net_cash_used is not None:
            break
    if net_cash_used is None:
        for pat in BURN_PATTERNS:
            for line in text.split("\n"):
                if re.search(pat, line, re.I):
                    val = extract_numeric_cell(line)
                    if val is not None:
                        net_cash_used = val
                        break
            if net_cash_used is not None:
                break

    return period_str, period_months, cash_val, net_cash_used

def get_cash_runway_for_ticker(ticker):
    try:
        cik = get_cik_from_ticker(ticker)
        for form in ["10-Q", "10-K"]:
            filing = get_latest_filing(cik, forms=(form,))
            if not filing:
                continue
            html = fetch_filing_html(cik, filing["accession"], filing.get("file_name") or filing.get("doc_link").split("/")[-1])
            period_str, period_months, cash_val, net_cash_used = extract_cash_and_burn_from_html(html)
            if (cash_val is not None) and (net_cash_used is not None):
                break
        burn_rate = None
        runway = None
        if cash_val is not None and net_cash_used is not None and period_months:
            burn_rate = abs(net_cash_used) / period_months
            runway = cash_val / burn_rate if burn_rate else None
        return {
            "period_string": period_str,
            "period_months": period_months,
            "cash": cash_val,
            "net_cash_used": net_cash_used,
            "burn_rate": round(burn_rate, 2) if burn_rate else None,
            "runway_months": round(runway, 2) if runway else None,
            "cik": cik,
            "accession": filing["accession"],
            "file_name": filing.get("file_name") or filing.get("doc_link").split("/")[-1],
            "form": filing["form"]
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

# ===================== Module 3: ATM Offering Capacity =====================
ATM_PHRASES = [
    r"at[-\s]?the[-\s]?market(?:\s+offering)?",
    r"committed equity financing",
    r"equity (?:purchase|distribution|line|sales) agreement",
    r"purchase agreement with [A-Za-z0-9, \.\-&]+",
    r"committed to purchase up to \$?\d+(\.\d+)?\s*(million|billion)?",
    r"aggregate offering price of \$?\d+(\.\d+)?\s*(million|billion)?",
    r"equity line of credit",
    r"shelf (?:offering|registration)",
    r"sales agreement with [A-Za-z0-9, \.\-&]+",
]
AMOUNT_REGEX = r"(?:up to|aggregate offering price of|maximum aggregate offering price of|commitment amount of)\s*\$?([\d,.]+)\s*(million|billion)?"

def get_atm_offering(cik, lookback=10):
    atm_results = []
    filings = get_all_filings(cik, forms=['S-1', 'S-3', '424B5', '8-K', 'F-3', 'F-1'], max_results=lookback)
    for filing in filings:
        filing_url = filing.get("primary_doc_url") or filing.get("filing_url") or filing.get("url")
        filing_type = filing.get("form")
        filing_date = filing.get("filed")
        try:
            html = fetch_filing_html(cik, filing["accession"], filing.get("file_name") or filing.get("doc_link").split("/")[-1])
        except Exception:
            continue
        if not html:
            continue
        text = html.lower()
        found = False
        for phrase in ATM_PHRASES:
            for match in re.finditer(phrase, text, re.IGNORECASE):
                context_window = text[max(0, match.start()-100):match.end()+500]
                amt_match = re.search(AMOUNT_REGEX, context_window, re.IGNORECASE)
                amount = None
                if amt_match:
                    amt = amt_match.group(1).replace(',', '')
                    scale = amt_match.group(2)
                    try:
                        amount = float(amt)
                        if scale:
                            scale = scale.lower()
                            if scale.startswith("b"):
                                amount *= 1e9
                            elif scale.startswith("m"):
                                amount *= 1e6
                    except Exception:
                        pass
                counterparty = None
                counterparty_match = re.search(r"(?:agreement with|sales agreement with)\s+([A-Za-z0-9, \.\-&]+)", context_window)
                if counterparty_match:
                    counterparty = counterparty_match.group(1).strip()
                atm_results.append({
                    "amount_usd": amount,
                    "filing_type": filing_type,
                    "filing_date": filing_date,
                    "counterparty": counterparty,
                    "phrase": match.group(),
                    "context": context_window[:400] + "..." if len(context_window) > 400 else context_window,
                    "filing_url": filing_url
                })
                found = True
        if found:
            break
    if atm_results:
        best = max(atm_results, key=lambda x: x["amount_usd"] or 0)
        return best["amount_usd"], best["filing_url"], best
    return None, None, None

# ===================== Module 4: Offering Ability =====================
AUTHORIZED_PATTERNS = [
    r"authorized[\s\-]*shares[^:\d]*[:\s]*([\d,]+)",
    r"number of shares authorized[^:\d]*[:\s]*([\d,]+)",
    r"common stock.*?authorized[^:\d]*[:\s]*([\d,]+)",
    r"shares of common stock authorized[^:\d]*[:\s]*([\d,]+)",
]
OUTSTANDING_PATTERNS = [
    r"outstanding[\s\-]*shares[^:\d]*[:\s]*([\d,]+)",
    r"number of shares outstanding[^:\d]*[:\s]*([\d,]+)",
    r"common stock.*?outstanding[^:\d]*[:\s]*([\d,]+)",
    r"shares of common stock outstanding[^:\d]*[:\s]*([\d,]+)",
]

def extract_first_match(text, patterns):
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            try:
                return int(val)
            except Exception:
                continue
    return None

def get_authorized_shares(cik):
    try:
        filing = get_latest_filing(cik, forms=("10-K", "10-Q", "DEF 14A", "S-1", "S-3"))
        html = fetch_filing_html(cik, filing["accession"], filing.get("file_name") or filing.get("doc_link").split("/")[-1])
        text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
        val = extract_first_match(text, AUTHORIZED_PATTERNS)
        if val:
            return val
        return None
    except Exception:
        return None

def get_outstanding_shares(cik):
    try:
        filing = get_latest_filing(cik, forms=("10-K", "10-Q", "DEF 14A", "S-1", "S-3"))
        html = fetch_filing_html(cik, filing["accession"], filing.get("file_name") or filing.get("doc_link").split("/")[-1])
        text = BeautifulSoup(html, "lxml").get_text().replace(",", "")
        val = extract_first_match(text, OUTSTANDING_PATTERNS)
        if val:
            return val
        return None
    except Exception:
        return None

def get_shelf_registered_shares(cik, num_filings=10):
    try:
        filing_types = ("S-3", "S-1", "424B3")
        filings = get_all_filings(cik, forms=filing_types, max_results=num_filings)
        all_text = ""
        for filing in filings:
            file_name = filing.get("file_name") or filing.get("doc_link").split("/")[-1]
            html = fetch_filing_html(cik, filing["accession"], file_name)
            text = BeautifulSoup(html, "lxml").get_text(separator="\n")
            all_text += "\n" + text
        dollar_matches = re.findall(
            r"offer(?:ing)?(?: and sell)? (?:up to|of up to)?\s*\$?([\d,.]+)\s*(million|billion)?",
            all_text, re.IGNORECASE)
        amounts = []
        for num_str, magnitude in dollar_matches:
            try:
                num = float(num_str.replace(",", "").strip())
                if magnitude:
                    if magnitude.lower() == "million":
                        num *= 1_000_000
                    elif magnitude.lower() == "billion":
                        num *= 1_000_000_000
                amounts.append(num)
            except Exception:
                continue
        share_matches = re.findall(
            r"offer(?:ing)?(?: and sell)? (?:up to|of up to)?\s*([\d,.]+)\s*(shares|common stock)?",
            all_text, re.IGNORECASE)
        for num_str, _ in share_matches:
            try:
                num = float(num_str.replace(",", "").strip())
                amounts.append(num)
            except Exception:
                continue
        if amounts:
            return max(amounts)
        return None
    except Exception:
        return None

def get_public_float(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        public_float = info.get('floatShares')
        return public_float
    except Exception:
        return None

def get_atm_capacity(cik):
    try:
        atm_usd, _, _ = get_atm_offering(cik)
        return atm_usd
    except Exception:
        return None

def estimate_offering_ability(cik, ticker):
    authorized = get_authorized_shares(cik)
    outstanding = get_outstanding_shares(cik)
    available_shares = authorized - outstanding if (authorized is not None and outstanding is not None) else None
    atm_capacity = get_atm_capacity(cik)
    shelf_usd = get_shelf_registered_shares(cik)
    float_shares = get_public_float(ticker)
    return {
        "Authorized Shares": authorized,
        "Outstanding Shares": outstanding,
        "Available Shares": available_shares,
        "ATM Capacity": atm_capacity,
        "Shelf Registered Shares": shelf_usd,
        "Public Float": float_shares
    }

# ===================== Module 5: Convertibles and Warrants =====================
CONVERTIBLE_PATTERNS = [
    r"convertible (?:note|debt|bond|security|securities|preferred stock)[^$]*?\$?([\d,.]+)(?:[^$]*?conversion price[^$]*?\$?([\d,.]+))?",
    r"aggregate principal amount of convertible[^$]*?\$?([\d,.]+)"
]
WARRANT_PATTERNS = [
    r"warrant[s]?(?: to purchase)?[^$]*?([\d,.]+) shares[^$]*?(?:exercise price[^$]*?\$?([\d,.]+))?",
    r"warrants outstanding[^$]*?([\d,.]+)",
    r"exercise price of (?:the )?warrants[^$]*?\$?([\d,.]+)"
]

def extract_convertibles_and_warrants(html):
    results = []
    text = BeautifulSoup(html, "lxml").get_text(separator="\n")
    for pat in CONVERTIBLE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            amt = m.group(1)
            price = m.group(2) if (len(m.groups()) > 1) else None
            try:
                amt_val = float(amt.replace(",", ""))
                price_val = float(price.replace(",", "")) if price else None
                results.append({'type': 'convertible', 'amount': amt_val, 'conversion_price': price_val, 'context': text[m.start():m.end()+200]})
            except Exception:
                continue
    for pat in WARRANT_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            amt = m.group(1)
            price = m.group(2) if (len(m.groups()) > 1) else None
            try:
                amt_val = float(amt.replace(",", ""))
                price_val = float(price.replace(",", "")) if price else None
                results.append({'type': 'warrant', 'amount': amt_val, 'exercise_price': price_val, 'context': text[m.start():m.end()+200]})
            except Exception:
                continue
    return results

def get_convertibles_and_warrants_with_amounts(cik, ticker, filings_to_check=5):
    filings = get_all_filings(cik, forms=['10-Q', '10-K', 'S-1', 'S-3', '8-K', 'F-1', 'F-3', '424B3'], max_results=filings_to_check)
    all_results = []
    for filing in filings:
        html = fetch_filing_html(cik, filing["accession"], filing.get("file_name") or filing.get("doc_link").split("/")[-1])
        results = extract_convertibles_and_warrants(html)
        all_results.extend(results)
    return all_results

def rate_convertible_warrant_risk(ticker, results, float_shares):
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get('regularMarketPrice')
        if not price or not float_shares:
            return "Unknown"
        at_money = 0
        for item in results:
            conv_price = item.get('conversion_price') or item.get('exercise_price')
            amt = item.get('amount', 0)
            if conv_price is not None:
                if abs(conv_price - price)/price <= 0.1:
                    at_money += amt
        ratio = at_money / float_shares if float_shares else 0
        if at_money == 0:
            return "None"
        elif ratio > 0.2:
            return "Large"
        elif ratio > 0.05:
            return "Moderate"
        else:
            return "Minimal"
    except Exception:
        return "Unknown"

def get_convertibles_score(instruments, ticker, float_shares=None):
    if not instruments or len(instruments) == 0:
        return 0, "None", {"at_money_shares": 0, "float_shares": float_shares}
    if float_shares is None:
        try:
            stock = yf.Ticker(ticker)
            float_shares = stock.info.get('floatShares')
        except Exception:
            float_shares = None
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get('regularMarketPrice')
    except Exception:
        price = None
    if price is None or float_shares is None or float_shares == 0:
        return 0, "Unknown", {"at_money_shares": 0, "float_shares": float_shares, "price": price}
    at_money = 0
    for item in instruments:
        conv_price = item.get('conversion_price') or item.get('exercise_price')
        amt = item.get('amount', 0)
        if conv_price is not None and amt:
            try:
                if abs(conv_price - price)/price <= 0.1:
                    at_money += amt
            except Exception:
                continue
    ratio = at_money / float_shares if float_shares else 0
    if at_money == 0:
        score = 0
        label = "None"
    elif ratio > 0.2:
        score = 15
        label = "Large"
    elif ratio > 0.05:
        score = 8
        label = "Moderate"
    else:
        score = 3
        label = "Minimal"
    return score, label, {
        "at_money_shares": at_money,
        "float_shares": float_shares,
        "price": price,
        "ratio": ratio
    }

# ===================== Module 6: Historical Capital Raises =====================
RAISE_PATTERNS = [
    r"(?:public|registered direct|private|direct|at[-\s]?the[-\s]?market|PIPE|equity) (?:offering|placement|financing)",
    r"securities purchase agreement",
    r"issued and sold[^.]+shares",
    r"gross proceeds of \$?([\d,.]+)(?:\s*(million|billion))?",
    r"aggregate gross proceeds of \$?([\d,.]+)(?:\s*(million|billion))?",
    r"sale of convertible notes",
    r"committed equity facility",
    r"pursuant to a prospectus supplement",
]

def extract_capital_raises_from_html(html):
    text = BeautifulSoup(html, "lxml").get_text(separator="\n")
    hits = []
    for pat in RAISE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            amount = None
            if m.lastindex and m.lastindex >= 1:
                amt_str = m.group(m.lastindex-1)
                magnitude = m.group(m.lastindex) if m.lastindex > 1 else None
                if amt_str:
                    try:
                        amount = float(amt_str.replace(',', ''))
                        if magnitude:
                            magnitude = magnitude.lower()
                            if magnitude.startswith('m'):
                                amount *= 1_000_000
                            elif magnitude.startswith('b'):
                                amount *= 1_000_000_000
                    except Exception:
                        amount = None
            context = text[max(0, m.start()-100):m.end()+200]
            hits.append({
                "pattern": pat,
                "context": context.strip(),
                "amount": amount
            })
    return hits

def get_historical_capital_raises(cik, months=18, filings_to_check=20):
    today = datetime.utcnow()
    cutoff = today - timedelta(days=int(months*30.44))
    filings = get_all_filings(
        cik,
        forms=['8-K', 'S-1', 'S-3', 'S-8', 'F-1', 'F-3', '424B3', '424B5', '10-Q', '10-K'],
        max_results=filings_to_check
    )
    results = []
    for filing in filings:
        try:
            filed_date = datetime.strptime(filing.get("filed", ""), "%Y-%m-%d")
            if filed_date < cutoff:
                continue
            html = fetch_filing_html(
                cik,
                filing["accession"],
                filing.get("file_name") or filing.get("doc_link").split("/")[-1]
            )
            matches = extract_capital_raises_from_html(html)
            for m in matches:
                m['filing_date'] = filed_date.date()
                m['filing_type'] = filing.get("form")
                m['filing_url'] = filing.get("primary_doc_url") or filing.get("filing_url") or filing.get("url")
                results.append(m)
        except Exception:
            continue
    return results

# ===================== Module 7: Dilution Pressure Score =====================
def get_atm_capacity_score(atm_capacity_usd, market_cap):
    if not atm_capacity_usd or not market_cap:
        return 10
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

def get_authorized_vs_outstanding_score(authorized, outstanding, market_cap=None):
    if authorized is None or outstanding is None or authorized == 0:
        return 0
    pct = outstanding / authorized
    if pct >= 0.9:
        return 25
    elif pct >= 0.75:
        return 20
    elif pct >= 0.5:
        return 10
    elif pct >= 0.25:
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
    def safe_value(value, default=0):
        return value if value is not None else default

    total_score = 0
    total_score += get_atm_capacity_score(safe_value(atm_capacity_usd), safe_value(market_cap))
    total_score += get_authorized_vs_outstanding_score(safe_value(authorized_shares), safe_value(outstanding_shares), safe_value(market_cap))
    convertible_score, _, _ = get_convertibles_score(convertibles, "", None)
    total_score += convertible_score
    total_score += get_capital_raises_score(safe_value(capital_raises_past_year))
    total_score += get_cash_runway_score(safe_value(cash_runway))
    return min(total_score, 100)

# ===================== Streamlit App =====================
st.title("Stock Analysis Dashboard")
st.markdown("Analyze dilution and financial health based on SEC filings.")

ticker = st.text_input("Enter a stock ticker (e.g., SYTA)", "").strip().upper()

if ticker:
    cik = get_cik_from_ticker(ticker)
    # Market Cap
    market_cap = get_market_cap(ticker)
    st.subheader("1. Market Cap")
    st.write(f"Market Cap: ${market_cap:,.0f}" if market_cap is not None else "Market Cap: Not available")

    # Cash Runway
    st.header("2. Cash Runway")
    with st.spinner("Analyzing cash runway..."):
        cash_result = get_cash_runway_for_ticker(ticker)
    if "error" in cash_result:
        st.error(f"Error: {cash_result['error']}")
        runway_months = None
    else:
        st.success(f"Latest {cash_result['form']} for {ticker} (CIK {cash_result['cik']}):")
        st.write(f"- Accession: {cash_result['accession']}")
        st.write(f"- File: {cash_result['file_name']}")
        if cash_result['period_string']:
            st.write(f"**Reporting period:** {cash_result['period_string']} ({cash_result['period_months']} months)")
        if cash_result['cash'] is not None:
            st.write(f"**Cash and cash equivalents:** ${cash_result['cash']:,}")
        if cash_result['net_cash_used'] is not None:
            st.write(f"**Net cash used in operating activities:** ${cash_result['net_cash_used']:,}")
        if cash_result['burn_rate'] is not None:
            st.write(f"**Burn rate:** ${cash_result['burn_rate']:,} per month")
        if cash_result['runway_months'] is not None:
            st.write(f"### 🚦 Estimated Cash Runway: **{cash_result['runway_months']} months**")
        else:
            st.warning("Could not estimate cash runway (missing data).")
        runway_months = cash_result.get('runway_months')

    # ATM Offering Capacity
    atm, atm_url, atm_details = get_atm_offering(cik, lookback=10)
    st.subheader("3. ATM / Committed Equity Facility Capacity")
    if atm_details:
        st.write(f"**ATM/Equity Facility Capacity:** ${atm:,.0f}" if atm else "ATM/Equity Facility found (amount not parsed)")
        if atm_details["counterparty"]:
            st.write(f"**Counterparty:** {atm_details['counterparty']}")
        st.write(f"**Filing Type:** {atm_details['filing_type']}  |  **Date:** {atm_details['filing_date']}")
        st.markdown(f"[Source Document]({atm_url})")
        with st.expander("Context excerpt from filing"):
            st.write(atm_details["context"])
    else:
        st.write("No ATM/Equity Facility or Committed Equity Financing found in recent filings.")

    # Offering Ability
    data = estimate_offering_ability(cik, ticker)
    st.subheader("4. Offering Ability")
    st.write(f"Authorized Shares: {_fmt(data['Authorized Shares'])}")
    st.write(f"Outstanding Shares: {_fmt(data['Outstanding Shares'])}")
    if data['Available Shares'] is not None:
        st.write(f"Available Shares: {_fmt(data['Available Shares'])}")
    else:
        st.write("Available Shares: N/A (missing data)")
    st.write(f"ATM Capacity: {_fmt(data['ATM Capacity'])}")
    st.write(f"Shelf Registered Shares: {_fmt(data['Shelf Registered Shares'])}")
    st.write(f"Public Float: {_fmt(data['Public Float'])}")
    authorized = data['Authorized Shares']
    outstanding = data['Outstanding Shares']
    float_shares = data['Public Float']

    # Convertibles & Warrants
    instruments = get_convertibles_and_warrants_with_amounts(cik, ticker)
    risk = rate_convertible_warrant_risk(ticker, instruments, float_shares)
    st.subheader("5. Convertibles and Warrants")
    if not instruments:
        st.write("None found in recent filings.")
    else:
        for item in instruments:
            st.write(f"Type: {item['type'].capitalize()}, Amount: {item['amount']:,.0f}, Price: ${item.get('conversion_price') or item.get('exercise_price')}")
    st.write(f"**Dilution Risk from Convertibles/Warrants:** {risk}")

    # Historical Capital Raises
    capital_raises = get_historical_capital_raises(cik)
    num_raises = len(capital_raises)
    st.subheader("6. Historical Capital Raises (last 18 months)")
    if not capital_raises:
        st.write("No capital raises found in the last 18 months.")
    else:
        st.write(f"Capital raises found: {num_raises}")
        for raise_event in capital_raises:
            st.write(f"- Date: {raise_event['filing_date']} | Amount: {raise_event['amount'] or 'N/A'} | Filing: {raise_event['filing_type']}")
            with st.expander("Context"):
                st.write(raise_event["context"])

    # Dilution Pressure Score
    st.subheader("7. Dilution Pressure Score")
    st.caption("Combines cash runway, ATM capacity, dilution ability, and more to assess dilution risk.")
    try:
        score = calculate_dilution_pressure_score(
            atm_capacity_usd=atm,
            authorized_shares=authorized,
            outstanding_shares=outstanding,
            convertibles=instruments,
            capital_raises_past_year=num_raises,
            cash_runway=runway_months,
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
else:
    st.warning("Please enter a stock ticker to begin analysis.")
