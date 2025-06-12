import requests
from bs4 import BeautifulSoup
import re
import os
import json
import logging
import warnings
from bs4 import XMLParsedAsHTMLWarning
from functools import lru_cache

USER_AGENT = {"User-Agent": "DilutionAnalyzerBot/1.0 (ASHLEYMCGAVERN@YAHOO.COM)"}

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_filings_html(cik, forms=None, max_results=10):
    cik = str(cik).lstrip('0')
    base_url = f"https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude"
    filings = []
    page_url = base_url
    session = requests.Session()
    session.headers.update(USER_AGENT)
    while page_url and len(filings) < max_results:
        resp = session.get(page_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find('table', class_='table')
        if not table:
            break
        tbody = table.find('tbody')
        if not tbody:
            break
        for row in tbody.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) < 5:
                continue
            form = cols[0].text.strip()
            if forms and form not in forms:
                continue
            accession = cols[2].text.strip().replace("-", "")
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
        next_link = soup.find('a', string='Next')
        if next_link and next_link.get('href'):
            page_url = "https://www.sec.gov" + next_link['href']
        else:
            page_url = None
    return filings

def fetch_filings_json(cik, forms=None, max_results=10):
    cik = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=USER_AGENT)
    if resp.status_code != 200:
        return []
    data = resp.json()
    filings = []
    recent = data.get("filings", {}).get("recent", {})
    for i, form in enumerate(recent.get("form", [])):
        if forms and form not in forms:
            continue
        try:
            accession = recent["accessionNumber"][i].replace("-", "")
            file_name = recent["primaryDocument"][i]
            date_filed = recent["filingDate"][i]
            filings.append({
                "form": form,
                "accession": accession,
                "file_name": file_name,
                "date_filed": date_filed,
                "source": "recent"
            })
            if len(filings) >= max_results:
                return filings
        except (IndexError, KeyError):
            continue
    for f in data.get("filings", {}).get("files", []):
        f_url = "https://data.sec.gov" + f["name"]
        try:
            f_resp = requests.get(f_url, headers=USER_AGENT)
            f_resp.raise_for_status()
            f_data = f_resp.json()
            for i, form in enumerate(f_data.get("form", [])):
                if forms and form not in forms:
                    continue
                try:
                    accession = f_data["accessionNumber"][i].replace("-", "")
                    file_name = f_data["primaryDocument"][i]
                    date_filed = f_data["filingDate"][i]
                    filings.append({
                        "form": form,
                        "accession": accession,
                        "file_name": file_name,
                        "date_filed": date_filed,
                        "source": "historical"
                    })
                    if len(filings) >= max_results:
                        return filings
                except (IndexError, KeyError):
                    continue
        except Exception:
            continue
    return filings

def get_latest_filing(cik, forms=None):
    filings = fetch_filings_html(cik, forms=forms, max_results=1)
    if filings:
        return filings[0]
    filings = fetch_filings_json(cik, forms=forms, max_results=1)
    if filings:
        return filings[0]
    raise ValueError(f"No {', '.join(forms) if forms else 'filings'} found in SEC HTML or JSON for CIK {cik}.")

def get_all_filings(cik, forms=None, max_results=10):
    filings = fetch_filings_html(cik, forms=forms, max_results=max_results)
    if filings:
        return filings
    return fetch_filings_json(cik, forms=forms, max_results=max_results)

@lru_cache(maxsize=100)
def get_cik_from_ticker(ticker):
    try:
        cik = get_cik_from_ticker(ticker)
    except ValueError as e:
        st.error(str(e))
        st.stop()
        # SEC company search URL
        url = f"https://www.sec.gov/edgar/browse/?CIK={ticker}&owner=exclude"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            m = re.search(r"CIK=(\d{10})", resp.text)
            if not m:
                # Try 8 or 9 digit CIK, pad to 10
                m = re.search(r"CIK=(\d+)", resp.text)
            if m:
                cik = m.group(1).zfill(10)
                return cik
        # Alternate fallback: try the new SEC search api (2024+)
        alt_url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
        resp2 = requests.get(alt_url, headers={"User-Agent": "Mozilla/5.0"})
        if resp2.status_code == 200:
            # This only works if user passes CIK as ticker
            return ticker.zfill(10)
    except Exception as e:
        pass

    raise ValueError(f"CIK not found for ticker: {ticker}")

# If your mapping is loaded via a file or global variable, adjust get_ticker_cik_mapping() accordingly.
def fetch_filing_html(cik, accession, file_name):
    """
    Fetches the raw HTML text for a specific SEC filing document.
    Args:
        cik (str): Central Index Key (10 digits, zero-padded)
        accession (str): Accession number without dashes
        file_name (str): Document file name (e.g., '10q.htm')
    Returns:
        html (str): The raw HTML of the document
    """
    SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{file_name}"
    url = SEC_ARCHIVES_URL.format(
        cik=str(int(cik)),  # Remove leading zeros for SEC path
        accession_nodash=accession,
        file_name=file_name
    )
    resp = requests.get(url, headers=USER_AGENT)
    resp.raise_for_status()
    return resp.text
