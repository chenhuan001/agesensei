"""Fetch full-text papers using Selenium headless Chrome.

Targets papers that only have abstracts (<3KB) in our cache.
Supports: bioRxiv, Nature, Cell, Science, PMC, PNAS, eLife.
"""
import os
import sys
import time
import json
import re
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

PAPERS_DIR = Path(__file__).parent.parent / "artifacts" / "papers_for_mempalace"
PROGRESS_FILE = Path(__file__).parent.parent / "artifacts" / "selenium_fetch_progress.json"


def create_driver():
    """Create headless Chrome driver."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    # Disable images for speed
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def extract_biorxiv_text(driver, doi):
    """Fetch full text from bioRxiv/medRxiv."""
    url = f"https://www.biorxiv.org/content/{doi}v1.full"
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.article"))
        )
        # Get main article content
        article = driver.find_element(By.CSS_SELECTOR, "div.article")
        text = article.text
        if len(text) > 1000:
            return text
    except Exception:
        pass

    # Try medrxiv
    url = f"https://www.medrxiv.org/content/{doi}v1.full"
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.article"))
        )
        article = driver.find_element(By.CSS_SELECTOR, "div.article")
        text = article.text
        if len(text) > 1000:
            return text
    except Exception:
        pass
    return None


def extract_nature_text(driver, doi):
    """Fetch full text from Nature/Springer."""
    url = f"https://www.nature.com/articles/{doi.split('/')[-1]}"
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
        )
        article = driver.find_element(By.CSS_SELECTOR, "article")
        text = article.text
        if len(text) > 2000:
            return text
    except Exception:
        pass
    return None


def extract_cell_text(driver, doi):
    """Fetch full text from Cell/Elsevier."""
    url = f"https://doi.org/{doi}"
    try:
        driver.get(url)
        time.sleep(3)  # Elsevier needs time to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.article-body, #body, article"))
        )
        # Try multiple selectors
        for selector in ["div.article-body", "#body", "article", "div.Body"]:
            try:
                elem = driver.find_element(By.CSS_SELECTOR, selector)
                text = elem.text
                if len(text) > 2000:
                    return text
            except Exception:
                continue
    except Exception:
        pass
    return None


def extract_science_text(driver, doi):
    """Fetch full text from Science/AAAS."""
    url = f"https://www.science.org/doi/{doi}"
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.article__body, article"))
        )
        for selector in ["div.article__body", "article", "div.core-container"]:
            try:
                elem = driver.find_element(By.CSS_SELECTOR, selector)
                text = elem.text
                if len(text) > 2000:
                    return text
            except Exception:
                continue
    except Exception:
        pass
    return None


def extract_pmc_html_text(driver, pmcid):
    """Fetch full text from PMC HTML view."""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.jig-ncbiinpagenav, article"))
        )
        for selector in ["div.jig-ncbiinpagenav", "article", "div#mc"]:
            try:
                elem = driver.find_element(By.CSS_SELECTOR, selector)
                text = elem.text
                if len(text) > 2000:
                    return text
            except Exception:
                continue
    except Exception:
        pass
    return None


def extract_generic_text(driver, doi):
    """Generic DOI resolution + text extraction."""
    url = f"https://doi.org/{doi}"
    try:
        driver.get(url)
        time.sleep(5)  # Wait for redirect
        # Try common selectors
        for selector in ["article", "div.article", "div.article-body",
                         "div.fulltext", "main", "div#content"]:
            try:
                elem = driver.find_element(By.CSS_SELECTOR, selector)
                text = elem.text
                if len(text) > 2000:
                    return text
            except Exception:
                continue
        # Fallback: get body text
        body = driver.find_element(By.TAG_NAME, "body")
        text = body.text
        if len(text) > 3000:
            return text
    except Exception:
        pass
    return None


def fetch_paper(driver, doi, filename):
    """Fetch full text for a paper and save it."""
    print(f"  Fetching: {doi}")
    text = None

    # Route to appropriate extractor based on DOI
    if "10.1101/" in doi:
        text = extract_biorxiv_text(driver, doi)
    elif "10.1038/" in doi:
        text = extract_nature_text(driver, doi)
    elif "10.1016/" in doi:
        text = extract_cell_text(driver, doi)
    elif "10.1126/" in doi:
        text = extract_science_text(driver, doi)

    # Fallback: generic extraction
    if not text or len(text) < 2000:
        text = extract_generic_text(driver, doi)

    if text and len(text) > 2000:
        # Save with proper header
        filepath = PAPERS_DIR / filename
        content = f"# {filename.replace('.txt','')}\n\n## full_text\n\n{text}"
        filepath.write_text(content, encoding='utf-8')
        print(f"  ✓ Saved {len(content)//1024}KB")
        return True
    else:
        print(f"  ✗ Failed (got {len(text) if text else 0} chars)")
        return False


def main():
    """Find all abstract-only papers and fetch full text."""
    # Find papers that are abstract-only (<3KB)
    abstract_only = []
    for f in sorted(PAPERS_DIR.iterdir()):
        if f.suffix == '.txt' and f.stat().st_size < 3000:
            # Extract DOI from filename
            name = f.stem
            if 'doi_fulltext_' in name:
                doi = name.replace('PMCdoi_fulltext_', '').replace('_', '/', 1)
                # Fix DOI format: first _ is /, rest stay as-is or become .
                # e.g., PMCdoi_fulltext_10.1101_2024.05.03.592390 -> 10.1101/2024.05.03.592390
                parts = name.replace('PMCdoi_fulltext_', '')
                # DOI always has format: 10.XXXX/rest
                match = re.match(r'(10\.\d+)_(.*)', parts)
                if match:
                    doi = f"{match.group(1)}/{match.group(2)}"
                    abstract_only.append((doi, f.name))

    print(f"Found {len(abstract_only)} abstract-only papers to fetch")

    # Load progress
    progress = {}
    if PROGRESS_FILE.exists():
        progress = json.loads(PROGRESS_FILE.read_text())

    # Filter already attempted
    to_fetch = [(doi, fname) for doi, fname in abstract_only
                if doi not in progress or not progress[doi].get('attempted')]

    print(f"Remaining to fetch: {len(to_fetch)}")

    if not to_fetch:
        print("Nothing to fetch!")
        return

    # Create driver
    driver = create_driver()
    success = 0
    failed = 0

    try:
        for i, (doi, filename) in enumerate(to_fetch):
            print(f"\n[{i+1}/{len(to_fetch)}] {doi}")
            result = fetch_paper(driver, doi, filename)
            progress[doi] = {"attempted": True, "success": result, "filename": filename}

            if result:
                success += 1
            else:
                failed += 1

            # Save progress periodically
            if (i + 1) % 5 == 0:
                PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
                print(f"  Progress: {success} success, {failed} failed, {len(to_fetch)-i-1} remaining")

            # Rate limit
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        driver.quit()
        PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
        print(f"\n=== Final: {success} success, {failed} failed ===")


if __name__ == "__main__":
    main()
