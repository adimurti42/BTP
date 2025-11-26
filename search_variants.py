import pandas as pd
import numpy as np

df = pd.read_csv("PATH_TO_CSV_FILE")


# %%
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_indigenomes(df,
                       driver_path: str | None = None,
                       headless: bool = True,
                       timeout: int = 15):
    """
    df: pandas.DataFrame with a 'query' column (strings like 'chr12-109586107-A-G')
    driver_path: optional path to chromedriver binary (use if you want to control the driver binary)
    headless: run browser headless (True) or visible (False)
    timeout: per-query timeout in seconds
    Returns: df with appended columns:
      ["Gene","Chr","Pos","Ref","Alt","dbSNP ID","Gene Function","Exonic Function"]
    """

    expected_cols = ["Gene", "Chr", "Pos", "Ref", "Alt", "dbSNP ID", "Gene Function", "Exonic Function"]

    # Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    if headless:
        # use new headless when possible
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
    # create driver
    if driver_path:
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=options)
    else:
        driver = webdriver.Chrome(options=options)

    base_url = "https://clingen.igib.res.in/indigen/index.php#/panel8"
    wait = WebDriverWait(driver, timeout)

    results = []

    for idx, q in enumerate(df['query'].astype(str).tolist(), 1):
        print(f"[{idx}/{len(df)}] Searching: {q}")
        driver.get(base_url)

        try:
            # wait for the global search box (the one you showed)
            search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.searchBar")))
        except Exception as e:
            print("  ! Search box not found:", e)
            results.append({c: None for c in expected_cols})
            continue

        # Clear, type and press Enter (no clicking the icon)
        search_box.clear()
        search_box.send_keys(q)
        search_box.send_keys(Keys.RETURN)

        # Poll for a valid data row inside the specific table container
        row_texts = None
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # select the specific wrapper/table you pasted
                table = driver.find_element(By.CSS_SELECTOR, "div.card-body.card-body-cascade.mx-2 table#example")
                tbody = table.find_element(By.TAG_NAME, "tbody")
                rows = tbody.find_elements(By.TAG_NAME, "tr")

                if not rows:
                    time.sleep(0.4)
                    continue

                # find the first row that looks like data (has >= 6 tds and not 'no data')
                selected = None
                for r in rows:
                    tds = r.find_elements(By.TAG_NAME, "td")
                    if len(tds) >= 6:
                        # get text and do a quick sanity check
                        texts = [td.text.strip() for td in tds]
                        joined = " ".join(texts).lower()
                        if "no data" in joined:
                            continue
                        selected = r
                        break

                if selected is None:
                    time.sleep(0.4)
                    continue

                # extract td texts
                row_texts = [td.text.strip() for td in selected.find_elements(By.TAG_NAME, "td")]
                # sanity: must contain something like 'chr' or 'rs' or alpha gene name
                joined = " ".join([t.lower() for t in row_texts if t])
                if ("chr" not in joined) and ("rs" not in joined) and len(row_texts) < 4:
                    # not fully populated yet
                    row_texts = None
                    time.sleep(0.4)
                    continue

                # got it
                break

            except Exception:
                time.sleep(0.4)
                continue

        if row_texts is None:
            print("  -> Timeout/no valid result for query.")
            results.append({c: None for c in expected_cols})
            continue

        # Try to read the header names from thead (if present) so mapping is robust
        try:
            ths = table.find_elements(By.CSS_SELECTOR, "thead th")
            headers = [th.text.strip() for th in ths] if ths else []
        except Exception:
            headers = []

        row_map = {}
        if headers and len(headers) == len(row_texts):
            # header-driven mapping
            for h, v in zip(headers, row_texts):
                row_map[h] = v
            # ensure expected cols present
            for c in expected_cols:
                row_map.setdefault(c, None)
        else:
            # fallback: assume the expected order
            for i, col in enumerate(expected_cols):
                row_map[col] = row_texts[i] if i < len(row_texts) else None

        # debug preview
        print("   -> scraped:", row_map.get("Gene"), row_map.get("Chr"), row_map.get("Pos"))
        results.append(row_map)

    driver.quit()

    results_df = pd.DataFrame(results)
    # ensure columns exist and order them
    for c in expected_cols:
        if c not in results_df.columns:
            results_df[c] = None
    final = pd.concat([df.reset_index(drop=True), results_df[expected_cols].reset_index(drop=True)], axis=1)
    return final


# %%

final = scrape_indigenomes(df, headless=False, timeout=15)
print(final.head())



```
