import pandas as pd
import requests
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SAFE_HEADERS = {"User-Agent": "url-extractor/1.0"}
CACHE = {}

def normalize_www(url, timeout=4):
    """Return URL with www. if that version is valid (responds <500)."""
    if not isinstance(url, str) or not url.strip():
        return url, "INVALID"

    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    parsed = urlparse(url)
    netloc = parsed.netloc
    if netloc.startswith("www."):
        print(f"[SKIP] Already has www → {url}")
        return url, "ALREADY_HAS_WWW"

    base_domain = netloc.lower()
    if base_domain in CACHE:
        cached = CACHE[base_domain]
        print(f"[CACHE] {url} → {cached}")
        return cached, "CACHED"

    # Step 1: Try original URL (see if it redirects to a www version)
    try:
        resp = requests.get(url, headers=SAFE_HEADERS, allow_redirects=True, timeout=timeout, stream=True)
        final_host = urlparse(resp.url).netloc
        if final_host.startswith("www."):
            CACHE[base_domain] = resp.url
            print(f"[REDIRECT] {url} → {resp.url}")
            return resp.url, "REDIRECTED_TO_WWW"
    except requests.RequestException:
        pass

    # Step 2: Try manually adding www
    test_url = urlunparse((parsed.scheme, "www." + netloc, parsed.path or "/", "", "", ""))
    try:
        resp = requests.get(test_url, headers=SAFE_HEADERS, allow_redirects=True, timeout=timeout, stream=True)
        if resp.status_code < 500:
            CACHE[base_domain] = test_url
            print(f"[ADDED] {url} → {test_url}")
            return test_url, "ADDED_WWW"
    except requests.RequestException:
        pass

    CACHE[base_domain] = url
    print(f"[UNCHANGED] {url} (www. not valid)")
    return url, "UNCHANGED"

def process_url(url):
    new_url, status = normalize_www(url)
    return url, new_url, status

def main():
    input_file = "data/extracted.csv"
    output_file = "data/extracted_with_www.csv"
    threads = 20  # adjust based on your connection and CPU

    print("Loading CSV...")
    df = pd.read_csv(input_file)
    if "URL" not in df.columns:
        raise ValueError("No 'URL' column found in the CSV")

    urls = df["URL"].tolist()
    results = []

    print(f"Processing {len(urls)} URLs using {threads} threads...\n")

    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_url = {executor.submit(process_url, u): u for u in urls}
        for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Normalizing"):
            try:
                url, new_url, status = future.result()
                results.append((url, new_url, status))
            except Exception as e:
                u = future_to_url[future]
                print(f"[ERROR] {u} → {e}")
                results.append((u, u, f"ERROR: {e}"))

    results_df = pd.DataFrame(results, columns=["Original_URL", "Normalized_URL", "Status"])
    merged_df = df.merge(results_df, left_on="URL", right_on="Original_URL", how="left")
    merged_df.drop(columns=["Original_URL"], inplace=True)
    merged_df.to_csv(output_file, index=False)

    print(f"\n✅ Done! Saved {len(merged_df)} rows to {output_file}")
    print("\nSummary:")
    print(merged_df["Status"].value_counts())

if __name__ == "__main__":
    main()
