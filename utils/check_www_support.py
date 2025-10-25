import csv
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

INPUT_FILE = "data/cloudflare50k.csv"
OUTPUT_FILE = "cloudflare50k_www_supported.csv"
MAX_WORKERS = 50   
TIMEOUT = 5
SAVE_INTERVAL = 100  


def check_www(domain):
    domain = domain.strip()
    if not domain:
        return (domain, False)

    if domain.startswith("www."):
        return (domain, True)

    test_url = f"https://www.{domain}"
    try:
        response = requests.head(test_url, timeout=TIMEOUT, allow_redirects=True)
        if response.status_code < 400:
            return (domain, True)
    except requests.RequestException:
        pass
    return (domain, False)


def load_checkpoint():
    if not os.path.exists(OUTPUT_FILE):
        return set(), []

    completed = set()
    results = []

    with open(OUTPUT_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row["domain"].strip()
            results.append((domain, row["www_supported"].lower() == "true"))
            completed.add(domain)

    print(f"Resuming from checkpoint: {len(completed)} domains already processed.")
    return completed, results


def main():
    # Load all domains
    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as f:
        all_domains = [line.strip() for line in f if line.strip()]

    completed, results = load_checkpoint()

    remaining_domains = [d for d in all_domains if d not in completed]

    print(f"Starting scan: {len(remaining_domains)} remaining domains.")

    processed = len(results)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, (domain, supported) in enumerate(
            tqdm(executor.map(check_www, remaining_domains), total=len(remaining_domains))
        ):
            results.append((domain, supported))
            processed += 1

            if processed % SAVE_INTERVAL == 0:
                with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["domain", "www_supported"])
                    writer.writerows(results)
                print(f"💾 Saved progress: {processed}/{len(all_domains)} domains")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "www_supported"])
        writer.writerows(results)

    print(f"Done! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
