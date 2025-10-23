import asyncio
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from dom_content_crawler_script import *
import pandas as pd
import os
import csv

DATA_PATH = "./data/phiusiil.csv"
OUTPUT_DIR = "data/dom_content"
URL_RANGE = 14000
BATCH_SIZE = 50
MAX_WORKERS = 5
COOLDOWN = 2

data_set = pd.read_csv(DATA_PATH)
urls = data_set[data_set["label"] == 0]["URL"].tail(URL_RANGE).tolist()
labels = data_set[data_set["label"] == 0]["label"].tail(URL_RANGE).tolist()


def create_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-popup-blocking")
    opts.add_argument("--disable-notifications")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service()
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(30)
    return driver


def analyze_page(url, label):
    driver = create_driver()
    try:
        load_page(driver, url)
        results = {
            "URL": url,
            "label": label,
            "dom_max_depth": dom_max_depth(driver),
            "dom_total_nodes": dom_total_nodes(driver),
            "dom_avg_branching_factor": dom_avg_branching_factor(driver),
            "dom_max_children_per_node": dom_max_children_per_node(driver),
            "dom_iframe_count": dom_iframe_count(driver),
            "dom_iframe_max_nesting": dom_iframe_max_nesting(driver),
            "dom_form_count": dom_form_count(driver),
            "dom_hidden_input_count": dom_hidden_input_count(driver),
            "dom_forms_external_action_ratio": dom_forms_external_action_ratio(driver),
            "dom_script_count": dom_script_count(driver),
            "dom_suspicious_script_count": dom_suspicious_script_count(driver),
            "dom_popup_indicators": dom_popup_indicators(driver),
            "dom_hover_url_mismatch_count": dom_hover_url_mismatch_count(driver),
            "dom_mixed_content_count": dom_mixed_content_count(driver),
            "dom_external_resource_count": dom_external_resource_count(driver),
            "dom_cross_origin_iframe_count": dom_cross_origin_iframe_count(driver),
            "dom_hidden_element_ratio": dom_hidden_element_ratio(driver),
            "dom_clickable_without_href_count": dom_clickable_without_href_count(
                driver
            ),
            "dom_keyboard_event_on_password_count": dom_keyboard_event_on_password_count(
                driver
            ),
            "dom_autocomplete_off_password_count": dom_autocomplete_off_password_count(
                driver
            ),
            "dom_setTimeout_or_setInterval_presence": dom_setTimeout_or_setInterval_presence(
                driver
            ),
            "dom_mutation_observer_presence": dom_mutation_observer_presence(driver),
            "dom_service_worker_register_presence": dom_service_worker_register_presence(
                driver
            ),
            "dom_clipboard_access_presence": dom_clipboard_access_presence(driver),
            "dom_download_link_count": dom_download_link_count(driver),
            "dom_data_uri_image_count": dom_data_uri_image_count(driver),
            "dom_form_target_blank_count": dom_form_target_blank_count(driver),
            "dom_anchor_noopener_missing_ratio": dom_anchor_noopener_missing_ratio(
                driver
            ),
            "dom_event_handler_attr_total": dom_event_handler_attr_total(driver),
            "dom_third_party_domains_unique": dom_third_party_domains_unique(driver),
        }
        return results
    finally:
        driver.quit()


async def process_batch(batch_num, batch_urls, batch_labels):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = f"DOM_CONTENT_BATCH_{batch_num}.csv"
    path = os.path.join(OUTPUT_DIR, file_name)

    if os.path.exists(path) and os.path.getsize(path) > 100:
        print(f"[Skipping batch] {batch_num} — file already exists ({file_name})")
        return

    print(f"\n[→] Processing Selenium batch {batch_num} ({len(batch_urls)} URLs)...")
    results = []

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        tasks = [
            loop.run_in_executor(executor, analyze_page, url, label)
            for url, label in zip(batch_urls, batch_labels)
        ]

        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[Error during processing]: {e}")

    if results:
        fieldnames = results[0].keys()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[✔] Batch {batch_num}: Saved {len(results)} entries → {file_name}")
    else:
        print(f"[!] Batch {batch_num} produced no results.")


async def main():
    total_batches = (len(urls) + BATCH_SIZE - 1) // BATCH_SIZE
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    completed_batches = {
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("DOM_CONTENT_BATCH_") and f.endswith(".csv")
    }

    for i in range(total_batches):
        batch_num = i + 1
        if batch_num in completed_batches:
            print(f"[Skipping batch] {batch_num} — already processed.")
            continue

        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(urls))
        batch_urls = urls[start:end]
        batch_labels = labels[start:end]

        if not batch_urls:
            break

        await process_batch(batch_num, batch_urls, batch_labels)
        await asyncio.sleep(COOLDOWN)

    print("\n✅ All batches completed or skipped successfully.")


if __name__ == "__main__":
    asyncio.run(main())
