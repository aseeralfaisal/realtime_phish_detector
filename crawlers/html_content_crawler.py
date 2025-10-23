from bs4 import BeautifulSoup
import requests, csv, math, re, os, time, signal
from urllib.parse import urlparse, urljoin
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import pandas as pd

DATA_PATH = "data/LegitPhish.csv"
URL_RANGE = 10000
OUTPUT_DIR = "data/html_content_features"

BATCH_SIZE = 50
MAX_WORKERS = 5
REQUEST_TIMEOUT = (5, 10) 
HARD_TIMEOUT = 30           


def shannon_entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    return round(-sum(p * math.log(p, 2) for p in prob), 3)


def fetch_url(url, label):
    def handler(signum, frame):
        raise TimeoutError("Hard signal timeout inside process")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(25) 

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/137.0.0.0 Safari/537.36"
        )
    }

    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(max_retries=1))
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=1))

    start_t = time.time()
    try:
        res = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT, verify=False)
        html = res.text
        soup = BeautifulSoup(html, "html.parser")
        domain = urlparse(url).netloc.lower()
        text = soup.get_text(separator=" ", strip=True).lower()
        total_words = len(text.split()) or 1

        features = {"url": url, "label": label}
        features["page_length_chars"] = len(html)
        features["text_to_html_ratio"] = round(len(text) / (len(html) + 1e-5), 3)

        scripts = soup.find_all("script")
        inline_scripts = [s.get_text() for s in scripts if not s.get("src") and s.get_text()]
        entropies = [shannon_entropy(js) for js in inline_scripts if js.strip()]
        features["script_entropy_avg"] = round(sum(entropies) / len(entropies), 3) if entropies else 0
        features["num_scripts"] = len(scripts)
        features["inline_script_ratio"] = round(len(inline_scripts) / (len(scripts) or 1), 3)
        features["num_obfuscated_scripts"] = sum(
            1 for js in inline_scripts if re.search(r"(eval\(|unescape\(|\\x[0-9a-f]{2,})", js)
        )
        features["max_script_length"] = max([len(js) for js in inline_scripts], default=0)

        forms = soup.find_all("form")
        inputs = soup.find_all("input")
        features["num_forms"] = len(forms)
        features["has_password_field"] = int(any("password" in (i.get("type") or "") for i in inputs))
        features["num_hidden_inputs"] = sum(1 for i in inputs if (i.get("type") or "").lower() == "hidden")
        features["total_input_fields"] = len(inputs)
        features["num_forms_external_action"] = sum(
            1
            for f in forms
            if f.get("action")
            and urlparse(urljoin(url, f["action"])).netloc.lower() != domain
        )
        features["suspicious_input_name_count"] = sum(
            1 for i in inputs if re.search(r"(user|email|login|pass|otp|ssn)", str(i.get("name") or "").lower())
        )
        features["credential_placeholder_count"] = sum(
            1 for i in inputs if re.search(r"(password|code|otp|pin)", str(i.get("placeholder") or "").lower())
        )

        imgs = soup.find_all("img")
        features["num_images"] = len(imgs)
        features["image_alt_anomaly_ratio"] = round(
            len([i for i in imgs if not i.get("alt")]) / (len(imgs) or 1), 3
        )

        css_links = soup.find_all("link", rel=lambda v: v and "stylesheet" in v.lower())
        js_ext_links = [s.get("src") for s in scripts if s.get("src")]
        features["external_css_count"] = sum(
            1 for c in css_links if urlparse(urljoin(url, c.get("href", ""))).netloc != domain
        )
        features["external_js_count"] = sum(
            1 for j in js_ext_links if urlparse(urljoin(url, j)).netloc != domain
        )

        anchors = soup.find_all("a", href=True)
        features["num_external_links"] = sum(
            1 for a in anchors if urlparse(urljoin(url, a["href"])).netloc != domain
        )
        features["ip_link_count"] = sum(
            1 for a in anchors if re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", urlparse(a["href"]).netloc)
        )
        features["mailto_link_count"] = sum(1 for a in anchors if a["href"].startswith("mailto:"))

        features["num_meta_refresh"] = len(
            [m for m in soup.find_all("meta") if m.get("http-equiv", "").lower() == "refresh"]
        )

        def domain_mismatch(tag, attr):
            if tag and tag.get(attr):
                tag_domain = urlparse(urljoin(url, tag[attr])).netloc.lower()
                return int(tag_domain and tag_domain != domain)
            return 0

        features["base_tag_domain_mismatch"] = domain_mismatch(soup.find("base", href=True), "href")
        features["canonical_domain_mismatch"] = domain_mismatch(soup.find("link", rel="canonical"), "href")
        og = soup.find("meta", property="og:url")
        features["og_url_domain_mismatch"] = domain_mismatch(og, "content")
        favicon = soup.find("link", rel=lambda v: v and "icon" in v.lower())
        features["favicon_domain_mismatch"] = domain_mismatch(favicon, "href")

        inline_events = re.findall(r"on[a-z]+\s*=", html.lower())
        features["inline_event_handler_count"] = len(inline_events)

        suspicious_words = ["login", "verify", "account", "bank", "password", "secure", "update", "confirm", "signin"]
        features["suspicious_keyword_density"] = round(
            sum(text.count(w) for w in suspicious_words) / total_words, 5
        )

        features["mixed_protocol_reference_count"] = (
            len(re.findall(r"http://", html)) if url.startswith("https") else 0
        )

        duration = round(time.time() - start_t, 2)
        print(f"[✓] {url} processed in {duration}s")
        return features

    except requests.exceptions.Timeout:
        print(f"[⏱] Request timeout for URL: {url}")
    except requests.exceptions.RequestException as e:
        print(f"[!] Request error for URL {url}: {e}")
    except TimeoutError:
        print(f"[⏱] Internal 25 s timeout hit for {url}")
    except Exception as e:
        print(f"[!] Unexpected error for {url}: {e}")
    finally:
        signal.alarm(0)

    return None

def process_batch(batch_urls, batch_labels, batch_index):
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(fetch_url, url, label): url for url, label in zip(batch_urls, batch_labels)}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result(timeout=HARD_TIMEOUT)
                if result:
                    results.append(result)
            except FuturesTimeoutError:
                print(f"[⏱] HARD timeout (> {HARD_TIMEOUT}s): killing process for {url}")
                future.cancel()
            except Exception as e:
                print(f"[!] Error processing {url}: {e}")

    if results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_name = f"HTML_CONTENT_BATCH_{batch_index}.csv"
        path = os.path.join(OUTPUT_DIR, file_name)
        fieldnames = results[0].keys()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[✔] Batch {batch_index}: Saved {len(results)} pages → {path}")

if __name__ == "__main__":
    data_set = pd.read_csv(DATA_PATH)
    urls = data_set[data_set["ClassLabel"] == 1]["URL"].tail(URL_RANGE).tolist()
    labels = data_set[data_set["ClassLabel"] == 1]["ClassLabel"].tail(URL_RANGE).tolist()

    total_batches = (len(urls) + BATCH_SIZE - 1) // BATCH_SIZE
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    completed_batches = {
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("HTML_CONTENT_BATCH_") and f.endswith(".csv")
    }

    for i in range(total_batches):
        batch_num = i + 1
        if batch_num in completed_batches:
            print(f"[Skipping batch] {batch_num} — already processed.")
            continue

        start, end = i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, len(urls))
        print(f"\n[→] Processing batch {batch_num}/{total_batches} ({end - start} URLs)...")
        process_batch(urls[start:end], labels[start:end], batch_num)
        time.sleep(2)
