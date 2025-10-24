import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from html import unescape
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote, urljoin
from extractor.urls_data import urls

import pandas as pd
import requests
import tldextract
from bs4 import BeautifulSoup

SAFE_HEADERS = {"User-Agent": "url-extractor/1.0"}

SOCIAL_DOMAINS = {
    "facebook.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "youtube.com",
    "instagram.com",
    "t.me",
    "telegram.org",
    "pinterest.com",
    "reddit.com",
    "wechat.com",
    "weibo.com",
    "vk.com",
}

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def norm_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def count_specials(url: str) -> Tuple[int, int, int, int]:
    return (
        url.count("="),
        url.count("?"),
        url.count("&"),
        sum(1 for c in url if not c.isalnum() and c not in "._-:/&?=#"),
    )


def is_ip(host: str) -> bool:
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host or ""))


def char_continuation_rate(s: str) -> float:
    if not s or len(s) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(s)) if s[i] == s[i - 1])
    return repeats / (len(s) - 1)


def estimate_url_char_logprob(
    url: str, char_probs: Optional[Dict[str, float]]
) -> float:
    if not url:
        return 0.0
    if not char_probs:
        char_probs = {chr(i): 1.0 / 95.0 for i in range(32, 127)}
    logp = 0.0
    for c in url:
        p = char_probs.get(c, char_probs.get("OTHER", 1e-6))
        if p <= 0:
            p = 1e-6
        logp += math.log(p)
    return logp


def load_tld_probs(path: Optional[str]) -> Dict[str, float]:
    d = {}
    if not path or not os.path.exists(path):
        return d
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        t = str(r[0]).lower() if "TLD" not in df.columns else str(r["TLD"]).lower()
        p = float(r[1]) if "prob" not in df.columns else float(r["prob"])
        d[t] = p
    return d


def load_char_probs(path: Optional[str]) -> Dict[str, float]:
    d = {}
    if not path or not os.path.exists(path):
        return d
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                ch, p = line.split(",", 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                ch, p = parts[0], parts[1]
            ch = ch.encode("utf-8").decode("unicode_escape")
            try:
                d[ch] = float(p)
            except ValueError:
                continue
    return d


def fetch(url: str, timeout: int = 12) -> Tuple[Optional[requests.Response], str]:
    try:
        resp = requests.get(
            url, headers=SAFE_HEADERS, timeout=timeout, allow_redirects=True
        )
        html = resp.text if resp and resp.text else ""
        return resp, html
    except Exception:
        return None, ""


def safe_host(url: str) -> Tuple[str, str, str, str]:
    u = urlparse(url)
    ext = tldextract.extract(url)
    tld = (ext.suffix or "").lower()
    domain = ext.registered_domain or (u.netloc or "")
    subdomain = ext.subdomain or ""
    return u.netloc or "", domain, subdomain, tld


def count_redirects(resp: Optional[requests.Response]) -> Tuple[int, int]:
    if not resp:
        return 0, 0
    total = len(resp.history)
    self_ref = 0
    try:
        final_host = urlparse(resp.url).netloc
        for h in resp.history:
            if urlparse(h.headers.get("Location", "")).netloc == final_host:
                self_ref += 1
    except Exception:
        pass
    return total, self_ref


def robots_flag(
    resp: Optional[requests.Response], soup: Optional[BeautifulSoup]
) -> int:
    # 1 if meta robots present or X-Robots-Tag header present
    if not (resp or soup):
        return 0
    if resp and any(h for h in resp.headers if h.lower() == "x-robots-tag"):
        return 1
    if soup:
        m = soup.find("meta", attrs={"name": re.compile("^robots$", re.I)})
        if m:
            return 1
    return 0


def responsive_flag(soup: Optional[BeautifulSoup]) -> int:
    if not soup:
        return 0
    vp = soup.find("meta", attrs={"name": re.compile("viewport", re.I)})
    return 1 if vp else 0


def favicon_flag(soup: Optional[BeautifulSoup]) -> int:
    if not soup:
        return 0
    return 1 if soup.find("link", rel=re.compile("icon", re.I)) else 0


def description_flag(soup: Optional[BeautifulSoup]) -> int:
    if not soup:
        return 0
    return (
        1 if soup.find("meta", attrs={"name": re.compile("^description$", re.I)}) else 0
    )


def count_css_js(soup: Optional[BeautifulSoup]) -> Tuple[int, int]:
    if not soup:
        return 0, 0
    css = len(
        soup.find_all("link", rel=lambda x: x and "stylesheet" in x.lower())
    ) + len(soup.find_all("style"))
    js = len(soup.find_all("script"))
    return css, js


def link_ref_counts(
    soup: Optional[BeautifulSoup], page_domain: str
) -> Tuple[int, int, int]:
    """NoOfSelfRef: anchors to same page (#...), NoOfEmptyRef: empty or '#', NoOfExternalRef: different domain"""
    if not soup:
        return 0, 0, 0
    self_ref = 0
    empty_ref = 0
    external_ref = 0
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href == "#":
            empty_ref += 1
        if href.startswith("#"):
            self_ref += 1
        netloc = urlparse(urljoin("http://" + page_domain, href)).netloc
        if page_domain and netloc and (netloc != page_domain):
            external_ref += 1
    return self_ref, empty_ref, external_ref


def popup_count(html: str) -> int:
    patterns = [r"window\.open\s*\(", r"alert\s*\(", r"confirm\s*\(", r"showModal\s*\("]
    return sum(len(re.findall(p, html or "", re.I)) for p in patterns)


def iframe_count(soup: Optional[BeautifulSoup]) -> int:
    return 0 if not soup else len(soup.find_all("iframe"))


def form_related_flags(
    soup: Optional[BeautifulSoup], page_domain: str
) -> Tuple[int, int, int, int, int]:
    """HasExternalFormSubmit, HasSocialNet, HasSubmitButton, HasHiddenFields, HasPasswordField"""
    if not soup:
        return 0, 0, 0, 0, 0
    has_ext_submit = 0
    has_submit_btn = 0
    has_hidden = 0
    has_password = 0
    for f in soup.find_all("form"):
        action = f.get("action", "").strip()
        if action:
            netloc = urlparse(urljoin("http://" + page_domain, action)).netloc
            if netloc and page_domain and (netloc != page_domain):
                has_ext_submit = 1
        if f.find("input", attrs={"type": re.compile("^submit$", re.I)}) or f.find(
            "button", attrs={"type": re.compile("^submit$", re.I)}
        ):
            has_submit_btn = 1
        if f.find("input", attrs={"type": re.compile("^hidden$", re.I)}):
            has_hidden = 1
        if f.find("input", attrs={"type": re.compile("^password$", re.I)}):
            has_password = 1

    # Social presence: any link to social domains
    has_social = 0
    for a in soup.find_all("a", href=True):
        netloc = urlparse(a["href"]).netloc.lower()
        if any(dom in netloc for dom in SOCIAL_DOMAINS):
            has_social = 1
            break
    return has_ext_submit, has_social, has_submit_btn, has_hidden, has_password


def copyright_flag(text: str) -> int:
    return (
        1
        if ("©" in text or "(c)" in text.lower() or "copyright" in text.lower())
        else 0
    )


def keyword_flags(text: str) -> Tuple[int, int, int]:
    t = text.lower()
    return (
        1 if "bank" in t else 0,
        1 if "pay" in t else 0,
        1 if "crypto" in t or "bitcoin" in t or "ethereum" in t else 0,
    )


def obfuscation_metrics(url: str) -> Tuple[int, int, float]:
    # Heuristics: count percent-encodings, long runs of non-letters/digits, '@'
    encodings = re.findall(r"%[0-9a-fA-F]{2}", url)
    noisy_runs = re.findall(r"[^a-zA-Z0-9]{3,}", url)
    at_count = url.count("@")
    count = len(encodings) + sum(len(r) for r in noisy_runs) + at_count
    has = 1 if count > 0 else 0
    ratio = count / max(1, len(url))
    return has, count, ratio

ALL_COLUMNS = [
    "FILENAME",
    "URL",
    "URLLength",
    "Domain",
    "DomainLength",
    "IsDomainIP",
    "TLD",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "Title",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
    "label",
]


def extract_for_url(
    url: str,
    tld_probs: Dict[str, float],
    char_probs: Dict[str, float],
    sleep_sec: float = 0.0,
) -> Dict[str, object]:
    row = {c: None for c in ALL_COLUMNS}
    row["URL"] = url
    row["URLLength"] = len(url)
    host, domain, subdomain, tld = safe_host(url)
    row["Domain"] = domain
    row["DomainLength"] = len(domain)
    row["IsDomainIP"] = 1 if is_ip(host) else 0
    row["TLD"] = tld
    row["TLDLength"] = len(tld)
    row["NoOfSubDomain"] = 0 if not subdomain else len(subdomain.split("."))
    row["IsHTTPS"] = 1 if urlparse(url).scheme.lower() == "https" else 0

    # filename: deterministic id
    row["FILENAME"] = sha1_hex(url) + ".html"

    # URL character stats
    url_unesc = unquote(url)
    letters = sum(ch.isalpha() for ch in url_unesc)
    digits = sum(ch.isdigit() for ch in url_unesc)
    specials = sum(1 for ch in url_unesc if not ch.isalnum())
    noeq, noqm, noamp, noother = count_specials(url_unesc)
    row["NoOfLettersInURL"] = letters
    row["LetterRatioInURL"] = letters / max(1, len(url_unesc))
    row["NoOfDegitsInURL"] = digits
    row["DegitRatioInURL"] = digits / max(1, len(url_unesc))
    row["NoOfEqualsInURL"] = noeq
    row["NoOfQMarkInURL"] = noqm
    row["NoOfAmpersandInURL"] = noamp
    row["NoOfOtherSpecialCharsInURL"] = noother
    row["SpacialCharRatioInURL"] = specials / max(1, len(url_unesc))

    # Obfuscation
    has_obf, obf_count, obf_ratio = obfuscation_metrics(url)
    row["HasObfuscation"] = has_obf
    row["NoOfObfuscatedChar"] = obf_count
    row["ObfuscationRatio"] = obf_ratio

    # Derived
    row["CharContinuationRate"] = char_continuation_rate(url_unesc)
    row["TLDLegitimateProb"] = tld_probs.get(tld.lower(), 0.5)
    row["URLCharProb"] = estimate_url_char_logprob(url_unesc, char_probs)

    # Network fetch
    resp, html = fetch(url)
    soup = BeautifulSoup(html, "html.parser") if html else None

    # Title features
    title = ""
    has_title = 0
    if soup and soup.title and soup.title.string:
        title = (soup.title.string or "").strip()
        has_title = 1
    row["HasTitle"] = has_title
    row["Title"] = title

    # Similarities
    url_tokens = norm_tokens(url_unesc)
    domain_tokens = norm_tokens(domain)
    title_tokens = norm_tokens(title)
    row["URLTitleMatchScore"] = jaccard(url_tokens, title_tokens)
    row["DomainTitleMatchScore"] = jaccard(domain_tokens, title_tokens)
    # URLSimilarityIndex (heuristic): similarity between domain tokens and path tokens
    path_tokens = norm_tokens(urlparse(url_unesc).path)
    row["URLSimilarityIndex"] = jaccard(domain_tokens, path_tokens)

    # HTML stats
    lines = html.splitlines() if html else []
    row["LineOfCode"] = len(lines)
    row["LargestLineLength"] = max((len(l) for l in lines), default=0)
    row["HasFavicon"] = favicon_flag(soup)
    row["Robots"] = robots_flag(resp, soup)
    row["IsResponsive"] = responsive_flag(soup)
    total_redir, self_redir = count_redirects(resp)
    row["NoOfURLRedirect"] = total_redir
    row["NoOfSelfRedirect"] = self_redir
    row["HasDescription"] = description_flag(soup)
    row["NoOfPopup"] = popup_count(html)
    row["NoOfiFrame"] = iframe_count(soup)
    # Images, CSS, JS counts
    row["NoOfImage"] = 0 if not soup else len(soup.find_all("img"))
    css_cnt, js_cnt = count_css_js(soup)
    row["NoOfCSS"] = css_cnt
    row["NoOfJS"] = js_cnt
    # Link refs
    self_ref, empty_ref, ext_ref = link_ref_counts(soup, domain)
    row["NoOfSelfRef"] = self_ref
    row["NoOfEmptyRef"] = empty_ref
    row["NoOfExternalRef"] = ext_ref
    # Forms & social
    has_ext_submit, has_social, has_submit_btn, has_hidden, has_password = (
        form_related_flags(soup, domain)
    )
    row["HasExternalFormSubmit"] = has_ext_submit
    row["HasSocialNet"] = has_social
    row["HasSubmitButton"] = has_submit_btn
    row["HasHiddenFields"] = has_hidden
    row["HasPasswordField"] = has_password

    # Keywords & copyright
    text = soup.get_text(" ", strip=True) if soup else ""
    bank, pay, crypto = keyword_flags(text)
    row["Bank"] = bank
    row["Pay"] = pay
    row["Crypto"] = crypto
    row["HasCopyrightInfo"] = copyright_flag(text)

    # label unknown for fresh URLs
    row["label"] = ""

    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return row


def read_input_urls(path: str) -> List[str]:
    urls = []
    _, ext = os.path.splitext(path.lower())
    if ext in (".csv", ".tsv"):
        df = pd.read_csv(path)
        candidate_cols = [c for c in df.columns if c.lower() in ("url", "link", "href")]
        if not candidate_cols:
            raise ValueError("CSV must contain a 'URL' column")
        urls = [
            str(u)
            for u in df[candidate_cols[0]].tolist()
            if isinstance(u, str) and u.strip()
        ]
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                urls.append(line)
    return urls


def main():
    ap = argparse.ArgumentParser(description="PhiUSIIL-compatible feature extractor")
    ap.add_argument("--tld-probs", default=None, help="Optional CSV with TLD,prob")
    ap.add_argument(
        "--char-probs",
        default=None,
        help="Optional char prob text (char,prob per line)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Polite delay between requests (seconds)",
    )
    args = ap.parse_args()

    tld_probs = load_tld_probs(args.tld_probs)
    char_probs = load_char_probs(args.char_probs)


    rows = []
    for u in urls:
        try:
            rows.append(extract_for_url(u, tld_probs, char_probs, args.sleep))
        except Exception as e:
            r = {c: None for c in ALL_COLUMNS}
            r["URL"] = u
            r["FILENAME"] = sha1_hex(u) + ".html"
            r["label"] = ""
            rows.append(r)
            continue

    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    boolish = [
        "IsDomainIP",
        "HasObfuscation",
        "IsHTTPS",
        "HasTitle",
        "HasFavicon",
        "Robots",
        "IsResponsive",
        "HasDescription",
        "HasExternalFormSubmit",
        "HasSocialNet",
        "HasSubmitButton",
        "HasHiddenFields",
        "HasPasswordField",
        "Bank",
        "Pay",
        "Crypto",
        "HasCopyrightInfo",
    ]
    for c in boolish:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    df.to_csv("data/extracted.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(df)} rows to data/extracted.csv")


if __name__ == "__main__":
    main()
