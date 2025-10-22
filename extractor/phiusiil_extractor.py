"""
PhiUSIIL-compatible feature extractor
-------------------------------------
Outputs a CSV with EXACTLY the same 56 columns as the PhiUSIIL dataset.

Usage:
  python phiusiil_extractor.py --input urls.txt --output phiusiil_features.csv
  # Or with a CSV that has a column named URL:
  python phiusiil_extractor.py --input urls.csv --output phiusiil_features.csv

Optional priors:
  --tld-probs tld_probs.csv        # CSV with columns: TLD,prob
  --char-probs char_probs.txt      # Plain text where each line is "char ru0020 <prob>" or "char,prob"

Notes:
- Implements heuristic approximations for some web-derived features (e.g., responsiveness, robots).
- All 56 columns are present and named exactly like the public dataset.
"""

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


# ---------------------
# Core extraction
# ---------------------

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

    urls = [
        "https://curved-pager-squeaking.on-fleek.app/yxmerhivmibwai2.html",
        "https://curved-pager-squeaking.on-fleek.app/yxmerhivmibwai3.html",
        "http://app.bnb-audit.com/",
        "https://bnb-audit.com/",
        "https://how-minecraft.ru/bl.html",
        "http://gas33slot.com/wp-admin/wx.htm",
        "https://aumail7.sspl.com.auaumail7.sspl.com.aumail7.sspl.com.au/mobile/front/",
        "https://com.aumail7.sspl.com.auwww.mail7.sspl.com.aucom.auwww.mail7.sspl.com.auwww.sspl.com.aumail7.sspl.com.auwww.mail7.sspl.com.au/mobile/front",
        "https://com.aumail7.sspl.com.aumail7.sspl.com.aucom.aumail7.sspl.com.ausspl.com.aumail7.sspl.com.aumail7.sspl.com.au/mobile/front/",
        "https://auwww.aumail7.sspl.com.auwww.mail7.sspl.com.auwww.sspl.com.aumail7.sspl.com.auwww.mail7.sspl.com.au/mobile/front",
        "https://auwww.mail7.sspl.com.auwww.sspl.com.auwww.mail7.sspl.com.aucom.auwww.mail7.sspl.com.auwww.sspl.com.aumail7.sspl.com.auwww.mail7.sspl.com.au/mobile/front",
        "https://com.auaumx2.sspl.com.aucom.auaumx2.sspl.com.auaumx2.sspl.com.auaumx2.sspl.com.aucom.auaumx2.sspl.com.aucom.auaumx2.sspl.com.auaumx2.sspl.com.auaumx2.sspl.com.au/mobile/login/",
        "https://auwww.aumail7.sspl.com.auwww.mail7.sspl.com.auwww.sspl.com.aumail7.sspl.com.auwww.mail7.sspl.com.au/mobile/login/",
        "https://aucom.auaumx2.sspl.com.auwww.com.auaumx2.sspl.com.auwww.aumx2.sspl.com.auaumx2.sspl.com.au/mobile/login/",
        "https://com.auwww.com.auwww.live.sspl.com.au/mobile/front/",
        "http://meta-xilvyn.pages.dev/",
        "http://meta-emissary.pages.dev/",
        "https://www.ptjh-wahtsapp.com/",
        "http://www.i9hhk79i-trezor.vercel.app/",
        "https://www.market.dratassianaalves.com.br/",
        "https://www.alumabrtemany.digital/",
        "https://448620.com/Phishing",
        "https://business-for-clarvion.pages.dev/",
        "https://business-for-performance-catalyst-center.pages.dev/",
        "http://372656135051488813789482033989mjrfadimxatvapj.adelon.com.br/",
        "https://ipfs.io/ipfs/bafybeibvzu52qfdgathxv4jnok34vgz5bb3q6isjsqa6jlpuzopuzxbfse/",
        "https://airbnb-reservations-rooms-31377018-properties.projedekorasyon.com.tr/login.html",
        "https://airbnb-reservations-rooms-31377018-properties.projedekorasyon.com.tr/login_email.html",
        "https://auth.properties/daftkNJwgEZJA2bf9g?/tasks/aes-corporationhome/PlanViews/OtKjpJYAELPk?Type=AssignedTo&Channel=Email&CreatedTime=22102025&Exp=dnrt",
        "https://www.kkinstagram.com/reel/DPyldsmiH-P/",
        "https://cobbeetty.top/Issue/index",
        "https://cobbeetty.top/Issue/details/Lang/en-us",
        "https://case10008nat66kzc19v1gaelluptf1ea41s.netlify.app/id/10008nat66kzc",
        "https://case10008nat66kzc19v1gaelluptf1ea41s.netlify.app/id/10008NAT66KZC.html",
        "http://www.tnnj-wahtsapp.com/",
        "http://get-liveledgr-auth.pages.dev/",
        "http://support-livelgr-learn.pages.dev/",
        "https://www.prmbike.store/",
        "http://meta-edge-33e.pages.dev/",
        "http://www.baucarshopee1439.weebly.com/",
        "http://www.zimbra-blue.vercel.app/",
        "https://kkinstagram.com/post?target=instagram&shortcode=DP_NdUCjLfy&media_id=3746772603982624754",
        "http://www.facebook-blog-com.blogspot.com.co/",
        "https://accounts.soffttek.com/?rid=s6DSY7Z",
        "https://kkinstagram.com/reel/DP_NdUCjLfy/?igsh=azIxbTAzZzk5cHln",
        "https://www.superset.lectec.com.au/mobile/login/",
        "https://www.superset.lectec.com.au/mobile/front/",
        "https://business-for-advertisers-policy-customer.pages.dev/welcome",
        "https://broad-byte-tiny.on-fleek.app/dtdszxjaqp1.html",
        "https://kkinstagram.com/reel/DOE5XA9jFt7/?igsh=MXRvcjNmNnZwZWgxcg==",
        "https://eng-legrcom-lean.pages.dev/",
        "https://blockbook.dash.zelcore.io/0",
        "http://biz.028426.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "https://b25fff.com/fish/173/",
        "https://auth.properties/uyZp5jWIrW5H?/tasks/aes-corporationhome/PlanViews/OtKjpJYAELPk?Type=AssignedTo&Channel=Email&CreatedTime=22102025&Exp=dnrt",
        "http://basicpension.028426.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "http://www.admin.scotiaverificationregistry.com/",
        "https://project-tokenmints30.vercel.app/",
        "http://mdrfgyzzqr.duckdns.org/en/main",
        "https://amazon-clone-psi-rust.vercel.app/store",
        "https://amazon-clone-psi-rust.vercel.app/register",
        "https://aadimarine.com/buyer/RFQ/Price/buyer/mic.html",
        "https://aadimarine.com/buyer/RFQ/Price/buyer/login2.php",
        "https://connect-bridge-trezr-en.pages.dev/",
        "https://bridge-trezzr-connect.pages.dev/",
        "https://ales.nysa.pl/media/ppq",
        "https://neko.flap.com.mx/invoices/TLGtPU2B1SRYpzjeVpECuaju",
        "https://holaserviciio11.cloudaccess.host/wp-content/uploads/2025/10/correoss/SS/corr/corr.php/billing.php",
        "https://neko.flap.com.mx/invoices/fVAdIsK3MnfGfEAIjYG327J6.html",
        "https://sbm1te.sbs/",
        "http://aktivpayletter.safetycs.biz.id/",
        "https://anacassaraodontologia.com.br/profissionais",
        "https://wap.porte-finestre.com/",
        "https://ipfs.io/ipfs/bafkreibj7yyu5xkc74lqf22hytauqxx2qralrb6vh3hiy7737kyqrunhku/",
        "http://www.kanny.vn/wordpress/wp-content/particuliers/aces/particulier/loginform3ad6.php",
        "https://trezor-suit-information.pages.dev/",
        "https://centerbusinesse-lawcontrolv-fds2010d.netlify.app/",
        "http://labanquepostale.jupiter-analytics.com/thierry--_--.barbier/brigitte.--_--boissel@/francoise--_--.mariani@/salvatore--_--.fazzalari",
        "http://labanquepostale.jupiter-analytics.com/thierry--_--.barbier/brigitte.--_--boissel@/francoise--_--.mariani@/salvatore--_--.fazzalari/",
        "http://pub-a585275162b94eeead6f34b59fc34175.r2.dev/woosl.html",
        "http://pub-85898ccf2bf04ae9bbc716993ae9553d.r2.dev/vfiqwt.html",
        "http://pub-20cfdc3c226149f1ad785b63a8a97ca6.r2.dev/khgfdsx.html",
        "http://pub-defeab315f6d492d9ba6a7f191e5bf6b.r2.dev/muri.html",
        "http://pub-682ad3b65d944376b919745aae3c56d4.r2.dev/document3.html",
        "http://pub-049a67ec10bd4bb19484d969ee9e7535.r2.dev/st.html",
        "http://app.qpointsurvey.com/s/vjmotsuhuve7al3c/",
        "http://pub-373a4b9dee8448f7ae6feab2f1fbeb3f.r2.dev/zana.html",
        "http://pub-ee0e857c9bde4bb78669ce75a3076842.r2.dev/wae.html",
        "http://pub-1c3a35bc3f574b6684c3e0a322c562b4.r2.dev/mycarimage.html",
        "http://pub-d017b0f78ed64420b48f8c1f9845e731.r2.dev/xquick.html",
        "http://pub-b8816ea2a950483f808e5152b306618f.r2.dev/murrr.html",
        "http://pub-4d2983de97694aaea4c6e5be6e9695d5.r2.dev/hm.html",
        "http://pub-dde186d3ef204edd89e847d256cdf5bd.r2.dev/ghupl.html",
        "http://pub-19b834342e7d465b9f00b3511325a081.r2.dev/qwertyuiopBowa.html",
        "http://pub-089fc651c6d94c6fa7c926754f7d3016.r2.dev/zac.html",
        "http://pub-e42cb528b102447385e3145197916f51.r2.dev/owa-pageeee-owa.html",
        "http://pub-8376b5b2334844d193a525d8b8548fa5.r2.dev/b99.html",
        "http://pub-5951c9805b754b349e7780aa7d7195e9.r2.dev/okuorun.html",
        "http://pub-d6b5f2cd37034e30aaf6c10e489c3f7e.r2.dev/osedfosu.html",
        "https://owa.sarbacame.com/",
        "http://owa.sarbacame.com/owa/auth/",
        "http://pub-b8d557d502ff4b20ad52a88c112349ba.r2.dev/dfghmnbv.html",
        "http://pub-d8ee52f2e86b49caba542c9fc5533130.r2.dev/fkjhgfdsa.html",
        "http://pub-398374cead0f4c34808233c18510c81b.r2.dev/kjkudex.html",
        "http://pub-ea88ee75fced4023a55270e18780e191.r2.dev/sd0x.html",
        "http://65b7d3757c56390008a2d84a--ods-android.netlify.app/components/floatingactionbuttons_docs/",
        "http://pub-ad961fc311f24bea9daef5a210bb5231.r2.dev/901.html",
        "http://pub-6db61ec326e14c4bb5e59a7a284b21ae.r2.dev/gggindex.html",
        "http://pub-a99c53d6c23946e4a025da611a9aea62.r2.dev/owa-pageeee-owa.html",
        "http://harsh-helicopter-early.on-fleek.app/umewuhuwharna10.html",
        "http://large-tiger-thundering.on-fleek.app/sgbpcyfwptnhky10.html",
        "http://fat-van-echoing.on-fleek.app/jiwfggtbsnd1.html",
        "http://pub-a1faa69ae47d4063a72c37d8cbb391d8.r2.dev/hrdept-owa.html",
        "https://campanha.propostacartao.com/",
        "https://agrobiobot.com/QGmMsSwN",
        "https://acoustic-apartment-low.on-fleek.app/aa2ratuotuwwonline9.html",
        "https://trezosuite.vercel.app/",
        "https://accounts.marketwebb.ninja/en/login/",
        "https://accounts.marketwebb.ninja/",
        "https://auth-secure.me/zTl4ZXIrgLThTVJ6?/workitems/3ont208h/WorkItemMention/VGhpcyBtaWdodCBiZSBhIEhveGh1bnQgc2ltdWxhdGlvbi4gT25seSBvbmUgd2F5IHRvIGZpbmQgb3V0Li4u",
        "https://sso-auth.com/t0OGy9oJS8uLXGk?/facebook_secure_account?id=759347502378427672987349826578129038123807394%22",
        "https://allegrolokalnie.pl-smart8758124.cfd/oferta/77811313/kierownica-logitech-g29-shifter",
        "https://allegrolokalnie.pl-smart8758124.cfd/oferta/77811313/kierownica",
        "http://6568bdfa15f2f206de383621--elegant-dango-fe59cb.netlify.app/",
        "https://my-acesso-01.dynv6.net/",
        "https://www.indianstaffingfederation.org/isf-images/news/plugins/login.php",
        "https://brasilpgs.com/wp-content/redirect.php",
        "https://pulsemax.com.br/wp-content/oca/oca/captcha.php",
        "http://intermatic0.site/pacifico/inicio/alert.php",
        "https://herimarc.fun/",
        "https://www.kkinstagram.com/reel/DPd0RcRCXr0",
        "https://www.5mp.eu/fajlok2/insnsb/owa_www.5mp.eu_.html",
        "https://histarnaverlogin.vercel.app/",
        "https://3nayana.github.io/Amazon_clone/",
        "https://inceptioncodes.github.io/Amazon-Clone/",
        "https://signin.broker/IcBW2-16ksKF?/workitems/3ont208h/WorkItemMention/VGhpcyBtaWdodCBiZSBhIEhveGh1bnQgc2ltdWxhdGlvbi4gT25seSBvbmUgd2F5IHRvIGZpbmQgb3V0Li4u",
        "https://sso-auth.com/JsZNGyJ2KT6FcA?/facebook_secure_account?id=759347502378427672987349826578129038123807394",
        "https://zsdxgt3.pages.dev/privacyprefs?ref_=footer_iba",
        "https://rbfcu-star.azurewebsites.net/(S(pamrjqwd2qjr23qx5u53di3n))/Main/Login",
        "https://rbfcu-star.azurewebsites.net/(S(i0cqpxe3uarclcrvoqdhfwr0))/",
        "https://bafybeibdyr3vrviyiqdrraxyk5dxy5fb6subupjbgqwyvki3b7pn2h32lm.ipfs.w3s.link/",
        "https://xn.wttef.my.id/pembatalan.dana.cicil/",
        "https://outlook.webaccess-alert.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "http://pjmathernee.wixsite.com/my-site-1",
        "https://abdul-rasheed-talal.github.io/Netflix-UI-Clone/",
        "https://ingresa-seguro.info/pacifico/persona.php",
        "https://webmail.webaccess-email.org/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "https://steamconnmunity.com/tradeoffers/new/partners=5612547902000BERgg&token=TLgzzJ132",
        "https://wildcard.facture-rapide.fr/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "https://connbbggt.top/Contract/index/Lang/en-us",
        "http://f.digitalmaillane.com/igit/4/btdk5zbTu64uj1kvb5Tr7mTtq79pl5Tv5aT87zTivTbT4",
        "https://kadaindex.pages.dev/documents-and-resources",
        "http://f.digitalmaillane.com/igit/4/5o4ao0pSvd18xdw7nhSrjyS6imgy0hS6hmSkjbSu7SnSg",
        "https://meta-for-business-security-project.pages.dev/",
        "http://f.digitalmaillane.com/igit/4/5p3dk25Ygs4xbudo4yYk0fYhqip3oyYoy3Y00sYboY4Yx",
        "http://f.digitalmaillane.com/igit/4/8dwxhnsQkuvncwfq60Qm2hQg0dx4uQq05Q32uQdqQ6Qz",
        "https://vaish-2510.github.io/NetFLix_Clone/",
        "https://summer130421.028426.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "https://zeus.028426.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "https://delivery.freightinternationalservices.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "https://mybdoonline.device-deleted.workers.dev/bdo-form/E9H5Vwr5G3Ty7KVET9i48SUOGXDHnpXdS3VuNlkPpa",
        "https://bloger.028426.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "https://carli.028426.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "https://366yulechengbocaizhuce.028426.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "http://f.digitalmaillane.com/igit/4/afp7qntJ2o7u6n6hxrJdt8J02aejsrJhrwJutlJ4hJxJq",
        "http://f.digitalmaillane.com/igit/4/bos3fqjPi6im8dx8oiPldyP1zzmz7PtgmPmkcPv8PoPh",
        "https://help-nddax.webflow.io/",
        "https://mail.all-global-hr.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "http://www.kkinstagram.com/reel/DI6s2-eoD_O/",
        "https://whats-tp.vip/",
        "https://goprox.cc/go/y28413a4/03b4",
        "https://gameprox.cc/go/y28413a4/03b4/?rdr=1",
        "https://zh-imtoken.org.cn/",
        "https://melbourneairportpickup.com.au/wp-includes/musaa456/",
        "https://tamanna10517.github.io/amazoneclone/",
        "http://btinternet-102879.weeblysite.com/",
        "http://bt-105604.weeblysite.com/",
        "http://bt-104062.weeblysite.com/",
        "https://muhammadhashir786.github.io/amazon-new-clone/",
        "http://btbt-106339.weeblysite.com/",
        "https://advertisers-client-support-impersonation-data.pages.dev/",
        "https://docshare21qwesdx32wed5trgfcvi8jwesd908jioew90jiowe9jisjioziok.calnash.com/",
        "https://advertisers-client-support-compliance-problem.pages.dev/",
        "https://productosdini.com.ar/no_htts/moviies/login.php",
        "https://humayao345678.github.io/Humayo/",
        "https://learn-legrcom-auth.pages.dev/",
        "https://kkinstagram.com/reel/DPaZtnmiCp8/?igsh=dDUxbmw4aWd2aWQw",
        "https://meta-framex-24f.pages.dev/",
        "https://facture-orange.vercel.app/",
        "http://f.digitalmaillane.com/igit/4/cc2hhbzK09jf04ozf9Kc4pKfaxx4m9Kk7dKab3KmzKfK8",
        "https://pub-988628dfc24e41159991efd386ba77a3.r2.dev/blo.html",
        "https://bakhtiarisrafil.com/8FFd6W7Qd22m1Yngmd3PnrrfG4x7oT9778ml2ZEuEcj96hMSJBtET9FgUXS7dyl9DUbdN4WK4bb6iGLU1u_CZyhA49gRfXOhrfjkkN_wkpnqkFxFSstw337/",
        "https://sejaumfranqueado.flowcoffee.com.br/cops/",
        "https://claro-uy.help/",
        "https://dpd-trackss.top/en/",
        "https://jatin2004-code.github.io/Netflix-clone/",
        "https://accountcenter-help-bussiness.pages.dev/es/login/10005454872268/",
        "https://linkccx-ld-webs.vanxx.co/",
        "https://literate-happiness-six.vercel.app/",
        "https://web-ob-whatsapp.com.cn/",
        "https://darshankardil-create.github.io/amazon/",
        "https://cobbssggr.top/Contract/index/Lang/en-us",
        "https://campanha.propostacartaox.com/device-blocked?reason=Desktop+acc",
        "https://s.teampp.cfd/p/gkhs-kwlp/bnklfgtrntn/",
        "http://verificationscotiaregistry.com/",
        "https://hjft-wahtsapp.com/_layouts/15/spinstall0.aspx",
        "https://bbj-wahst5pp.com/boke",
        "https://ptjh-wahtsapp.com/_layouts/15/spinstall0.aspx",
        "https://neheta.github.io/Amazon-Clone/",
        "https://apple-icloud-login.com/",
        "https://kkinstagram.com/reel/DP1QZitiCm0/?igsh=M2M3Z2h2Zzg0Z2wy",
        "http://hwsrv-472845.hostwindsdns.com/",
        "http://hwsrv-472845.hostwindsdns.com/start.html",
        "https://superfacilfeirao-ghee.shop/inicio/",
        "https://goprox.cc/go/43f403/03a4",
        "http://01-pontosagora.dynv6.net/mobile/index.php?hash=82267844468f7bef",
        "https://auth-secure.me/O_iri9EeuYtp2Zg?/workitems/3ont208h/WorkItemMention/VGhpcyBtaWdodCBiZSBhIEhveGh1bnQgc2ltdWxhdGlvbi4gT25seSBvbmUgd2F5IHRvIGZpbmQgb3V0Li4u",
        "https://i9hhk79i-trezor.vercel.app/",
        "https://kkinstagram.com/reel/DP9KO8ZCAPU/?igsh=MXRsbGpvb21zM254MA==",
        "http://market.dratassianaalves.com.br/",
        "https://lfn-wahst5pp.com/_layouts/15/spinstall0.aspx",
        "http://dpd-tracksg.top/en/",
        "https://listado.mercadolibre.com.uy/pagina/stickerland/",
        "https://tnnj-wahtsapp.com/boke",
        "https://rbfcu-star.azurewebsites.net/(S(kky5gavxxhoe4qfgebt54fdr))/Main/Login",
        "https://rbfcu-star.azurewebsites.net/(S(w32fhmdo2stkupru42lixrlb))/",
        "https://www.kkinstagram.com/reel/DPw52-FEhOT/?igsh=MWttNzU0ajZvY2NjNg==",
        "https://bxdp-wahtsapp.com/boke",
        "https://wwm-wahst5pp.com/shiyongjiaocheng",
        "https://kkinstagram.com/reel/DO8Q8xmjC5U/?igsh=MXZ5N3Z5bmlucWhxMA==",
        "http://recebidohojettk.shop/",
        "https://tarefaspagasttk.shop/",
        "https://tiktok.shop.hoverboardtrix.shop/",
        "https://creme-rosa.ttiktok.shop/",
        "http://att-t.bitbucket.io/",
        "http://pageid564938564-ads.ubpages.com/ils2025/",
        "https://kkinstagram.com/reel/DNIgF-II7WL/?igsh=MXV1cDJsaWEzMGhxdw==",
        "https://084793.com/s/63BZGFSVBWSFCDX7Y9/584dd8/90eab167-7429-489f-99f6-ce86e8d0d81a",
        "https://rbfcu-star.azurewebsites.net/(S(30wrm5ovsgbqgqyq2sx11qcw))/Main/Login",
        "https://rbfcu-star.azurewebsites.net/(S(clrhwua2chs4zr5dvmuuuqy3))/",
        "https://027saibo.com/",
        "https://www.azikus-official.com/",
        "https://rbfcu-star.azurewebsites.net/(S(igm2ufwl55dt3y2rokfm4zfh))/Main/Login",
        "https://rbfcu-star.azurewebsites.net/(S(hl25vmfznxwfqs1ks35ll2yv))/",
        "http://baollll2.cc/",
        "https://rbfcu-star.azurewebsites.net/(S(s3xs32za2c3j0beub2xmik4i))/Main/Login",
        "https://rbfcu-star.azurewebsites.net/(S(2vcih3llgkzxb2zp44tpbk52))/",
        "http://schedulecallwithqb.com/",
        "http://trx-transactions.app/",
        "https://kkinstagram.com/p/CwYUXsmMAN_/?img_index=1",
        "http://998706chadmccann-karnsi.crabdance.com/",
        "https://www.lns-europe.cam/vine1/upload/en.php?rand=13InboxLightaspxn.1774256418&fid.4.1252899642&fid=1&fav.1&rand.13InboxLight.aspxn.1774256418&fid.1252899642&fid.1&fav.1&email=eWFob29AeWFob28uY29t&.rand=13InboxLight.aspx?n=1774256418&fid=4",
        "https://comfermt.loophole.site/gat/",
        "http://dnsflarenetx.sa.com/scss/en/microsoftonline/onedrive/auth",
        "https://mobilebanking.bdosecurity.workers.dev/bdo-form/qrDXBqcPDb4mPde4VkbrKai935GJGsi4khDLNiKEWV",
        "https://whats-xtx.vip/",
        "https://meta-bitcore.pages.dev/",
        "https://meta-bond-con.pages.dev/",
        "https://meta-cell-9gy.pages.dev/",
        "https://meta-clay.pages.dev/",
        "https://meta-codex.pages.dev/",
        "https://whats-xtk.vip/",
        "https://meta-blockhub.pages.dev/",
        "https://meta-bloomcore.pages.dev/",
        "https://meta-codehub.pages.dev/",
        "https://meta-compact-bf5.pages.dev/",
        "https://meta-beaconx.pages.dev/",
        "https://meta-band.pages.dev/",
        "https://meta-batch-eu4.pages.dev/",
        "https://meta-consolex-e49.pages.dev/",
        "https://meta-concept-7uu.pages.dev/",
        "https://metalyx-dynamics.pages.dev/",
        "https://meta-connectx-du1.pages.dev/",
        "https://business-for-progressvision-hub.pages.dev/",
        "https://metalyx-grid.pages.dev/",
        "https://business-for-marketpath-labs.pages.dev/",
        "https://metalyx-stack.pages.dev/",
        "https://business-for-corevision-labs-ap1.pages.dev/",
        "https://meta-yernz.pages.dev/",
        "https://business-for-lighthousevision-hub.pages.dev/",
        "https://meta-conflux-44g.pages.dev/",
        "https://business-for-forwardforge-labs.pages.dev/",
        "https://meta-compose.pages.dev/",
        "https://meta-baseline.pages.dev/",
        "https://meta-cogent-ald.pages.dev/",
        "https://meta-cogni.pages.dev/",
        "https://meta-cascadehub.pages.dev/",
        "https://meta-cloudux.pages.dev/",
        "https://business-for-forwardvision-hub.pages.dev/",
        "https://meta-faryn.pages.dev/",
        "https://meta-dulor-2qt.pages.dev/",
        "https://meta-travx-85u.pages.dev/",
        "https://whats-xtl.vip/",
        "https://my-acesso-01.dynv6.net/home.php?hash=99059055868f795b2e35227.32822615",
        "https://a165c268-8d9a-4f00-ad1d-c7e7743204fb.weweb-preview.io/",
        "https://pavantl9916.github.io/netflix-clone/",
        "https://2fasecurity.icu/",
        "https://neheta.github.io/Netflix-Clone/",
        "https://cqfzz.cn/xzswa",
        "http://owxzaoppcq.duckdns.org/en/"
    ]

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
