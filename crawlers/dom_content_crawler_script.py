from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def get_stable_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--disable-popup-blocking")
    opts.add_argument("--ignore-certificate-errors")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=opts)

def safe_execute(driver, script, default=0):
    try:
        return driver.execute_script(script)
    except Exception:
        return default

def load_page(driver, url, timeout=25):
    driver.get(url)
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    last_html = None
    stable_start = None
    end_time = time.time() + timeout
    while time.time() < end_time:
        current_html = driver.page_source
        if current_html == last_html:
            if stable_start is None:
                stable_start = time.time()
            elif time.time() - stable_start >= 1.5:
                break
        else:
            last_html = current_html
            stable_start = None
        time.sleep(0.5)

def dom_max_depth(driver):
    js = """
    return (function(){
      function getDepth(node){
        if(!node || !node.children) return 1;
        let max = 0;
        for(let i = 0; i < node.children.length; i++){
          max = Math.max(max, getDepth(node.children[i]));
        }
        return 1 + max;
      }
      return getDepth(document.documentElement);
    })();
    """
    return safe_execute(driver, js)

def dom_total_nodes(driver):
    return safe_execute(driver, "return document.getElementsByTagName('*').length;")

def dom_avg_branching_factor(driver):
    js = """
    let nodes = document.getElementsByTagName('*');
    let totalChildren = 0;
    for (let n of nodes) {
        totalChildren += n.children.length;
    };
    return totalChildren / Math.max(nodes.length, 1);
    """
    return safe_execute(driver, js)

def dom_iframe_count(driver):
    return safe_execute(driver, "return document.getElementsByTagName('iframe').length;")

def dom_form_count(driver):
    return safe_execute(driver, "return document.getElementsByTagName('form').length;")

def dom_hidden_input_count(driver):
    js = "return document.querySelectorAll('input[type=hidden]').length;"
    return safe_execute(driver, js)

def dom_forms_external_action_ratio(driver):
    js = """
    let forms = document.getElementsByTagName('form');
    if (forms.length === 0) return 0;
    let count = 0;
    let currentDomain = location.hostname;
    for (let f of forms){
      if (f.action && (new URL(f.action, location.href)).hostname !== currentDomain)
          count++;
    }
    return count / forms.length;
    """
    return safe_execute(driver, js)

def dom_script_count(driver):
    return safe_execute(driver, "return document.getElementsByTagName('script').length;")

def dom_suspicious_script_count(driver):
    js = """
    let scripts = document.getElementsByTagName('script');
    let suspicious = 0;
    for (let s of scripts){
        let txt = s.innerText || "";
        if (txt.includes("eval(") || txt.includes("setTimeout(") || txt.match(/[A-Za-z0-9]{40,}/))
            suspicious++;
    }
    return suspicious;
    """
    return safe_execute(driver, js)

def dom_popup_indicators(driver):
    js = """
    let txt = Array.from(document.scripts).map(s=>s.innerText).join('\\n');
    return /(window\\.open|alert\\(|confirm\\(|prompt\\()/.test(txt) ? 1 : 0;
    """
    return safe_execute(driver, js)

def dom_hover_url_mismatch_count(driver):
    js = """
    let links = document.getElementsByTagName('a');
    let mismatches = 0;
    for (let l of links) {
        if (l.href && l.textContent) {
            try {
                let hrefDomain = new URL(l.href, location.href).hostname.replace('www.','');
                let textDomain = (l.textContent.match(/([a-z0-9-]+\\.[a-z]+)/i)||[''])[0];
                if (textDomain && !hrefDomain.includes(textDomain)) mismatches++;
            } catch(e){}
        }
    }
    return mismatches;
    """
    return safe_execute(driver, js)

def dom_mixed_content_count(driver):
    js = """
    if (location.protocol !== 'https:') return 0;
    let tags = ['img','script','iframe','link'];
    let count = 0;
    for (let tag of tags) {
        let elems = document.getElementsByTagName(tag);
        for (let e of elems){
            if (e.src && e.src.startsWith('http:')) count++;
            if (e.href && e.href.startsWith('http:')) count++;
        }
    }
    return count;
    """
    return safe_execute(driver, js)

def dom_external_resource_count(driver):
    js = """
    let resources = Array.from(document.querySelectorAll('script[src], link[href], img[src]'));
    let currentDomain = location.hostname;
    let count = 0;
    for (let r of resources){
        let url = r.src || r.href;
        if (url && (new URL(url, location.href)).hostname !== currentDomain) count++;
    }
    return count;
    """
    return safe_execute(driver, js)

def dom_cross_origin_iframe_count(driver):
    js = """
    let iframes = document.getElementsByTagName('iframe');
    let count = 0;
    for (let f of iframes) {
        try {
            if (f.contentWindow.location.hostname !== location.hostname) count++;
        } catch(e){ count++; }
    }
    return count;
    """
    return safe_execute(driver, js)

def dom_hidden_element_ratio(driver):
    js = """
    let elems = document.getElementsByTagName('*');
    let hidden = 0;
    for (let e of elems) {
        let style = window.getComputedStyle(e);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0')
            hidden++;
    }
    return hidden / Math.max(elems.length, 1);
    """
    return safe_execute(driver, js)

def dom_clickable_without_href_count(driver):
    js = """
    let elems = document.querySelectorAll('[onclick]');
    let count = 0;
    for (let e of elems) if (!e.hasAttribute('href')) count++;
    return count;
    """
    return safe_execute(driver, js)

def dom_autocomplete_off_password_count(driver):
    js = "return document.querySelectorAll('input[type=password][autocomplete=\"off\"]').length;"
    return safe_execute(driver, js)

def dom_setTimeout_or_setInterval_presence(driver):
    js = """
    let txt = Array.from(document.scripts).map(s=>s.innerText).join('\\n');
    return /(setTimeout\\(|setInterval\\()/.test(txt) ? 1 : 0;
    """
    return safe_execute(driver, js)

def dom_mutation_observer_presence(driver):
    js = """
    let txt = Array.from(document.scripts).map(s=>s.innerText).join('\\n');
    return /MutationObserver/.test(txt) ? 1 : 0;
    """
    return safe_execute(driver, js)

def dom_service_worker_register_presence(driver):
    js = "return ('serviceWorker' in navigator) ? 1 : 0;"
    return safe_execute(driver, js)

def dom_download_link_count(driver):
    js = """
    let links = document.getElementsByTagName('a');
    let count = 0;
    for (let l of links)
        if (l.href && l.href.match(/\\.(exe|zip|rar|7z)$/i)) count++;
    return count;
    """
    return safe_execute(driver, js)

def dom_data_uri_image_count(driver):
    js = "return document.querySelectorAll('img[src^=\"data:\"]').length;"
    return safe_execute(driver, js)

def dom_form_target_blank_count(driver):
    js = "return document.querySelectorAll('form[target=\"_blank\"]').length;"
    return safe_execute(driver, js)

def dom_anchor_noopener_missing_ratio(driver):
    js = """
    let anchors = document.querySelectorAll('a[target=\"_blank\"]');
    if (anchors.length === 0) return 0;
    let missing = 0;
    for (let a of anchors)
        if (!a.rel || !a.rel.includes('noopener'))
            missing++;
    return missing / anchors.length;
    """
    return safe_execute(driver, js)

def dom_event_handler_attr_total(driver):
    js = """
    let nodes = document.getElementsByTagName('*');
    let count = 0;
    for (let n of nodes)
        for (let attr of n.attributes)
            if (attr.name.startsWith('on')) count++;
    return count;
    """
    return safe_execute(driver, js)

def dom_third_party_domains_unique(driver):
    js = """
    let resources = Array.from(document.querySelectorAll('[src],[href]'));
    let currentDomain = location.hostname;
    let domains = new Set();
    for (let r of resources) {
        let url = r.src || r.href;
        if (url){
            try{
                let host = (new URL(url, location.href)).hostname;
                if (host !== currentDomain) domains.add(host);
            } catch(e){}
        }
    }
    return domains.size;
    """
    return safe_execute(driver, js)

def dom_page_load_time(driver):
    js = "return performance.timing.loadEventEnd - performance.timing.navigationStart;"
    return safe_execute(driver, js)

def dom_anchor_ratio(driver):
    js = """
    let total = document.getElementsByTagName('*').length;
    let links = document.getElementsByTagName('a').length;
    return links / Math.max(total, 1);
    """
    return safe_execute(driver, js)