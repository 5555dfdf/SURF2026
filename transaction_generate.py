import requests
from x_client_transaction.utils import handle_x_migration, get_ondemand_file_url, generate_headers
from x_client_transaction import ClientTransaction
import bs4
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _build_ondemand_url_from_html(html_text):
    patterns = [
        r'ondemand\.s\.([0-9a-f]+)a\.js',
        r'https://abs\.twimg\.com/responsive-web/client-web/ondemand\.s\.([0-9a-f]+)a\.js',
        r'https://abs\.twitter\.com/responsive-web/client-web/ondemand\.s\.([0-9a-f]+)a\.js',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE)
        if match:
            filename = match.group(1)
            return f"https://abs.twimg.com/responsive-web/client-web/ondemand.s.{filename}a.js"
    return None


def get_url_path(url):
    path = re.findall(r'https?://(?:x|twitter)\.com(.*?)\?', url)[0]
    return path

def get_transaction_id(proxy=None, timeout=30, max_retries=3, verify=False):
    # https://github.com/iSarabjitDhiman/XClientTransaction

    session = requests.Session()
    session.headers = generate_headers()
    session.verify = verify
    if proxy:
        session.proxies.update({
            "http": proxy,
            "https": proxy,
        })

    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    home_page_response = handle_x_migration(session=session)
    home_page = session.get(url="https://x.com", timeout=timeout, verify=verify)
    home_page_response = bs4.BeautifulSoup(home_page.content, 'html.parser')
    try:
        ondemand_file_url = get_ondemand_file_url(response=home_page_response)
    except Exception:
        ondemand_file_url = _build_ondemand_url_from_html(str(home_page_response))
    if not ondemand_file_url:
        raise RuntimeError("无法从 X 首页解析 ondemand 文件 URL，transaction-id 生成失败")
    ondemand_file = session.get(url=ondemand_file_url, timeout=timeout, verify=verify)
    ondemand_file_response = bs4.BeautifulSoup(ondemand_file.content, 'html.parser')
    ondemand_file_response = ondemand_file

    ct = ClientTransaction(home_page_response,ondemand_file_response)
    return ct
