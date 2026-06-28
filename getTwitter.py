import csv
import glob
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from time import sleep
from urllib.parse import quote

import httpx

from transaction_generate import get_transaction_id, get_url_path

DEFAULT_SEARCH_TERMS = [
    "social media ban minors",
    "gaming restrictions minors",
    "screen time policy",
    "protect children online"
]

DEFAULT_TOPIC_LABEL = "Digital Restrictions for Minors"
DEFAULT_MAX_TWEETS = 100
DEFAULT_LOOKBACK_DAYS = 14
PAGE_SIZE = 20
PRODUCT = "Latest"
REQUEST_INTERVAL_SECONDS = 2
LONG_PAUSE_EVERY_PAGES = 10
LONG_PAUSE_SECONDS = 15
TRANSACTION_ID_RETRY_COUNT = 3
TRANSACTION_ID_RETRY_SLEEP_SECONDS = 5
REQUEST_RETRY_COUNT = 5
REQUEST_RETRY_SLEEP_SECONDS = 5
SEARCH_OPERATION_IDS = [
    "AIdc203rPpK_k_2KWSdm7g",
    "yiE17ccAAu3qwM34bPYZkQ",
]
SEARCH_HOSTS = [
    "https://x.com",
    "https://twitter.com",
]


def normalize_search_terms(raw_terms):
    if isinstance(raw_terms, str):
        raw_terms = [item.strip() for item in raw_terms.split(",")]

    terms = []
    for term in raw_terms or []:
        term = str(term).strip()
        if not term:
            continue
        if term.startswith("(") and term.endswith(")"):
            terms.append(term)
        elif " OR " in term:
            terms.append(term)
        elif " " in term:
            terms.append(f'"{term}"')
        else:
            terms.append(term)
    return terms


def slugify_topic(text):
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "topic"


def stamp2time(msecs_stamp):
    time_array = time.localtime(msecs_stamp / 1000)
    return time.strftime("%Y-%m-%d %H:%M", time_array)


def time2stamp(timestr):
    datetime_obj = datetime.strptime(timestr, "%Y-%m-%d")
    msecs_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return msecs_stamp


def load_settings():
    with open("settings.json", "r", encoding="utf-8") as f:
        settings = json.load(f)

    save_root = settings.get("save_path") or os.getcwd()
    proxy = settings.get("proxy") or None
    cookies = []

    cookie_keys = sorted(key for key in settings if re.fullmatch(r"cookie_\d+", key))
    for key in cookie_keys:
        cookie_value = (settings.get(key) or "").strip()
        if cookie_value:
            cookies.append({"name": key, "value": cookie_value})

    if not cookies:
        cookie = (settings.get("cookie") or "").strip()
        if cookie:
            cookies.append({"name": "cookie", "value": cookie})

    if not cookies:
        raise ValueError("settings.json 里的 cookie / cookie_0 / cookie_1 为空，无法请求 X/Twitter API。")

    search_terms = normalize_search_terms(
        settings.get("search_terms")
        or settings.get("keyword")
        or DEFAULT_SEARCH_TERMS
    )
    if not search_terms:
        search_terms = normalize_search_terms(DEFAULT_SEARCH_TERMS)

    topic_label = (settings.get("topic_label") or settings.get("topic") or DEFAULT_TOPIC_LABEL).strip()
    max_tweets = int(settings.get("max_tweets") or DEFAULT_MAX_TWEETS)
    lookback_days = int(settings.get("lookback_days") or DEFAULT_LOOKBACK_DAYS)
    language = (settings.get("language") or "en").strip()
    exclude_retweets = settings.get("exclude_retweets", True)

    return {
        "save_root": save_root,
        "proxy": proxy,
        "cookies": cookies,
        "search_terms": search_terms,
        "topic_label": topic_label,
        "max_tweets": max_tweets,
        "lookback_days": lookback_days,
        "language": language,
        "exclude_retweets": exclude_retweets,
    }


def build_headers(cookie, query, referer_host):
    token_match = re.findall(r"ct0=(.*?);", cookie)
    if not token_match:
        raise ValueError("cookie 中未找到 ct0，无法生成 x-csrf-token。")

    return {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "authorization": (
            "Bearer "
            "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D"
            "1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
        ),
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "origin": referer_host,
        "cookie": cookie,
        "x-csrf-token": token_match[0],
        "referer": f"{referer_host}/search?q={quote(query)}&src=typed_query&f=live",
    }


def apply_cookie_to_client(client, cookie_info, query, referer_host):
    client.headers.update(build_headers(cookie_info["value"], query, referer_host))


def get_next_cookie_index(current_index, cookies):
    if not cookies:
        return current_index
    return (current_index + 1) % len(cookies)


def create_transaction_client(proxy):
    last_error = None
    for attempt in range(1, TRANSACTION_ID_RETRY_COUNT + 1):
        try:
            return get_transaction_id(proxy=proxy)
        except Exception as e:
            last_error = e
            print(f"生成 x-client-transaction-id 失败，第 {attempt}/{TRANSACTION_ID_RETRY_COUNT} 次重试。")
            if attempt < TRANSACTION_ID_RETRY_COUNT:
                sleep(TRANSACTION_ID_RETRY_SLEEP_SECONDS * attempt)
    print(f"生成 x-client-transaction-id 最终失败，将降级为无 transaction-id 模式。错误: {last_error}")
    return None


def create_http_client(proxy, cookie_info, query, referer_host):
    transport = httpx.HTTPTransport(retries=3)
    client = httpx.Client(
        proxy=proxy,
        timeout=30.0,
        verify=False,
        http2=False,
        transport=transport,
    )
    apply_cookie_to_client(client, cookie_info, query, referer_host)
    return client


def apply_transaction_id(client, tx_client, url, method="GET"):
    if tx_client is None:
        client.headers.pop("x-client-transaction-id", None)
        return
    path = get_url_path(url)
    client.headers["x-client-transaction-id"] = tx_client.generate_transaction_id(
        method=method,
        path=path,
    )


def build_query(search_terms, start_date, until_date, language="en", exclude_retweets=True):
    if len(search_terms) == 1:
        term_clause = search_terms[0]
    else:
        term_clause = "(" + " OR ".join(search_terms) + ")"

    parts = [term_clause, f"since:{start_date}", f"until:{until_date}"]
    if language:
        parts.append(f"lang:{language}")
    if exclude_retweets:
        parts.append("-is:retweet")
    return " ".join(parts)


def make_output_dir(save_root, topic_label, start_date, until_date):
    topic_slug = slugify_topic(topic_label)
    folder = os.path.join(save_root, f"{topic_slug}_{start_date}_to_{until_date}")
    os.makedirs(folder, exist_ok=True)
    return folder, topic_slug


def create_csv(output_dir, topic_label, query, max_tweets, lookback_days):
    csv_path = os.path.join(
        output_dir,
        f"{slugify_topic(topic_label)}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_annotation.csv",
    )
    f = open(csv_path, "w", encoding="utf-8-sig", newline="")
    writer = csv.writer(f)
    writer.writerow(["Topic", topic_label])
    writer.writerow(["Query", query])
    writer.writerow(["Save Path", output_dir])
    writer.writerow(
        [
            "Tweet ID",
            "Topic",
            "Tweet Date",
            "Display Name",
            "User Name",
            "Tweet URL",
            "Tweet Content",
            "Favorite Count",
            "Retweet Count",
            "Reply Count",
            "Gold Label",
            "Notes",
        ]
    )
    return f, writer, csv_path


def append_csv(csv_path):
    f = open(csv_path, "a", encoding="utf-8-sig", newline="")
    writer = csv.writer(f)
    return f, writer


def load_existing_tweet_ids(csv_path):
    tweet_ids = set()
    if not os.path.exists(csv_path):
        return tweet_ids

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index < 4 or not row:
                continue
            tweet_ids.add(row[0])
    return tweet_ids


def load_state(state_path):
    if not os.path.exists(state_path):
        return None

    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path, state):
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_default_date_range(lookback_days):
    today = date.today()
    start_date = today - timedelta(days=lookback_days)
    until_date = today + timedelta(days=1)
    return start_date.isoformat(), until_date.isoformat()


def find_resume_state(save_root, topic_slug):
    pattern = os.path.join(save_root, "*", "state.json")
    candidates = []
    for state_path in glob.glob(pattern):
        state = load_state(state_path)
        if not state or state.get("completed"):
            continue
        if state.get("topic_slug") != topic_slug:
            continue
        if not os.path.exists(state.get("csv_path", "")):
            continue
        candidates.append((os.path.getmtime(state_path), state_path, state))

    if not candidates:
        return None, None

    candidates.sort(reverse=True)
    _, state_path, state = candidates[0]
    return state_path, state


def strip_tco(text):
    return re.sub(r"https?://t\.co/\w+\s*$", "", text).strip()


def normalize_tweet_result(tweet_result):
    if isinstance(tweet_result, dict) and "tweet" in tweet_result and isinstance(tweet_result["tweet"], dict):
        return tweet_result["tweet"]
    return tweet_result


def parse_entry(entry, topic_label):
    tweet_results = (
        entry.get("content", {})
        .get("itemContent", {})
        .get("tweet_results", {})
    )
    tweet_result = tweet_results.get("result")
    if not tweet_result:
        return None

    tweet = normalize_tweet_result(tweet_result)
    if isinstance(tweet, dict) and tweet.get("__typename") == "TweetWithVisibilityResults":
        tweet = normalize_tweet_result(tweet.get("tweet", {}))

    if not tweet or "legacy" not in tweet or "core" not in tweet:
        return None

    try:
        edit_control = tweet["edit_control"]
        if "editable_until_msecs" in edit_control:
            time_stamp = int(edit_control["editable_until_msecs"]) - 3600000
        elif "edit_control_initial" in edit_control:
            time_stamp = int(edit_control["edit_control_initial"]["editable_until_msecs"]) - 3600000
        else:
            return None

        legacy = tweet["legacy"]
        user_legacy = tweet["core"]["user_results"]["result"]["legacy"]
        tweet_id = tweet["rest_id"]
        display_name = user_legacy["name"]
        screen_name = user_legacy["screen_name"]
        tweet_url = f"https://x.com/{screen_name}/status/{tweet_id}"
        tweet_content = strip_tco(legacy.get("full_text", ""))

        return [
            tweet_id,
            topic_label,
            stamp2time(time_stamp),
            display_name,
            f"@{screen_name}",
            tweet_url,
            tweet_content,
            legacy.get("favorite_count", 0),
            legacy.get("retweet_count", 0),
            legacy.get("reply_count", 0),
            "",
            "",
        ]
    except Exception:
        return None


def extract_cursor_and_entries(raw_data):
    instructions = (
        raw_data.get("data", {})
        .get("search_by_raw_query", {})
        .get("search_timeline", {})
        .get("timeline", {})
        .get("instructions", [])
    )
    next_cursor = None
    entries = []

    for instruction in instructions:
        if "entries" in instruction:
            for entry in instruction["entries"]:
                entry_id = entry.get("entryId", "")
                if entry_id.startswith("cursor-bottom"):
                    next_cursor = entry.get("content", {}).get("value", next_cursor)
                elif entry_id.startswith("tweet-"):
                    entries.append(entry)
        if "entry" in instruction:
            entry = instruction["entry"]
            if entry.get("entryId", "").startswith("cursor-bottom"):
                next_cursor = entry.get("content", {}).get("value", next_cursor)

    return next_cursor, entries


def build_search_request(query, cursor, operation_id, host):
    variables = json.dumps(
        {
            "rawQuery": query,
            "count": PAGE_SIZE,
            "cursor": cursor,
            "querySource": "typed_query",
            "product": PRODUCT,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    features = json.dumps(
        {
            "rweb_video_screen_enabled": False,
            "profile_label_improvements_pcf_label_in_post_enabled": True,
            "rweb_tipjar_consumption_enabled": True,
            "verified_phone_label_enabled": False,
            "creator_subscriptions_tweet_preview_api_enabled": True,
            "responsive_web_graphql_timeline_navigation_enabled": True,
            "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
            "premium_content_api_read_enabled": False,
            "communities_web_enable_tweet_community_results_fetch": True,
            "c9s_tweet_anatomy_moderator_badge_enabled": True,
            "responsive_web_grok_analyze_button_fetch_trends_enabled": False,
            "responsive_web_grok_analyze_post_followups_enabled": True,
            "responsive_web_jetfuel_frame": False,
            "responsive_web_grok_share_attachment_enabled": True,
            "articles_preview_enabled": True,
            "responsive_web_edit_tweet_api_enabled": True,
            "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
            "view_counts_everywhere_api_enabled": True,
            "longform_notetweets_consumption_enabled": True,
            "responsive_web_twitter_article_tweet_consumption_enabled": True,
            "tweet_awards_web_tipping_enabled": False,
            "responsive_web_grok_show_grok_translated_post": False,
            "responsive_web_grok_analysis_button_from_backend": False,
            "creator_subscriptions_quote_tweet_preview_enabled": False,
            "freedom_of_speech_not_reach_fetch_enabled": True,
            "standardized_nudges_misinfo": True,
            "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
            "longform_notetweets_rich_text_read_enabled": True,
            "longform_notetweets_inline_media_enabled": True,
            "responsive_web_grok_image_annotation_enabled": True,
            "responsive_web_enhance_cards_enabled": False,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    url = f"{host}/i/api/graphql/{operation_id}/SearchTimeline"
    params = {
        "variables": variables,
        "features": features,
    }
    return url, params


def build_search_urls(query, cursor):
    urls = []
    for host in SEARCH_HOSTS:
        for operation_id in SEARCH_OPERATION_IDS:
            urls.append(build_search_request(query, cursor, operation_id, host))
    return urls


def main():
    settings = load_settings()
    cookies = settings["cookies"]
    search_terms = settings["search_terms"]
    topic_label = settings["topic_label"]
    max_tweets = settings["max_tweets"]
    lookback_days = settings["lookback_days"]
    language = settings["language"]
    exclude_retweets = settings["exclude_retweets"]
    topic_slug = slugify_topic(topic_label)

    state_path, state = find_resume_state(settings["save_root"], topic_slug)

    if state:
        output_dir = os.path.dirname(state_path)
        start_date = state["start_date"]
        until_date = state["until_date"]
        query = state["query"]
    else:
        start_date, until_date = get_default_date_range(lookback_days)
        query = build_query(
            search_terms,
            start_date,
            until_date,
            language=language,
            exclude_retweets=exclude_retweets,
        )
        output_dir, topic_slug = make_output_dir(settings["save_root"], topic_label, start_date, until_date)
        state_path = os.path.join(output_dir, "state.json")

    active_cookie_index = 0
    if state:
        active_cookie_index = state.get("active_cookie_index", 0)
    active_cookie_index %= len(cookies)

    if state and state.get("query") == query and os.path.exists(state.get("csv_path", "")):
        csv_path = state["csv_path"]
        csv_file, writer = append_csv(csv_path)
        tweet_ids = load_existing_tweet_ids(csv_path)
        cursor = state.get("cursor", "")
        page = state.get("page", 0)
        print(f"检测到断点状态，从第 {page} 页后继续。")
        print(f"已加载 {len(tweet_ids)} 条历史记录。")
    else:
        csv_file, writer, csv_path = create_csv(output_dir, topic_label, query, max_tweets, lookback_days)
        tweet_ids = set()
        cursor = ""
        page = 0
        save_state(
            state_path,
            {
                "topic_label": topic_label,
                "topic_slug": topic_slug,
                "search_terms": search_terms,
                "query": query,
                "start_date": start_date,
                "until_date": until_date,
                "csv_path": csv_path,
                "cursor": cursor,
                "page": page,
                "saved_count": 0,
                "completed": False,
                "active_cookie_index": active_cookie_index,
            },
        )

    print(f"搜索语句: {query}")
    print(f"输出目录: {output_dir}")
    print(f"当前账号: {cookies[active_cookie_index]['name']}")
    tx_client = None
    client = create_http_client(settings["proxy"], cookies[active_cookie_index], query, SEARCH_HOSTS[0])

    try:
        while len(tweet_ids) < max_tweets:
            page += 1
            candidate_urls = build_search_urls(query, cursor)

            attempted_cookie_indexes = set()
            switched_account_for_current_page = False
            restart_from_top = False

            for url, params in candidate_urls:
                attempted_cookie_indexes.add(active_cookie_index)
                host_match = re.match(r"https?://[^/]+", url)
                referer_host = host_match.group(0) if host_match else SEARCH_HOSTS[0]
                response = None
                network_error = None

                try:
                    try:
                        client.close()
                    except Exception:
                        pass
                    client = create_http_client(settings["proxy"], cookies[active_cookie_index], query, referer_host)
                except Exception:
                    client = create_http_client(settings["proxy"], cookies[active_cookie_index], query, SEARCH_HOSTS[0])

                apply_transaction_id(client, tx_client, url, method="POST")

                for request_attempt in range(1, REQUEST_RETRY_COUNT + 1):
                    try:
                        response = client.post(url, params=params)
                        network_error = None
                        break
                    except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                        network_error = e
                        print(
                            f"网络请求失败，第 {request_attempt}/{REQUEST_RETRY_COUNT} 次重试。"
                            f"{e.__class__.__name__}: {e}"
                        )
                        try:
                            client.close()
                        except Exception:
                            pass
                        client = create_http_client(settings["proxy"], cookies[active_cookie_index], query, referer_host)
                        apply_transaction_id(client, tx_client, url, method="POST")
                        if request_attempt < REQUEST_RETRY_COUNT:
                            sleep(REQUEST_RETRY_SLEEP_SECONDS * request_attempt)

                if network_error is not None:
                    continue

                if response is None:
                    continue

                if response.status_code == 429:
                    print(f"请求失败: HTTP 429，当前账号 {cookies[active_cookie_index]['name']} 被限流。")
                    if len(attempted_cookie_indexes) < len(cookies):
                        next_cookie_index = get_next_cookie_index(active_cookie_index, cookies)
                        while next_cookie_index in attempted_cookie_indexes:
                            next_cookie_index = get_next_cookie_index(next_cookie_index, cookies)
                        active_cookie_index = next_cookie_index
                        switched_account_for_current_page = True
                        save_state(
                            state_path,
                            {
                                "topic_label": topic_label,
                                "topic_slug": topic_slug,
                                "search_terms": search_terms,
                                "query": query,
                                "start_date": start_date,
                                "until_date": until_date,
                                "csv_path": csv_path,
                                "cursor": cursor,
                                "page": page - 1,
                                "saved_count": len(tweet_ids),
                                "completed": False,
                                "active_cookie_index": active_cookie_index,
                            },
                        )
                        print(f"已切换到账号 {cookies[active_cookie_index]['name']}，5 秒后继续。")
                        sleep(5)
                        switched_account_for_current_page = True
                        break
                    print("所有可用账号都遇到 429，请稍后再试。")
                    return

                if response.status_code in (400, 404):
                    print(f"候选请求失败: HTTP {response.status_code}，url={url}")
                    print(response.text[:500])
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    print(f"请求失败: HTTP {response.status_code}")
                    print(response.text[:1000])
                    return
                break

            if switched_account_for_current_page:
                continue

            if response is None:
                print("所有候选请求都失败了，已保存进度后停止。")
                save_state(
                    state_path,
                    {
                        "topic_label": topic_label,
                        "topic_slug": topic_slug,
                        "search_terms": search_terms,
                        "query": query,
                        "start_date": start_date,
                        "until_date": until_date,
                        "csv_path": csv_path,
                        "cursor": cursor,
                        "page": page - 1,
                        "saved_count": len(tweet_ids),
                        "completed": False,
                        "active_cookie_index": active_cookie_index,
                    },
                )
                return

            if restart_from_top:
                sleep(REQUEST_INTERVAL_SECONDS)
                continue

            try:
                raw_data = response.json()
            except Exception:
                print("返回内容不是合法 JSON:")
                print(response.text[:500])
                break

            next_cursor, entries = extract_cursor_and_entries(raw_data)
            if not entries:
                print("没有更多推文了，提前结束。")
                break

            added_this_page = 0
            for entry in entries:
                row = parse_entry(entry, topic_label)
                if not row:
                    continue
                tweet_id = row[0]
                if tweet_id in tweet_ids:
                    continue
                tweet_ids.add(tweet_id)
                writer.writerow(row)
                added_this_page += 1
                if len(tweet_ids) >= max_tweets:
                    break

            csv_file.flush()
            save_state(
                state_path,
                {
                    "topic_label": topic_label,
                    "topic_slug": topic_slug,
                    "search_terms": search_terms,
                    "query": query,
                    "start_date": start_date,
                    "until_date": until_date,
                    "csv_path": csv_path,
                    "cursor": next_cursor or cursor,
                    "page": page,
                    "saved_count": len(tweet_ids),
                    "completed": False,
                    "active_cookie_index": active_cookie_index,
                },
            )
            print(
                f"第 {page} 页新增 {added_this_page} 条，累计 {len(tweet_ids)} 条，"
                f"下一游标: {'有' if next_cursor else '无'}"
            )

            sleep(REQUEST_INTERVAL_SECONDS)
            if page % LONG_PAUSE_EVERY_PAGES == 0:
                print(f"已完成 {page} 页，主动暂停 {LONG_PAUSE_SECONDS} 秒。")
                sleep(LONG_PAUSE_SECONDS)

            if not next_cursor or next_cursor == cursor:
                print("游标没有继续推进，停止抓取。")
                break
            cursor = next_cursor

    finally:
        client.close()
        csv_file.close()

    completed = len(tweet_ids) >= max_tweets
    save_state(
        state_path,
        {
            "topic_label": topic_label,
            "topic_slug": topic_slug,
            "search_terms": search_terms,
            "query": query,
            "start_date": start_date,
            "until_date": until_date,
            "csv_path": csv_path,
            "cursor": cursor,
            "page": page,
            "saved_count": len(tweet_ids),
            "completed": completed,
            "active_cookie_index": active_cookie_index,
        },
    )

    print(f"抓取完成，共保存 {len(tweet_ids)} 条，时间范围为 {start_date} 至 {until_date}。")
    print(f"CSV 文件: {csv_path}")


if __name__ == "__main__":
    main()
