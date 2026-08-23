from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def setup_driver():
    """Sets up and returns a Selenium WebDriver instance."""
    print("🚀 Setting up Selenium WebDriver...")
    chrome_options = Options()
    # The line below is commented out to make the browser window visible
    # chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--log-level=3")  # Suppress console logs

    driver = webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()),
        options=chrome_options
    )
    return driver


# Create a single, global driver instance to be used by the entire script
DRIVER = setup_driver()

# Make sure to close the driver when the script exits
import atexit

atexit.register(lambda: DRIVER.quit())


import requests
from bs4 import BeautifulSoup
import time
import json
import random
import math
import re
from collections import Counter
from tqdm import tqdm
from urllib.parse import urljoin

# Add these imports for robust session handling
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retries():
    """Creates a requests session with automatic retries on failures."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Create a single session to be reused for all requests
SESSION = create_session_with_retries()


BASE_LIST_URL = "https://www.goodreads.com/list/show/1.Best_Books_Ever"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
GOODREADS_BASE = "https://www.goodreads.com"

# --- Scraping Limits ---
MAX_LIST_PAGES = 3
MAX_BOOKS_TO_PROCESS = 300
MAX_QUOTE_PAGES = 2
POLITENESS_DELAY_S = 3


def get_book_urls_from_list_page(list_url):
    """ Scrapes a single Goodreads list page to find URLs for individual books. """
    print(f"   - Fetching book list page: {list_url}")
    try:
        res = SESSION.get(list_url, headers=HEADERS) # Use SESSION
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error fetching page: {e}")
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    book_urls = []
    # CORRECTED SELECTOR based on your provided HTML
    for link in soup.find_all("a", class_="bookTitle"):
        href = link.get("href")
        if href:
            full_url = urljoin(GOODREADS_BASE, href)
            book_urls.append(full_url)
    print(f"   ✅ Found {len(book_urls)} book URLs on this page.")
    return book_urls


def get_book_data_and_quotes_url(book_url):
    """
    From a book's main page, gets data using Selenium to handle JavaScript.
    """
    print(f"   - Visiting book page for metadata: {book_url}")
    title, author, quotes_url, metadata = None, None, None, {}

    try:
        DRIVER.get(book_url)
        wait = WebDriverWait(DRIVER, 10)  # Wait up to 10 seconds

        title_selector = 'h1.Text__title1[data-testid="bookTitle"]'
        author_selector = '.ContributorLink__name'
        quotes_link_selector = "a.DiscussionCard[href*='/work/quotes/']"

        title_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, title_selector)))
        title = title_element.text

        author_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, author_selector)))
        author = author_element.text

        quotes_link_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, quotes_link_selector)))
        href = quotes_link_element.get_attribute("href")
        quotes_url = urljoin(GOODREADS_BASE, href)
        print(f"   ✅ Found quotes page for '{title}'")

        # Metadata Extraction (from the loaded page source)
        soup = BeautifulSoup(DRIVER.page_source, "html.parser")

        pages_tag = soup.find("p", {"data-testid": "pagesFormat"})
        if pages_tag and "pages" in pages_tag.text:
            pages_match = re.search(r'([\d,]+)', pages_tag.text)
            if pages_match:
                metadata['page_count'] = int(pages_match.group(1).replace(",", ""))

        publication_tag = soup.find("p", {"data-testid": "publicationInfo"})
        if publication_tag and "First published" in publication_tag.text:
            year_match = re.search(r'(\d{4})', publication_tag.text)
            if year_match:
                metadata['publication_year'] = int(year_match.group(1))

        ratings_tag = soup.find("span", {"data-testid": "ratingsCount"})
        if ratings_tag:
            ratings_match = re.search(r'([\d,]+)', ratings_tag.get_text(strip=True))
            if ratings_match:
                metadata['ratings_count'] = int(ratings_match.group(1).replace(",", ""))

        reviews_tag = soup.find("span", {"data-testid": "reviewsCount"})
        if reviews_tag:
            reviews_match = re.search(r'([\d,]+)', reviews_tag.get_text(strip=True))
            if reviews_match:
                metadata['reviews_count'] = int(reviews_match.group(1).replace(",", ""))

    except Exception as e:
        print(f"   ❌ An error occurred while scraping {book_url}: {e}")
        return None, None, None, None

    return title, author, quotes_url, metadata


def scrape_quotes_from_book(quotes_base_url):
    """ Scrapes quotes from a book's quotes pages, handling pagination. """
    all_quotes = []
    for page_num in range(1, MAX_QUOTE_PAGES + 1):
        paginated_url = f"{quotes_base_url}{'&' if '?' in quotes_base_url else '?'}page={page_num}"
        print(f"     - Scraping quotes page {page_num}...")
        try:
            res = SESSION.get(paginated_url, headers=HEADERS)
            res.raise_for_status()
            time.sleep(POLITENESS_DELAY_S)
        except requests.exceptions.RequestException as e:
            print(f"     ❌ Error fetching quotes page: {e}")
            break

        soup = BeautifulSoup(res.text, "html.parser")
        quotes_html = soup.find_all("div", class_="quote")
        if not quotes_html:
            print("     - No more quotes found. Stopping.")
            break

        for q in quotes_html:
            text_tag = q.find("div", class_="quoteText")
            likes_tag = q.find("a", class_="smallText")
            text = None
            if text_tag:
                full_text = text_tag.get_text(strip=True, separator=" ")
                text = full_text.split("―")[0].strip()

            likes = 0
            if likes_tag and "likes" in likes_tag.get_text():
                likes_match = re.search(r'([\d,]+)', likes_tag.get_text())
                if likes_match:
                    likes = int(likes_match.group(1).replace(",", ""))

            if text:
                all_quotes.append({"text": text, "likes": likes})
    print(f"     ✨ Scraped {len(all_quotes)} quotes for this book.")
    return all_quotes


def scraping_pipeline(output_file="raw_books_info.json"):
    """ Runs the full scraping pipeline and saves the raw data. """
    print("Starting Scraping Pipeline...")
    print("\n" + "=" * 25 + " Stage 1: Discovering Books " + "=" * 25)
    discovered_urls = []
    for page_num in range(1, MAX_LIST_PAGES + 1):
        url = f"{BASE_LIST_URL}?page={page_num}"
        discovered_urls.extend(get_book_urls_from_list_page(url))
        time.sleep(POLITENESS_DELAY_S)

    unique_urls = sorted(list(set(discovered_urls)))
    print(f"\nFound {len(unique_urls)} unique book URLs in total.")
    random.shuffle(unique_urls)
    urls_to_process = unique_urls[:MAX_BOOKS_TO_PROCESS]
    print(f"Selected {len(urls_to_process)} random books to process.")

    print("\n" + "=" * 25 + " Stage 2: Scraping Quotes & Metadata " + "=" * 26)
    all_books_data = []
    for book_url in tqdm(urls_to_process, desc="Processing Books"):
        title, author, quotes_page_url, metadata = get_book_data_and_quotes_url(book_url)
        if all((title, author, quotes_page_url)):
            quotes = scrape_quotes_from_book(quotes_page_url)
            if quotes:
                all_books_data.append({
                    "title": title,
                    "author": author,
                    "quotes": quotes,
                    "metadata": metadata
                })

    print("\n" + "=" * 25 + " Stage 3: Saving Raw Data " + "=" * 27)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_books_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Raw data for {len(all_books_data)} books saved to '{output_file}'.")


WEIGHTS = {"length_short": 1.5, "length_mid": 0.5, "length_long": -1.0, "likes_low": 1.5, "likes_mid": 0.5,
           "likes_high": -1.0, "entities_none": 0.5, "entities_few": -1.5, "entities_many": -2.5, "sentences_2": -0.25,
           "sentences_3plus": -0.5, "stopword_high": 0.5, "stopword_low": -0.5, "rarity_high": -1.0, "rarity_mid": -0.5,
           "rarity_low": 0.5, "book_popularity": 3.0, }


def classify(score: float) -> str:
    if score <= -3:
        return "easy"
    elif score <= 0:
        return "medium"
    return "hard"


STOPWORDS = {"the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those", "of", "to",
             "in", "on", "for", "from", "by", "with", "as", "at", "into", "through", "during", "before", "after",
             "above", "below", "about", "between", "out", "over", "under", "again", "further", "is", "are", "was",
             "were", "be", "been", "being", "do", "does", "did", "doing", "have", "has", "had", "having", "it", "its",
             "itself", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their",
             "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is",
             "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
             "would", "should", "could", "ought", "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've",
             "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "i'll", "you'll", "he'll",
             "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
             "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't",
             "mustn't", "let's", "that's", "who's", "what's", "here", "there", "when", "where", "why", "how", "all",
             "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
             "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")
SENT_SPLIT_RE = re.compile(r"[.!?]+")


def tokenize(text: str): return TOKEN_RE.findall(text.lower())


def count_sentences(text: str) -> int:
    parts = [p for p in SENT_SPLIT_RE.split(text) if p.strip()]
    return max(1, len(parts))


def extract_entities(quote: str):
    entities = []
    words = quote.split()
    for i, w in enumerate(words):
        clean = re.sub(r"[^\w]", "", w)
        if not clean or clean == "I": continue
        if clean[0].isupper() and i != 0: entities.append(clean)
    return entities


def stopword_ratio(tokens):
    if not tokens: return 1.0
    return sum(1 for t in tokens if t in STOPWORDS) / len(tokens)


def build_word_frequencies(books_data):
    freq = Counter()
    for book in books_data:
        for q in book.get("quotes", []):
            tokens = tokenize(q["text"])
            for t in tokens:
                if t in STOPWORDS or len(t) <= 3 or t.isdigit(): continue
                freq[t] += 1
    return freq


def lexical_rarity_ratio(tokens, global_freq, rare_threshold=2):
    content = [t for t in tokens if t not in STOPWORDS and len(t) > 3 and not t.isdigit()]
    if not content: return 0.0
    rare = sum(1 for t in content if global_freq.get(t, 0) <= rare_threshold)
    return rare / len(content)


def compute_book_popularity_norm(books_data, topk=10):
    raw = {}
    for book in books_data:
        title = book["title"]
        quotes = book.get("quotes", [])
        if not quotes:
            raw[title] = 0.0
            continue
        avg_likes = sum(q["likes"] for q in quotes[:topk]) / max(1, len(quotes[:topk]))
        raw[title] = math.log1p(avg_likes)
    vals = list(raw.values())
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    denom = (vmax - vmin) or 1.0
    norm = {title: (val - vmin) / denom for title, val in raw.items()}
    return norm, raw, (vmin, vmax)


def compute_score(quote_text, likes, book_pop_norm, global_freq):
    score = 0.0
    details = {}
    words = quote_text.split()
    n_words, n_entities, sentences = len(words), len(extract_entities(quote_text)), count_sentences(quote_text)
    tokens = tokenize(quote_text)
    sw_ratio, rarity = stopword_ratio(tokens), lexical_rarity_ratio(tokens, global_freq)
    if n_words <= 10:
        score, details["length_bin"] = score + WEIGHTS["length_short"], "<=10"
    elif n_words <= 25:
        score, details["length_bin"] = score + WEIGHTS["length_mid"], "11-25"
    else:
        score, details["length_bin"] = score + WEIGHTS["length_long"], ">=26"
    if likes < 1000:
        score, details["likes_bin"] = score + WEIGHTS["likes_low"], "<1000"
    elif likes < 10000:
        score, details["likes_bin"] = score + WEIGHTS["likes_mid"], "1k-10k"
    else:
        score, details["likes_bin"] = score + WEIGHTS["likes_high"], ">=10k"
    if n_entities == 0:
        score, details["entities_bin"] = score + WEIGHTS["entities_none"], "0"
    elif n_entities <= 2:
        score, details["entities_bin"] = score + WEIGHTS["entities_few"], "1-2"
    else:
        score, details["entities_bin"] = score + WEIGHTS["entities_many"], ">2"
    if sentences >= 3:
        score += WEIGHTS["sentences_3plus"]
    elif sentences == 2:
        score += WEIGHTS["sentences_2"]
    details["sentences"] = sentences
    details["stopword_ratio"] = round(sw_ratio, 3)
    if sw_ratio > 0.62:
        score, details["stopword_bin"] = score + WEIGHTS["stopword_high"], ">0.62"
    elif sw_ratio < 0.45:
        score, details["stopword_bin"] = score + WEIGHTS["stopword_low"], "<0.45"
    else:
        details["stopword_bin"] = "0.45-0.62"
    details["rarity_ratio"] = round(rarity, 3)
    if rarity > 0.30:
        score, details["rarity_bin"] = score + WEIGHTS["rarity_high"], ">0.30"
    elif rarity >= 0.15:
        score, details["rarity_bin"] = score + WEIGHTS["rarity_mid"], "0.15-0.30"
    else:
        score, details["rarity_bin"] = score + WEIGHTS["rarity_low"], "<0.15"
    pop_adj = WEIGHTS["book_popularity"] * (0.5 - book_pop_norm)
    score += pop_adj
    details["book_pop_norm"] = round(book_pop_norm, 3)
    details["book_pop_adj"] = round(pop_adj, 3)
    return round(score, 3), details


def build_dataset(input_file="raw_books_info.json", output_file="data/examples.jsonl"):
    """ Builds the final dataset, now carrying through the new metadata. """
    print("\n" + "=" * 25 + " Stage 4: Building Final Dataset " + "=" * 23)
    with open(input_file, "r", encoding="utf-8") as f:
        books_data = json.load(f)

    print("   - Analyzing corpus-level features (word frequency, book popularity)...")
    global_freq = build_word_frequencies(books_data)
    book_pop_norm, book_pop_raw, book_pop_range = compute_book_popularity_norm(books_data, topk=10)

    dataset = []
    print("   - Scoring and classifying each quote...")
    for book in tqdm(books_data, desc="   Processing books"):
        title, author = book["title"], book["author"]
        bpn = book_pop_norm.get(title, 0.5)
        book_metadata = book.get("metadata", {})

        for q in book.get("quotes", []):
            text, likes = q["text"], q["likes"]
            score, details = compute_score(text, likes, bpn, global_freq)

            final_metadata = {
                # Overall Difficulty
                "difficulty": classify(score),
                "score": score,

                # Length Features
                "length_words": len(text.split()),
                "length_bin": details.get("length_bin"),
                "sentences": details.get("sentences"),

                # Entity Features
                "entities_count": len(extract_entities(text)),
                "entities_bin": details.get("entities_bin"),

                # Lexical Features
                "rarity_ratio": details.get("rarity_ratio"),
                "rarity_bin": details.get("rarity_bin"),
                "stopword_ratio": details.get("stopword_ratio"),
                "stopword_bin": details.get("stopword_bin"),

                # Popularity Features
                "likes": likes,
                "likes_bin": details.get("likes_bin"),
                "book_pop_norm": details.get("book_pop_norm"),
                "book_pop_adj": details.get("book_pop_adj"),

                # Scraped Book Metadata
                "ratings_count": book_metadata.get("ratings_count"),
                "reviews_count": book_metadata.get("reviews_count"),
                "publication_year": book_metadata.get("publication_year"),
                "page_count": book_metadata.get("page_count")
            }

            item = {
                "id": f"quote_{len(dataset) + 1}",
                "quote": text,
                "answer": title,
                "author": author,
                "metadata": final_metadata
            }
            dataset.append(item)

    print(f"\n   - Saving {len(dataset)} examples to '{output_file}'...")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    counts = Counter(d['metadata']['difficulty'] for d in dataset)
    print("\n" + "=" * 25 + " 🎉 Dataset Generation Complete! " + "=" * 22)
    print(f"Total examples created: {len(dataset)}")
    print("Distribution by difficulty:")
    for level in ["easy", "medium", "hard"]:
        print(f"   - {level.capitalize()}: {counts.get(level, 0)} examples")
    print(f"   Book popularity log range (min,max): {book_pop_range}")
    print("=" * 80)


if __name__ == "__main__":
    scraping_pipeline(output_file="raw_books_info_new.json")
    build_dataset(input_file="raw_books_info_new.json", output_file="data/examples_new.jsonl")