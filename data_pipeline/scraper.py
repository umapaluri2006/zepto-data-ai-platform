# -*- coding: utf-8 -*-
"""
data_pipeline/scraper.py
Scraper module for books.toscrape.com catalog data.
Extracts title, price, star_rating, availability, and category.
"""

import requests
from bs4 import BeautifulSoup
import urllib.parse
from typing import List, Dict, Any

BASE_URL = "http://books.toscrape.com/"

def get_category_links(base_url: str = BASE_URL) -> Dict[str, str]:
    """Fetch available category names and their full relative URLs."""
    response = requests.get(base_url, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    categories = {}
    cat_container = soup.select("div.side_categories ul li ul li a")
    for link in cat_container:
        cat_name = link.text.strip()
        cat_href = link.get("href")
        full_url = urllib.parse.urljoin(base_url, cat_href)
        categories[cat_name] = full_url
    return categories

def scrape_category_books(category_name: str, category_url: str) -> List[Dict[str, Any]]:
    """Scrape all books for a specific category across all its paginated pages."""
    books = []
    current_url = category_url
    
    while current_url:
        resp = requests.get(current_url, timeout=15)
        if resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        
        articles = soup.select("article.product_pod")
        for art in articles:
            # 1. Title
            title_tag = art.select_one("h3 a")
            title = title_tag.get("title") if title_tag and title_tag.get("title") else (title_tag.text.strip() if title_tag else "Unknown")
            
            # 2. Price (raw GBP string e.g. '51.77')
            price_tag = art.select_one("p.price_color")
            price_raw = price_tag.text.strip() if price_tag else "0.00"
            
            # 3. Rating text (e.g. 'star-rating Three')
            rating_tag = art.select_one("p.star-rating")
            rating_classes = rating_tag.get("class", []) if rating_tag else []
            rating_text = "Zero"
            for cls in rating_classes:
                if cls.lower() != "star-rating":
                    rating_text = cls
                    break
            
            # 4. Availability text
            avail_tag = art.select_one("p.instock.availability")
            avail_text = avail_tag.text.strip() if avail_tag else "In stock"
            
            books.append({
                "title": title,
                "price": price_raw,
                "star_rating": rating_text,
                "availability": avail_text,
                "category": category_name
            })
            
        # Check next page in pagination
        next_button = soup.select_one("li.next a")
        if next_button:
            next_href = next_button.get("href")
            current_url = urllib.parse.urljoin(current_url, next_href)
        else:
            current_url = None
            
    return books

def scrape_books(target_categories: List[str] = None, min_books: int = 60) -> List[Dict[str, Any]]:
    """
    Scrapes books across specified categories (minimum 3 categories and >= 60 books).
    """
    all_categories = get_category_links()
    if target_categories is None:
        target_categories = ["Travel", "Mystery", "Historical Fiction", "Sequential Art"]
        
    collected_books = []
    for cat in target_categories:
        if cat in all_categories:
            print(f"Scraping category: {cat} ({all_categories[cat]})...")
            cat_books = scrape_category_books(cat, all_categories[cat])
            print(f"  -> Extracted {len(cat_books)} books from {cat}")
            collected_books.extend(cat_books)
            
    if len(collected_books) < min_books:
        for cat, url in all_categories.items():
            if cat not in target_categories:
                print(f"Adding extra category: {cat} to meet >= {min_books} threshold...")
                cat_books = scrape_category_books(cat, url)
                collected_books.extend(cat_books)
                if len(collected_books) >= min_books:
                    break
                    
    print(f"Total books scraped: {len(collected_books)} across {len(set(b['category'] for b in collected_books))} categories.")
    return collected_books

if __name__ == "__main__":
    books = scrape_books()
    print("Sample book record:", books[0] if books else "No books scraped")
