import asyncio
import json
import re

import aiohttp
from bs4 import BeautifulSoup

NBSP = u'\xa0'

def normalize(s: str) -> str:
    _TRANSLATION_TABLE = {
        NBSP: " ",
        ord('\u200e'): None
    }
    s = (s or "").translate(_TRANSLATION_TABLE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

async def extract_tilda_content_html(html: str) -> list[str]:
    """
    Возвращает список текстовых блоков, начиная с первого <h1> (включая его)
    и до подвала <footer id="t-footer">, отфильтровав только элементы
    с data-elem-type="text" + сам заголовок.
    """
    soup = BeautifulSoup(html, "html.parser")

    allrecords = soup.select_one("#allrecords") or soup
    footer = allrecords.select_one('footer#t-footer')

    h1 = None
    for h in allrecords.select("h1"):
        if not h.find_parent("header"):
            h1 = h
            break
    if not h1:
        return []

    result = []

    h1_text = normalize(h1.get_text(" ", strip=True))
    if h1_text:
        result.append(h1_text)

    after_h1 = False
    for element in allrecords.descendants:
        if element is h1:
            after_h1 = True
            continue
        if footer and element is footer:
            break
        if not after_h1:
            continue
        if not getattr(element, "attrs", None):
            continue

        if element.attrs.get("data-elem-type") == "text":
            atom = element.select_one(".tn-atom")
            text = normalize(atom.get_text(" ", strip=True) if atom else element.get_text(" ", strip=True))
            if not text or "с нашей Политикой" in text or "Напишите нам" in text:
                continue
            result.append(text)

    return result


async def extract_content_url(url: str) -> list[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36"
    }

    async with aiohttp.request(
        "GET",
        url=url,
        headers=headers
    ) as response:
        response.raise_for_status()
        html = await response.text()

    return await extract_tilda_content_html(html)


async def main():
    with open('data/links.json', 'r', encoding='utf-8') as f:
        links = json.load(f)
    
    tasks = []
    for link in links:
        tasks.append(extract_content_url(link))

    texts = await asyncio.gather(*tasks)

    data = []
    for link, text in zip(links, texts):
        data.append({
            "url": link,
            "text": "\n\n".join(text)
        })

    with open('data/content.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
