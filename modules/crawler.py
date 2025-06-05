import json
import os

import asyncio
import logging
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
from crawl4ai.deep_crawling import DFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

# Configure logging
if not os.path.exists("logger"):
    os.makedirs("logger")

logging.basicConfig(
    filename="logger/crawler.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Web_Crawler')


class Crawler:
    def __init__(self, max_depth=2, max_pages=50):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self._create_config()
        browser_conf = BrowserConfig(headless=True, java_script_enabled=True)

        self.crawler = AsyncWebCrawler(config=browser_conf)

    def _create_config(self):

        self.config = CrawlerRunConfig(
            deep_crawl_strategy=DFSDeepCrawlStrategy(
                max_depth=self.max_depth,
                include_external=False,
                max_pages=self.max_pages,
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            verbose=True
        )

    async def crawl(self, url):
        logger.info(f"Starting crawl of {url}")
        try:
            results = await self.crawler.arun(url, config=self.config)
            logger.info(f"Crawl completed with {len(results)} pages")
            return results
        except Exception as e:
            logger.error(f"Error during crawl: {str(e)}")
            return []
        
def run_crawler(url, max_pages=5):
    async def inner():
        crawler = Crawler(max_depth=2, max_pages=max_pages)
        results = await crawler.crawl(url)

        data = {
            "URLS": [r.url for r in results],
            "tables": [r.media.get("tables", []) for r in results],
            "markdown": [r.markdown for r in results],
        }
        
        if not os.path.exists("data"):
            os.makedirs("data")

        with open("data/crawl_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print("Crawl complete. Data saved to crawled_data.json")

    asyncio.run(inner())

# Run the crawler
if __name__ == "__main__":

    url = "https://botpenguin.com"
    max_pages = 5 

    run_crawler(url, max_pages)
