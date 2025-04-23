import re
import json
import logging
import requests

from bs4 import BeautifulSoup
from chunking import RegexChunking, SlidingWindowChunking, MultiLevelChunking
from chunking import CosineSimilarityExtractor

# Setup logging
logging.basicConfig(filename="logger\extractor.log", level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class WebScrapeProcessor:
    def __init__(self, input_file, query=""):
        self.input_file = input_file
        self.data = self._load_data()
        self.total_context = ""
        self.query = query

    def _load_data(self):
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                logging.info("Loaded input file successfully.")
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load input file: {e}")
            return {}

    def extract_core_info(self, urls):
        core_data = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")

                title = soup.title.string.strip() if soup.title else "No title"
                meta_desc = soup.find("meta", attrs={"name": "description"})
                meta_desc = meta_desc["content"].strip() if meta_desc and "content" in meta_desc.attrs else "No description"

                headers = {
                    "h1": [h.get_text(strip=True) for h in soup.find_all("h1")],
                    "h2": [h.get_text(strip=True) for h in soup.find_all("h2")],
                    "h3": [h.get_text(strip=True) for h in soup.find_all("h3")],
                }

                for script in soup(["script", "style"]): script.decompose()
                visible_text = soup.get_text(separator=" ", strip=True)
                summary = ' '.join(visible_text.split()[:100])

                emails = list(set(re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", visible_text)))
                phones = list(set(re.findall(r"\+?\d[\d\-\(\) ]{7,}\d", visible_text)))

                core_data.append({
                    "url": url,
                    "title": title,
                    "description": meta_desc,
                    "summary": summary,
                    "headers": headers,
                    "emails": emails,
                    "phones": phones
                })

            except Exception as e:
                logging.warning(f"Error fetching {url}: {e}")
                core_data.append({"url": url, "error": str(e)})

        return json.dumps(core_data)

    def clean_markdown(self, markdown):
        markdown = re.sub(r'!\[.*?\]\(.*?\)', '', markdown)
        markdown = markdown.replace('*', '\n')
        markdown = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1: \2', markdown)
        markdown = re.sub(r'\s+', ' ', markdown)
        markdown = re.sub(r'([a-z])([A-Z])', r'\1 \2', markdown)
        return markdown.strip()

    def flatten_tables(self):
        return json.dumps([k for table in self.data.get('tables', []) for k in table])

    def build_context(self):
        urls_info = self.extract_core_info(self.data.get("URLS", []))

        markdown_info = ""
        for md in self.data.get("markdown", []):
            markdown_info += self.clean_markdown(md)

        table_info = self.flatten_tables()
        self.total_context = table_info + urls_info + markdown_info
        return self.total_context

    def chunk_text(self, text):
        text_length = len(text)

        # Heuristic for dynamic chunking sizes
        window_size = max(512, min(2048, text_length // 10 * 2))
        step = max(128, min(window_size // 2, text_length // 20 * 2))

        logging.info(f"Using window_size={window_size}, step={step} for sliding window chunking.")

        sentence_chunker = RegexChunking(patterns=[r'(?<=[.!?])\s'])
        sliding_window_chunker = SlidingWindowChunking(window_size=window_size, step=step)

        multi_level_chunker = MultiLevelChunking(
            chunkers=[ sliding_window_chunker ], # sentence_chunker],
            min_chunk_size=512
        )

        chunks = multi_level_chunker.chunk(text)
        logging.info(f"Chunked text into {len(chunks)} pieces.")
        return chunks

    def extract_relevant_chunks(self, chunks):
        extractor = CosineSimilarityExtractor(self.query)
        relevant_chunks = extractor.find_relevant_chunks(chunks)
        relevant_chunks.sort(key=lambda x: x[1], reverse=True)
        logging.info(f"Top relevant chunk score: {relevant_chunks[0][1] if relevant_chunks else 'N/A'}")
        return relevant_chunks


    def process(self, output_file="data/processed_chunks.json", top_k=50, query=None):
        logging.info("Processing started...")
        
        if query:
            self.query = query

        
        context = self.build_context()
        chunks = self.chunk_text(context)
        relevant_chunks = self.extract_relevant_chunks(chunks)

        # Simple output format
        # output_data = [
        #     {"chunk": chunk, "score": score}
        #     for chunk, score in relevant_chunks[:top_k]
        # ]

        merged_chunks = []
        buffer = ""
        buffer_score = 0
        count = 0

        for chunk, score in relevant_chunks:
            if count >= top_k:
                break
            if len(buffer) + len(chunk) < 500:
                buffer += " " + chunk
                buffer_score = max(buffer_score, score)
            else:
                if buffer:
                    merged_chunks.append({"chunk": buffer.strip(), "score": buffer_score})
                    count += 1
                buffer = chunk
                buffer_score = score

            while len(buffer) > 1000 and count < top_k:
                split_point = buffer.rfind(" ", 0, 1000)
                if split_point == -1:
                    split_point = 1000
                merged_chunks.append({"chunk": buffer[:split_point].strip(), "score": buffer_score})
                buffer = buffer[split_point:].strip()
                count += 1

        if buffer and count < top_k:
            merged_chunks.append({"chunk": buffer.strip(), "score": buffer_score})


        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(merged_chunks, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved top {top_k} relevant chunks to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save output: {e}")

        logging.info("Processing complete.")
        print("Processing complete. Filtered chunks saved to processed_chunks.json")
        return merged_chunks


if __name__ == "__main__":
    processor = WebScrapeProcessor("data/crawl_data.json")
    results = processor.process(top_k=25)
    print(f"Top relevant chunks {len(results)} saved.")

