import argparse

from modules.crawler import run_crawler
from modules.processor import run_processor
from modules.memory import run_memory

def main():
    parser = argparse.ArgumentParser(description="Web Scraping + Q&A CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Crawl subcommand
    crawl_parser = subparsers.add_parser("crawl", help="Run the crawler ")
    crawl_parser.add_argument("url", help="Start URL to crawl (Enter url: for user preferred website) ")
    crawl_parser.add_argument("max_pages", type=int, default=5, help="Maximum Pages to crawl (default: 10)")

    # Process subcommand
    process_parser = subparsers.add_parser("process", help="Process data into chunks (query: for user preferred processing)")
    process_parser.add_argument("query", nargs='?', default='', help="User guided chunk filtering")
    process_parser.add_argument("numbers", type=int, default=20, help="Enter the numbers of data chunks (default: 20)")

    # Summarize subcommand
    memory_parser = subparsers.add_parser("memory", help="memory and classify relevant chunks")

    # add the QA subcommand
    question_parser = subparsers.add_parser("ask", help="ask any question related to website")
    question_parser.add_argument("query", nargs='?', default='', help="User query to Answer by BOT")

    args = parser.parse_args()

    if args.command == "crawl":
        run_crawler(args.url, args.max_pages)
    elif args.command == "process":
        run_processor(args.query, args.numbers)
    elif args.command == "memory":
        run_memory()
    elif args.command == "ask":
        print(args.query, "\ncurrently untested! ")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
