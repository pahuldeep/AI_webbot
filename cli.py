import argparse
from modules.crawler import run_crawler
from modules.processor import run_processor
from modules.memory import run_summarizer
# from modules.summarizer_module import answer_query

def main():
    parser = argparse.ArgumentParser(description="Modular Web Scraping + Q&A CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Crawl subcommand
    crawl_parser = subparsers.add_parser("crawl", help="Run the crawler")
    crawl_parser.add_argument("url", help="Start URL to crawl")

    # Process subcommand
    process_parser = subparsers.add_parser("process", help="Process crawled data into chunks")
    process_parser.add_argument("query", help="User query to guide chunk filtering")

    # Summarize subcommand
    summarize_parser = subparsers.add_parser("update_memory", help="Summarize and classify relevant chunks")

    # Ask subcommand
    ask_parser = subparsers.add_parser("ask", help="Answer a user query")
    ask_parser.add_argument("query", help="Question to answer from processed summaries")

    args = parser.parse_args()

    if args.command == "crawl":
        run_crawler(args.url)
    elif args.command == "process":
        run_processor(args.query)
    elif args.command == "update_memory":
        run_summarizer()
    # elif args.command == "ask":
    #     answer_query(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
