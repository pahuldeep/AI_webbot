import json
import logging

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from huggingface_hub import InferenceClient
from scipy.special import expit

# Setup logging
logging.basicConfig(filename="logger/summary.log", level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class SummaryGenerator:
    def __init__(self, model_name="facebook/bart-large-cnn", hf_token="your_token_here", use_local=False):
        logging.info(f"Initializing summary generator with model: {model_name}")

        self.model_name = model_name
        self.use_local = use_local
        self.hf_token = hf_token

        if not self.use_local:
            try:
                self.client = InferenceClient(token=hf_token)
                _ = self.client.text_generation(prompt="text", model=self.model_name, max_new_tokens=5)
                logging.info("Using InferenceClient for summarization.")
            except Exception as e:
                logging.warning(f"Inference API failed, falling back to local pipeline. Reason: {e}")
                self.use_local = True

        if self.use_local:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)

        # Load topic classifier
        self.topic_model_name = "cardiffnlp/tweet-topic-21-multi"
        self.use_local_topic = True

        try:
            self.topic_client = InferenceClient(token=hf_token)
            _ = self.topic_client.text_classification(model=self.topic_model_name, inputs="test")
            logging.info("Using InferenceClient for topic classification.")
            self.use_local_topic = False
        except Exception as e:
            logging.warning(f"Topic classifier API failed, falling back to local. Reason: {e}")

        if self.use_local_topic:
            self.topic_tokenizer = AutoTokenizer.from_pretrained(self.topic_model_name)
            self.topic_model = AutoModelForSequenceClassification.from_pretrained(self.topic_model_name)
            self.class_mapping = self.topic_model.config.id2label

    def generate_summary(self, text, max_length=256, min_length=30):
        try:
            logging.info("Generating summary...")
            if self.use_local:
                result = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
                return result[0]["summary_text"]
            else:
                response = self.client.text_generation(
                    prompt=f"summarize: {text}",
                    model=self.model_name,
                    max_new_tokens=max_length,
                    do_sample=False
                )
                return response.strip()
        except Exception as e:
            logging.error(f"Failed to generate summary: {e}")
            return ""

    def classify_topic(self, summary):
        try:
            if self.use_local_topic:
                tokens = self.topic_tokenizer(summary, return_tensors='pt')
                output = self.topic_model(**tokens)
                scores = output[0][0].detach().numpy()
                scores = expit(scores)
                predictions = (scores >= 0.5) * 1
                labels = [self.class_mapping[i] for i, pred in enumerate(predictions) if pred == 1]
                return labels
            else:
                response = self.topic_client.text_classification(
                    model=self.topic_model_name,
                    inputs=summary
                )
                return [r["label"] for r in response if r["score"] >= 0.5]
        except Exception as e:
            logging.error(f"Failed to classify topic: {e}")
            return []

    def summarize_chunks(self, input_file="data/processed_chunks.json", output_file="data/memory.json"):
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            summaries = []
            for item in chunks:
                raw_chunk = item["chunk"]
                summary = self.generate_summary(raw_chunk)
                labels = self.classify_topic(summary)

                summaries.append({
                    "original": raw_chunk,
                    "summary": summary,
                    "labels": labels,
                    "score": item["score"],
                })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)

            logging.info(f"Saved summaries to {output_file}")
        except Exception as e:
            logging.error(f"Failed to summarize chunks: {e}")

def run_summarizer(use_local=True, hf_token="your_token_here"):

    if hf_token.startswith("hf_"):
        print("Using InferenceClient for summarization.")
        summarizer = SummaryGenerator(hf_token=hf_token)

    else:
        print("Using local model for summarization.")
        summarizer = SummaryGenerator(use_local=use_local)

    summarizer.summarize_chunks()
    print("Update Memmory and topics saved to memory.json")

if __name__ == "__main__":
    run_summarizer()
