import json
import torch
import logging
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoModel
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline


class Chatbot:
    def __init__(self,
                 memory_file="web_scraping/data/summaries.json",
                 dialogue_model_path="facebook/blenderbot-400M-distill",
                 topic_model_path="cardiffnlp/tweet-topic-21-multi",
                 hf_token="your_token_here",
                 use_local=True):

        logging.basicConfig(filename="web_scraping/logger/chatbot.log", level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        logging.info("Initializing chatbot...")

        # Load memory
        with open(memory_file, "r", encoding="utf-8") as f:
            self.memory = json.load(f)

        self.memory_texts = [item["summary"] for item in self.memory]

        # Use a basic tokenizer + embedding from transformer if sentence_transformers is not available
        self.embedding_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.embedding_model = AutoModel.from_pretrained("bert-base-uncased")

        # Pre-compute mean pooled embeddings
        self.memory_embeddings = self._encode_memory(self.memory_texts)

        # Load conversational model
        self.dialogue_tokenizer = BlenderbotTokenizer.from_pretrained(dialogue_model_path)
        self.dialogue_model = BlenderbotForConditionalGeneration.from_pretrained(dialogue_model_path)

        # Load topic model
        self.topic_tokenizer = AutoTokenizer.from_pretrained(topic_model_path)
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(topic_model_path)

        # Load Inference Client
        self.use_local = use_local
        self.model_name = "facebook/blenderbot-400M-distill"

        if not self.use_local:
            try:
                self.inference_client = InferenceClient(token=hf_token)
                _ = self.inference_client.text_generation(prompt="test", model=self.model_name, max_new_tokens=5)
                logging.info("Using InferenceClient for chatbot generation.")
            except Exception as e:
                logging.warning(f"Inference API failed, falling back to local pipeline. Reason: {e}")
                self.use_local = True

    def _encode_memory(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                mean_pooled = outputs.last_hidden_state.mean(dim=1).squeeze()
                embeddings.append(mean_pooled)
        return torch.stack(embeddings)

    def retrieve_memory(self, query, top_k=25):
        inputs = self.embedding_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

        scores = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), self.memory_embeddings)
        top_indices = torch.topk(scores, top_k).indices.tolist()
        return [self.memory[i] for i in top_indices]

    def classify_topic(self, text):
        inputs = self.topic_tokenizer(text, return_tensors="pt")
        outputs = self.topic_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        return predicted_class

    def chat(self, user_input):

        logging.info(f"User input: {user_input}")
        topic = self.classify_topic(user_input)

        

        relevant_memories = self.retrieve_memory(user_input)

        context = "\n".join([m["summary"] for m in relevant_memories])
        dialogue_input = f"{context}\nUser: {user_input}"

        if self.use_local:
            inputs = self.dialogue_tokenizer(dialogue_input, return_tensors="pt")
            outputs = self.dialogue_model.generate(**inputs, max_length=256)
            reply = self.dialogue_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            try:
                reply = self.inference_client.text_generation(
                    prompt=dialogue_input,
                    model=self.model_name,
                    max_new_tokens=256,
                    do_sample=False
                ).strip()
            except Exception as e:
                logging.error(f"Inference API failed during chat: {e}")
                reply = "I'm having trouble responding at the moment."

        return reply, relevant_memories


if __name__ == "__main__":
    bot = Chatbot(memory_file="web_scraping/data/summaries.json")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply, memories = bot.chat(user_input)
        print("Bot:", reply)
        print("Relevant Memories:", [m["summary"][:100] + "..." for m in memories])
