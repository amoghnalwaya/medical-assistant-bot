#!/usr/bin/env python3
"""
Medical Q&A Chat System - Simple retrieval-based medical question answering
"""

import pandas as pd
import os
from retrieval import FixedHybridRetriever, BaselineTFIDFRetriever, PatternRetriever

class MedicalChatSystem:
    def __init__(self, data_path=None):
        self.retriever = None
        self.data = None
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)

    def load_data(self, data_path):
        self.data = pd.read_csv(data_path)
        print(f"Loaded {len(self.data)} medical Q&A pairs")

        # Initialize hybrid retriever
        self.retriever = FixedHybridRetriever([
            ('TFIDF', BaselineTFIDFRetriever(ngram_range=(1, 2), max_features=20000, stop_words="english")),
            ('Pattern', PatternRetriever())
        ], weights=[0.7, 0.3])

        self.retriever.fit(self.data)

    def get_answer(self, question):
        if not self.retriever:
            return "No data loaded."

        results = self.retriever.retrieve(question, k=1)
        if len(results) == 0:
            return "No relevant information found."

        return results.iloc[0]['answer']

    def start_chat(self):
        if not self.retriever:
            # Try to find data automatically
            for path in ["data/processed_medical_qa.csv", "processed_medical_qa.csv"]:
                if os.path.exists(path):
                    self.load_data(path)
                    break
            else:
                print("No data found. Please provide CSV with 'question' and 'answer' columns.")
                return

        print("Medical Q&A System - Type 'exit' to quit")

        try:
            while True:
                question = input("Question: ").strip()
                if question.lower() in ['exit', 'quit', 'q']:
                    break
                if question:
                    answer = self.get_answer(question)
                    print(f"Answer: {answer}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")

def main():
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    chat_system = MedicalChatSystem(data_path)
    chat_system.start_chat()

if __name__ == "__main__":
    main()