import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

class RetrievalEvaluator:
    """Evaluation framework for retrieval models"""

    def unigram_f1(self, pred_answer, ref_answer):
        """Calculate unigram F1 score"""
        if not pred_answer or not ref_answer:
            return 0.0

        pred_tokens = pred_answer.lower().split()
        ref_tokens = ref_answer.lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        overlap = sum((pred_counter & ref_counter).values())

        if overlap == 0:
            return 0.0

        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def calculate_hit_rate(self, predictions, references, k=1):
        """Calculate hit rate at k"""
        hits = sum(1 for pred_list, ref in zip(predictions, references) 
                  if any(self.unigram_f1(pred, ref) > 0.3 for pred in pred_list[:k]))
        return hits / len(references)

    def evaluate_retriever(self, retriever, test_df):
        """Evaluate a single retriever with essential metrics"""
        all_predictions = []
        all_references = []
        f1_scores = []

        for idx, row in test_df.iterrows():
            results = retriever.retrieve(row['question'], k=5)
            
            if len(results) > 0:
                pred_answers = results['answer'].tolist()
                f1_scores.append(self.unigram_f1(pred_answers[0], row['answer']))
            else:
                pred_answers = []
                f1_scores.append(0.0)
                
            all_predictions.append(pred_answers)
            all_references.append(row['answer'])

        return {
            'mean_f1': np.mean(f1_scores),
            'hit_rate@1': self.calculate_hit_rate(all_predictions, all_references, 1),
            'hit_rate@5': self.calculate_hit_rate(all_predictions, all_references, 5)
        }

    def compare_retrievers(self, retrievers, test_df, names=None):
        """Compare multiple retrievers"""
        if names is None:
            names = [f"Retriever_{i}" for i in range(len(retrievers))]

        results = {}
        for name, retriever in zip(names, retrievers):
            metrics = self.evaluate_retriever(retriever, test_df)
            results[name] = metrics
            print(f"{name}: F1={metrics['mean_f1']:.3f}, Hit@1={metrics['hit_rate@1']:.3f}, Hit@5={metrics['hit_rate@5']:.3f}")

        return results

def run_experiment(df_path, test_split=0.1, random_state=42):
    """Run simplified experiment with data splitting and evaluation"""
    from retrieval import create_retrievers

    df = pd.read_csv(df_path)
    train_df, test_df = train_test_split(df, test_size=test_split, random_state=random_state)

    print(f"Dataset: Train {len(train_df)}, Test {len(test_df)}")

    retrievers = create_retrievers(train_df)

    evaluator = RetrievalEvaluator()
    results = evaluator.compare_retrievers(
        [ret for _, ret in retrievers],
        test_df,
        names=[name for name, _ in retrievers]
    )

    return results
