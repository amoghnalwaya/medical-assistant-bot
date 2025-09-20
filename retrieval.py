import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class BaselineTFIDFRetriever:
    """
    Simple TF-IDF + cosine retriever with minimal deps
    """

    def __init__(self, ngram_range=(1, 2), max_features=20000, stop_words="english"):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.stop_words = stop_words
        self.vectorizer = None
        self.X = None
        self.train_data = None

    def fit(self, train_df):
        self.train_data = train_df.copy()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words=self.stop_words,
            norm="l2",
            sublinear_tf=True,
        )
        self.X = self.vectorizer.fit_transform(self.train_data["question"])
        print(f"TF-IDF: Indexed {len(self.train_data)} documents, {self.X.shape[1]} features")

    def retrieve(self, query, k=5):
        if self.vectorizer is None:
            return pd.DataFrame(columns=["question", "answer", "similarity_score"])
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.X)[0]
        top = np.argsort(sims)[::-1][:k]
        out = self.train_data.iloc[top].copy()
        out["similarity_score"] = sims[top]
        return out.reset_index(drop=True)


class FixedHybridRetriever:
    """Fixed hybrid retriever with proper score normalization"""

    def __init__(self, retrievers, weights=None, normalize_scores=True):
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)
        self.normalize_scores = normalize_scores
        self.train_data = None

    def fit(self, train_df):
        """Fit all retrievers"""
        self.train_data = train_df.copy()

        for name, retriever in self.retrievers:
            retriever.fit(train_df)

    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1] range"""
        if len(scores) == 0:
            return scores

        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones_like(scores) * 0.5

        return (scores - min_score) / (max_score - min_score)

    def retrieve(self, query, k=5, pool_size=50):
        """Retrieve using fixed weighted combination"""
        # Collect all candidates with normalized scores
        all_candidates = {}

        for (name, retriever), weight in zip(self.retrievers, self.weights):
            results = retriever.retrieve(query, k=pool_size)

            if len(results) == 0:
                continue

            # Get scores and normalize them
            scores = results['similarity_score'].values
            if self.normalize_scores:
                scores = self._normalize_scores(scores)

            # Add to candidates
            for idx, (_, row) in enumerate(results.iterrows()):
                # Use actual dataframe index as key
                doc_key = row.name if hasattr(row, 'name') else idx
                question = row['question']
                answer = row['answer']

                # Create unique key based on content
                content_key = f"{question}||{answer}"

                if content_key not in all_candidates:
                    all_candidates[content_key] = {
                        'data': row,
                        'total_score': 0.0,
                        'component_scores': {}
                    }

                all_candidates[content_key]['total_score'] += weight * scores[idx]
                all_candidates[content_key]['component_scores'][name] = scores[idx]

        if not all_candidates:
            return pd.DataFrame()

        # Sort by combined score and get top-k
        sorted_candidates = sorted(
            all_candidates.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )

        # Build results DataFrame
        result_rows = []
        for content_key, candidate_info in sorted_candidates[:k]:
            row_data = candidate_info['data'].copy()
            row_data['similarity_score'] = candidate_info['total_score']

            # Add component scores
            for name, _ in self.retrievers:
                row_data[f'{name}_score'] = candidate_info['component_scores'].get(name, 0.0)

            result_rows.append(row_data)

        return pd.DataFrame(result_rows).reset_index(drop=True)

class PatternRetriever:
    """Simple pattern-based retriever for template questions"""

    def __init__(self):
        self.patterns = {
            r"how many people are affected by (.+)\?": "affected_by",
            r"how to diagnose (.+)\?": "diagnose",
            r"how to prevent (.+)\?": "prevent",
            r"what are the complications of (.+)\?": "complications",
            r"what are the stages of (.+)\?": "stages",
            r"what are the symptoms of (.+)\?": "symptoms",
            r"what causes (.+)\?": "causes"
        }
        self.template_index = {}
        self.train_data = None

    @staticmethod
    def _norm(s):
        return re.sub(r"\s+", " ", s.strip().lower())

    def fit(self, train_df):
        self.train_data = train_df.copy()
        for pattern_type in self.patterns.values():
            self.template_index[pattern_type] = {}

        for idx, row in train_df.iterrows():
            q = self._norm(row["question"])
            a = row["answer"]
            for pat, ptype in self.patterns.items():
                m = re.search(pat, q)
                if m:
                    disease = self._norm(m.group(1))
                    self.template_index[ptype].setdefault(disease, {
                        "answer": a,
                        "question": row["question"],
                        "idx": idx
                    })
                    break

        total = sum(len(d) for d in self.template_index.values())
        print(f"Pattern Retriever: Indexed {total} template entries")

    def retrieve(self, query, k=5):
        q = self._norm(query)
        results = []
        for pat, ptype in self.patterns.items():
            m = re.search(pat, q)
            if not m:
                continue
            disease = self._norm(m.group(1))
            exact = self.template_index.get(ptype, {}).get(disease)
            if exact:
                results.append({
                    "question": exact["question"],
                    "answer": exact["answer"],
                    "similarity_score": 1.0
                })
            else:
                # soft match: same pattern_type, substring overlap
                for stored_disease, entry in self.template_index.get(ptype, {}).items():
                    if disease in stored_disease or stored_disease in disease:
                        results.append({
                            "question": entry["question"],
                            "answer": entry["answer"],
                            "similarity_score": 0.8
                        })
            break  # matched a pattern; stop checking others

        results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:k]
        return pd.DataFrame(results) if results else pd.DataFrame(columns=["question", "answer", "similarity_score"])

def create_retrievers(train_df):
    """Create and fit baseline TF-IDF + Pattern + Hybrid (only)."""
    tfidf = BaselineTFIDFRetriever(ngram_range=(1, 2), max_features=20000, stop_words="english")
    pattern = PatternRetriever()

    hybrid = FixedHybridRetriever([
        ("TFIDF", BaselineTFIDFRetriever(ngram_range=(1, 2), max_features=20000, stop_words="english")),
        ("Pattern", PatternRetriever())
    ], weights=[0.7, 0.3])  # slightly prefer lexical baseline; easy to tweak

    retrievers = [
        ("Baseline_TFIDF", tfidf),
        ("Pattern_Retriever", pattern),
        ("Fixed_Hybrid", hybrid)
    ]

    print("--- Fitting Retrievers ---")
    for name, r in retrievers:
        print(f"Fitting {name}...")
        r.fit(train_df)

    return retrievers