import pandas as pd
import re

class MedicalQAPreprocessor:
    def __init__(self, min_answer_length=10, max_answer_length=1000):
        """
        Medical Q&A Data Preprocessing Pipeline

        Args:
            min_answer_length: Minimum character length for valid answers
            max_answer_length: Maximum character length for answers (for truncation)
        """
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length

    def normalize_text(self, text):
        """Basic text normalization"""
        if not isinstance(text, str) or pd.isna(text):
            return ""

        text = text.strip()
        text = re.sub(r"\s+", " ", text)  # Collapse spaces

        return text

    def clean_data(self, df):
        """Handle basic data quality issues"""
        print(f"Starting with {len(df):,} rows")
        initial_count = len(df)

        # Normalize text
        df['question'] = df['question'].apply(self.normalize_text)
        df['answer'] = df['answer'].apply(self.normalize_text)

        # Remove rows with null/empty content
        df = df[(df['question'].str.len() > 0) & (df['answer'].str.len() > 0)].copy()

        # Remove answers that are too short or too long
        answer_lengths = df['answer'].str.len()
        df = df[(answer_lengths >= self.min_answer_length) & (answer_lengths <= self.max_answer_length)].copy()

        # Remove exact duplicates
        df = df.drop_duplicates(subset=['question', 'answer']).reset_index(drop=True)

        print(f"After cleaning: {len(df):,} rows ({len(df)/initial_count:.1%} retained)")
        return df

    def analyze_dataset(self, df):
        """Provide basic analysis of the processed dataset"""
        print("\n=== DATASET ANALYSIS ===")
        print(f"Total questions: {len(df):,}")
        print(f"Average answer length: {df['answer'].str.len().mean():.0f} chars")

        # Length distribution
        answer_lengths = df['answer'].str.split().str.len()
        print(f"Answer length (words): Min {answer_lengths.min()}, "
              f"Median {answer_lengths.median():.0f}, Max {answer_lengths.max()}")

    def process_medical_qa_dataset(self, df):
        """
        Simple preprocessing pipeline for medical Q&A data

        Args:
            df: DataFrame with 'question' and 'answer' columns

        Returns:
            processed_df: Clean, deduplicated DataFrame
            processing_report: Dictionary with processing statistics
        """
        print("=== MEDICAL Q&A PREPROCESSING ===")

        initial_count = len(df)
        df_clean = self.clean_data(df.copy())
        self.analyze_dataset(df_clean)

        processing_report = {
            'initial_rows': initial_count,
            'final_rows': len(df_clean),
            'reduction_ratio': len(df_clean) / initial_count
        }

        return df_clean, processing_report

def preprocess_medical_dataset(csv_path, output_path=None, **kwargs):
    """
    Preprocess medical Q&A dataset

    Args:
        csv_path: Path to input CSV file
        output_path: Path to save processed CSV (optional)
        **kwargs: Additional parameters for MedicalQAPreprocessor

    Returns:
        processed_df, processing_report
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")

    preprocessor = MedicalQAPreprocessor(**kwargs)
    processed_df, report = preprocessor.process_medical_qa_dataset(df)

    if output_path:
        processed_df.to_csv(output_path, index=False)
        print(f"\nSaved processed dataset to {output_path}")

    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Rows: {report['initial_rows']:,} → {report['final_rows']:,} ({report['reduction_ratio']:.1%})")

    return processed_df, report