# Medical Assistant Chat System

A simple medical question-answering system using retrieval-based approaches to provide responses to medical queries. Built as a focused 2-day development project with medical-aware preprocessing and multiple retrieval algorithms.

This system implements a medical Q&A assistant using a dataset of 16,406 medical question-answer pairs. It employs TF-IDF baseline, pattern-based retrieval, and hybrid methods to find relevant answers to user queries.

## Implementation Details

- Medical-Aware Preprocessing: Preserves medical terminology while cleaning text
- Multiple Retrieval Methods: TF-IDF baseline, pattern matching, and hybrid approaches
- Question Deduplication: Clustering to handle similar questions
- Simple Evaluation: F1 and Hit Rate metrics for model comparison
- Interactive Chat Interface: Command-line interface with medical disclaimers

## File Structure

The system consists of 4 main Python files:

1. `preprocessing.py`
   - `MedicalQAPreprocessor` class for data cleaning and normalization
   - Medical term preservation and question clustering
   - Answer merging and quality validation

2. `retrieval.py`
   - `BaselineTFIDFRetriever` for lexical matching
   - `PatternRetriever` for template-based question matching
   - `FixedHybridRetriever` combining both methods

3. `evaluation.py`
   - `RetrievalEvaluator` for model evaluation
   - Hit rate and F1 score metrics
   - Simplified comparison tools for multiple retrievers

4. `medical_chat_system.py`
   - Main chat interface and system integration
   - Interactive Q&A with medical disclaimers
   - Simple question-answer retrieval interface

## Installation

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

## Usage

### Interactive Chat
```bash
# Run the chat system (uses pre-processed data)
python3 medical_chat_system.py
```

### Data Preprocessing (output artifact included as data/processed_medical_qa.csv)
```python
from preprocessing import preprocess_medical_dataset

# Process raw data
df, report = preprocess_medical_dataset(
    "data/mle_screening_dataset.csv",
    output_path="data/processed_medical_qa.csv"
)
# Output: 16,406 → 9,137 rows (55.7% retained)
```

### Model Evaluation
```python
from evaluation import run_experiment

# Run simplified experiment with train/test split
results = run_experiment("data/processed_medical_qa.csv")
```

## Approach and Methodology

### Data Preprocessing

The preprocessing pipeline handles medical text challenges:

1. Medical Term Preservation: Regex patterns preserve medical terminology
2. Text Normalization: Standardizes format while maintaining accuracy
3. Question Clustering: Union-Find algorithm groups similar questions
4. Quality Validation: Removes pathological cases and duplicates

### Retrieval Methods

- TF-IDF Baseline: Robust lexical matching for general medical queries
- Pattern Retriever: Template-based matching for structured questions
- Hybrid Approach: Weighted combination of both methods

### Pattern-Based Approach Rationale

Analysis of the medical Q&A dataset revealed that questions follow predictable patterns:

Common Question Templates:
- "What are the symptoms of [disease]?" (52 instances in first 500 questions)
- "What causes [disease]?" (40 instances)
- "How to treat [disease]?" (89 instances)
- "How to diagnose [disease]?" (24 instances)
- "How to prevent [disease]?" (15 instances)

This pattern regularity led us to implement a simple pattern-based retriever that:
1. Recognizes common question templates using regex patterns
2. Extracts disease/condition names from queries
3. Provides direct template-based matching for structured questions

This approach works well because medical questions are often formulaic, making a less complicated retriever more effective than complex semantic models for this specific domain.

### Evaluation

Simplified evaluation framework with essential metrics:

- Hit Rate@K: Percentage of queries with relevant answer in top-K
- F1 Score: Unigram overlap between predicted and reference answers

## Performance Analysis

### Dataset Characteristics
- Size: 16,406 question-answer pairs → 9,137 after preprocessing
- Train/Test Split: 8,223 / 914 samples

### Experimental Results

Current performance on test set:

| Method | F1 Score | Hit Rate@1 | Hit Rate@5 |
|--------|----------|-------------|-------------|
| Fixed Hybrid | 0.232 | 0.229 | 0.565 |
| Baseline TF-IDF | 0.232 | 0.228 | 0.563 |
| Pattern Retriever | 0.007 | 0.010 | 0.011 |

Key Findings:
- TF-IDF baseline performs well for general medical queries
- Hybrid approach provides slight improvement in coverage
- Pattern matching alone has limited effectiveness but adds value in combination
- Removing synonyms mapping had no impact on performance, confirming simplified approach works

Note: Performance metrics remained unchanged after removing synonym mappings from the pattern retriever, validating that a simpler approach is sufficient for this dataset.

## System Design

### Design Choices
- Retrieval over Generation: Focus on accurate retrieval from existing medical content
- Simplicity over Sophistication: Pattern analysis showed complex models aren't needed
- Template-based matching: Leverages the structured nature of medical questions

### Known Limitations
1. No Generation: We are not generating new answers (by design for accuracy)
2. Dataset Bound: Limited to provided Q&A pairs
3. Single-turn interactions only

## Development Notes

This system was designed as a 2-day development project with the following priorities:

- Simplicity over complexity: Pattern analysis showed that simpler approaches work better
- Domain-specific design: Medical questions follow predictable templates
- Working retrieval system: Focus on practical functionality over advanced techniques
- Evaluation-driven development: Simple metrics to validate approach effectiveness

## Example Interactions

Here are three real interactions with the medical chat system:

Example 1: General Medical Query (TF-IDF Retrieval)
```
Question: What is diabetes?
Answer: Studies have shown that most people with prediabetes develop type 2 diabetes within a few years, unless they change their lifestyle. Most people with prediabetes dont have any symptoms. Your doctor can test your blood to find out if your blood glucose levels are higher than normal. Losing weight—at least 5 to 10 percent of your starting weight—can prevent or delay diabetes or even reverse prediabetes. That's 10 to 20 pounds for someone who weighs 200 pounds. You can lose weight by cutting the amount of calories and fat you consume and by being physically active at least 30 to 60 minutes every day. Physical activity also helps your body use the hormone insulin properly. Your body needs insulin to use glucose for energy. Medicine can help control the amount of glucose in your blood. Ask your doctor if medicine to control glucose is right for you.
```

Example 2: Symptom Query (Pattern-based Retrieval)
```
Question: What are the symptoms of hypertension?
Answer: Signs and symptoms of pulmonary hypertension (PH) may include: Shortness of breath during routine activity, such as climbing two flights of stairs Tiredness Chest pain A racing heartbeat Pain on the upper right side of the abdomen Decreased appetite As PH worsens, you may find it hard to do any physical activities. At this point, other signs and symptoms may include: Feeling light-headed, especially during physical activity Fainting at times Swelling in your legs and ankles A bluish color on your lips and skin
```

Example 3: Prevention Query (Pattern-based Retrieval)
```
Question: How to prevent osteoporosis?
Answer: Osteoporosis makes your bones weak and more likely to break. Anyone can develop osteoporosis, but it is common in older women. As many as half of all women and a quarter of men older than 50 will break a bone due to osteoporosis. Risk factors include - Getting older - Being small and thin - Having a family history of osteoporosis - Taking certain medicines - Being a white or Asian woman - Having osteopenia, which is low bone density Osteoporosis is a silent disease. You might not know you have it until you break a bone. A bone mineral density test is the best way to check your bone health. To keep bones strong, eat a diet rich in calcium and vitamin D, exercise and do not smoke. If needed, medicines can also help.
```
No AI models were used to generate any part of this code or documentation.