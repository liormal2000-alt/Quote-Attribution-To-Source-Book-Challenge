### CONFIGURATION ###

import os
import json
import random
import time
from tqdm import tqdm
from collections import defaultdict, Counter
from evaluation import eval


try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy import stats
    import seaborn as sns

    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

# Set random seed for reproducibility
SEED = 40
random.seed(SEED)

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# models to evaluate
MODELS_TO_EVALUATE = [
    "openrouter/google/gemini-2.5-pro", #
    "openrouter/google/gemini-2.5-flash", #
    "openrouter/google/gemini-2.5-flash-lite", #
    "openrouter/google/gemini-2.0-flash-001", #
    "openrouter/openai/gpt-5-mini", #
    "openrouter/anthropic/claude-sonnet-4", #
    "openrouter/x-ai/grok-4-fast", #
    "openrouter/deepseek/deepseek-r1"
]

# file paths and settings
DATASET_PATH = "data/examples.jsonl"
RESULTS_DIR = "results"
ANALYSIS_DIR = "analysis"
CSV_REPORTS_DIR = os.path.join(ANALYSIS_DIR, "csv_reports")
N_PER_DIFFICULTY = 150


METADATA_PALETTES = {
    'length_category': 'Blues',
    'year_category': 'Greens',
    'rarity_category': 'Oranges',
    'entity_category': 'Purples',
    'popularity_category': 'Reds',
    'score_category': 'GnBu'
}

### MODEL API ###

def query_model(prompt_text, model_name):
    """
    Query ANY model available on OpenRouter using the litellm library.
    """
    if not LITELLM_AVAILABLE:
        return "error: litellm not available", 0, False

    start_time = time.perf_counter()
    try:
        messages = [{"role": "user", "content": prompt_text}]
        response = litellm.completion(
            model=model_name,
            messages=messages,
            api_key=OPENROUTER_API_KEY
        )
        duration = time.perf_counter() - start_time
        prediction = response.choices[0].message.content
        if not prediction:
            return "error: response was empty", duration, True
        return prediction.strip(), duration, False
    except Exception as e:
        duration = time.perf_counter() - start_time
        print(f"litellm API Error for model '{model_name}': {e}")
        return f"error: {str(e)}", duration, True

### DATA METHODS ###

def load_and_sample_dataset(file_path, n_per_difficulty):
    """Loads and samples a balanced test set from a JSON Lines (.jsonl) dataset."""
    print(f"\nLoading dataset from '{file_path}'...")

    if not os.path.exists(file_path):
        print(f"❌ Dataset file not found at '{file_path}'")
        return {}

    difficulties = {"easy": [], "medium": [], "hard": []}

    try:
        # Read the file line-by-line for .jsonl format
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                diff = item.get("metadata", {}).get("difficulty", "medium")
                if diff in difficulties:
                    difficulties[diff].append(item)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return {}

    print("✅ Dataset loaded. Sampling test set...")
    sampled_data = {}
    total_samples = 0

    for diff, items in difficulties.items():
        if not items:
            print(f"   No examples found for difficulty: {diff}")
            continue

        sample_size = min(len(items), n_per_difficulty)
        sampled_data[diff] = random.sample(items, sample_size)
        total_samples += sample_size
        print(f"   - Sampled {sample_size} '{diff}' examples")

    print(f"Total questions in test set: {total_samples}")
    return sampled_data

def get_dataset_statistics(sampled_data):
    """Calculate statistics about the dataset."""
    stats = {
        'total_questions': 0,
        'difficulty_distribution': {},
        'avg_quote_length': 0,
        'avg_difficulty_score': 0,
        'quote_length_range': (0, 0),
        'difficulty_score_range': (0, 0),
    }

    all_examples = []
    for diff, examples in sampled_data.items():
        stats['difficulty_distribution'][diff] = len(examples)
        stats['total_questions'] += len(examples)
        all_examples.extend(examples)

    if all_examples:
        quote_lengths = [len(ex['quote']) for ex in all_examples]
        difficulty_scores = [ex['metadata'].get('difficulty_score', 0) for ex in all_examples]

        stats['avg_quote_length'] = sum(quote_lengths) / len(quote_lengths)
        stats['avg_difficulty_score'] = sum(difficulty_scores) / len(difficulty_scores)
        stats['quote_length_range'] = (min(quote_lengths), max(quote_lengths))
        stats['difficulty_score_range'] = (min(difficulty_scores), max(difficulty_scores))

    return stats

### CATEGORIZATION ###
def categorize_publication_year(year):
    """
    Categorize publication year into four eras.

    EXPLANATION:
       Pre-1950: Classic literature from before 1950
       1950-1980: Mid-20th century works
       1980-2000: Late 20th century literature
       Post-2000: Contemporary 21st century works

       Newer books might have better online coverage and digital presence. But older books
       might also get a lot of data presense online because they are old, and probably had the chance
       to be mentioned more times, and probably are recited more over the years than new books.
    """
    if not year or not isinstance(year, (int, float)):
        return "Unknown"
    if year < 1950:
        return "Pre-1950"
    elif year < 1980:
        return "1950-1980"
    elif year < 2000:
        return "1980-2000"
    else:
        return "Post-2000"


def categorize_rarity_ratio(ratio):
    """
    Categorize lexical rarity into descriptive bins.

    XPLANATION:
       Low Rarity (Generic): Common words and phrases (ratio < 0.15)
       Medium Rarity: Moderately unique phrasing (ratio 0.15-0.30)
       High Rarity (Unique): Very distinctive language (ratio > 0.30)

    💡 Quotes with unique rare words are easier to find through search,
       as they stand out more distinctly in text databases.
    """
    if ratio is None or not isinstance(ratio, (int, float)):
        return "Unknown"
    if ratio < 0.15:
        return "Low Rarity (Generic)"
    elif ratio < 0.30:
        return "Medium Rarity"
    else:
        return "High Rarity (Unique)"


def categorize_quote_length(length):
    """
    Categorize quote length into bins.

    EXPLANATION:
       very_short: < 50 characters (brief snippets)
       short: 50-100 characters (sentence fragments)
       medium: 100-200 characters (full sentences)
       long: 200-300 characters (multiple sentences)
       very_long: > 300 characters (paragraphs)

       Longer quotes provide more context and unique phrases,
       making them potentially easier to match to their source.
    """
    if length < 50:
        return "very_short"
    elif length < 100:
        return "short"
    elif length < 200:
        return "medium"
    elif length < 300:
        return "long"
    else:
        return "very_long"

def categorize_difficulty_score(score):
    """
    Categorize raw difficulty score into four ranges.
    """
    if score < -3.0:
        return "Easy (Score < -3)"
    elif score <= 0.0:
        return "Medium (Score -3 to 0)"
    elif score <= 3.0:
        return "Hard (Score 0 to 3)"
    else: # score > 3.0
        return "Very Hard (Score > 3)"

def categorize_book_popularity(ratings_count, all_ratings):
    """
    Categorize book popularity based on thirds of the distribution.

    EXPLANATION:
       Bottom Third (Unpopular): Least known books with fewer ratings
       Middle Third (Moderate): Moderately popular books
       Top Third (Popular): Most well-known books with many ratings

       More popular books are more likely to be discussed online,
       quoted in reviews, and available in training data, making them easier to identify.
    """
    if not all_ratings or ratings_count is None:
        return "Unknown"

    sorted_ratings = sorted(all_ratings)
    n = len(sorted_ratings)

    third_1 = sorted_ratings[n // 3]
    third_2 = sorted_ratings[2 * n // 3]

    if ratings_count <= third_1:
        return "Bottom Third (Unpopular)"
    elif ratings_count <= third_2:
        return "Middle Third (Moderate)"
    else:
        return "Top Third (Popular)"


def categorize_named_entities(entities_count):
    """
    Categorize named entities into three informative bins.

       EXPLANATION:
           No Entities: Generic text without proper nouns (e.g., no names, places)
           Few Entities (1-2): Contains some names/places
           Many Entities (3+): Rich in proper nouns

       Named entities (character names, locations) are highly distinctive
       and make quotes much easier to search for and match to their source.
    """
    if entities_count == 0:
        return "No Entities"
    elif entities_count <= 2:
        return "Few Entities (1-2)"
    else:
        return "Many Entities (3+)"

### STATISTICS ###

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
import statsmodels.api as sm

def calculate_statistics(results):
    """Calculate comprehensive statistics from results."""
    if not results:
        return {}

    df = pd.DataFrame(results) if VIZ_AVAILABLE else None

    stats = {
        'overall_accuracy': sum(r['is_correct'] for r in results) / len(results) * 100,
        'total_questions': len(results),
        'total_correct': sum(r['is_correct'] for r in results),
        'total_errors': sum(r['had_error'] for r in results),
        'avg_response_time': sum(r['response_time_sec'] for r in results) / len(results),
        'median_response_time': sorted(r['response_time_sec'] for r in results)[len(results) // 2],
    }

    if df is not None:
        try:
            # Check if there is any variation in the correctness data
            if df['is_correct'].nunique() > 1:
                corr_difficulty = df['difficulty_score'].corr(df['is_correct'].astype(int))
                stats['correlation_difficulty_accuracy'] = corr_difficulty

                corr_length = df['quote_length'].corr(df['is_correct'].astype(int))
                stats['correlation_length_accuracy'] = corr_length

                corr_popularity = df['book_popularity'].corr(df['is_correct'].astype(int))
                stats['correlation_popularity_accuracy'] = corr_popularity
            else:
                print("Skipping correlation calculation: All answers were either correct or incorrect (no variance).")
                stats['correlation_difficulty_accuracy'] = float('nan')
                stats['correlation_length_accuracy'] = float('nan')
                stats['correlation_popularity_accuracy'] = float('nan')

        except Exception as e:
            print(f"   Could not calculate correlations: {e}")

    return stats


def calculate_significance_matrix(all_model_results, baseline_model_name):
    """
    Performs McNemar's test between a baseline model and all other models.
    Returns a DataFrame with p-values. A low p-value (e.g., < 0.05) suggests a
    statistically significant difference in the error rates.
    """
    if not VIZ_AVAILABLE or baseline_model_name not in all_model_results:
        return None

    print(f"\nCalculating statistical significance against baseline: {baseline_model_name}")

    baseline_results = {r['id']: r['is_correct'] for r in all_model_results[baseline_model_name]}
    model_names = sorted(all_model_results.keys())

    significance_data = []

    for model_name in model_names:
        if model_name == baseline_model_name:
            continue

        model_results = {r['id']: r['is_correct'] for r in all_model_results[model_name]}

        b_correct_m_wrong = 0
        b_wrong_m_correct = 0

        for q_id, baseline_correct in baseline_results.items():
            model_correct = model_results.get(q_id)
            if model_correct is None: continue

            if baseline_correct and not model_correct:
                b_correct_m_wrong += 1
            elif not baseline_correct and model_correct:
                b_wrong_m_correct += 1

        table = [[0, b_correct_m_wrong], [b_wrong_m_correct, 0]]

        try:
            # we want the p-value
            result = mcnemar(table, exact=True)
            p_value = result.pvalue
        except ValueError:
            p_value = 1.0

        significance_data.append({
            "Comparison Model": model_name,
            "p-value": p_value,
            "Is Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
        })

    return pd.DataFrame(significance_data)


def analyze_trend_significance(all_model_results, category_col, category_order):
    """
    Performs a Chi-Squared Test for Trend (Cochran-Armitage) to see if there's a
    significant trend in accuracy across ordered categories of a metadata feature.

    Args:
        all_model_results (dict): The main dictionary of all model results.
        category_col (str): The name of the metadata column to analyze (e.g., 'difficulty').
        category_order (list): The specific order of the categories (e.g., ['easy', 'medium', 'hard']).

    Returns:
        A pandas DataFrame with the p-value for the trend for each model.
    """
    if not VIZ_AVAILABLE:
        return None

    print(f"\nAnalyzing trend significance for feature: '{category_col}'")
    all_results_list = [item for results in all_model_results.values() for item in results]
    if not all_results_list:
        return None

    df = pd.DataFrame(all_results_list)

    trend_data = []

    for model_name, group in df.groupby('model_name'):
        # Create a contingency table for the model:

        contingency_table = []
        for category in category_order:
            subset = group[group[category_col.replace('_category', '')] == category]
            if subset.empty:
                contingency_table.append([0, 0])
                continue

            correct_count = subset['is_correct'].sum()
            incorrect_count = len(subset) - correct_count
            contingency_table.append([correct_count, incorrect_count])

        table = np.array(contingency_table).T

        if table.shape[1] < 2 or np.sum(table) == 0:
            p_value = float('nan')
        else:
            try:
                chi2_test = sm.stats.Table(table)
                trend_test_result = chi2_test.test_ordinal_association()
                p_value = trend_test_result.pvalue
            except Exception:
                p_value = float('nan')

        trend_data.append({
            "Model": model_name,
            "Feature": category_col,
            "p-value": p_value,
            "Significant Trend (p < 0.05)": "Yes" if p_value < 0.05 else "No"
        })

    return pd.DataFrame(trend_data)


def analyze_feature_correlations(all_model_results, output_dir_csv, output_dir_viz):
    """
    Calculates the correlation between metadata features and accuracy,
    and saves the results as a CSV and a heatmap visualization.
    """
    if not VIZ_AVAILABLE:
        print("Cannot run correlation analysis because visualization libraries are not installed.")
        return

    print("\nAnalyzing feature correlations with accuracy...")

    try:
        # Combine all results into a single DataFrame
        all_results_list = [item for results in all_model_results.values() for item in results]
        if not all_results_list:
            print("   No results to analyze.")
            return

        df = pd.DataFrame(all_results_list)

        # We only need the relevant numeric/ordinal columns for correlation
        correlation_df = df[[
            'is_correct',
            'difficulty_score',
            'quote_length',
            'book_popularity',
            'entities_count',
            'publication_year',
            'rarity_ratio'
        ]].copy()

        # Convert boolean 'is_correct' to integer (1 for True, 0 for False)
        correlation_df['is_correct'] = correlation_df['is_correct'].astype(int)

        # Calculate the correlation matrix
        corr_matrix = correlation_df.corr()

        # We only care about the correlation with 'is_correct'
        accuracy_correlations = corr_matrix[['is_correct']].drop('is_correct')
        accuracy_correlations.rename(columns={'is_correct': 'Correlation with Accuracy'}, inplace=True)

        # --- Save the data as a CSV file ---
        csv_path = os.path.join(output_dir_csv, '5_feature_accuracy_correlations.csv')
        accuracy_correlations.to_csv(csv_path)
        print(f"   ✅ Correlation data saved to: {os.path.basename(csv_path)}")

        # --- Generate and save the heatmap visualization ---
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            accuracy_correlations,
            annot=True,  # Show the correlation values on the map
            cmap='coolwarm',  # Use a diverging colormap (blue for neg, red for pos)
            vmin=-1, vmax=1,  # Set the color scale from -1 to 1
            linewidths=.5,
            fmt=".2f"  # Format numbers to two decimal places
        )
        plt.title('Correlation of Metadata Features with Model Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        viz_path = os.path.join(output_dir_viz, 'feature_correlation_heatmap.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Correlation heatmap saved to: {os.path.basename(viz_path)}")

    except Exception as e:
        print(f"   ❌ Error during correlation analysis: {e}")


def calculate_category_correlation(all_model_results):
    """
    Calculates the Spearman rank correlation between the ordered difficulty
    categories and the average accuracy across all models.
    """
    if not VIZ_AVAILABLE:
        return

    print("\nAnalyzing correlation of difficulty categories vs. accuracy...")

    try:
        # Combine all results into a single DataFrame
        all_results_list = [item for results in all_model_results.values() for item in results]
        if not all_results_list:
            print("   No results to analyze.")
            return

        df = pd.DataFrame(all_results_list)

        # 1. Calculate the average accuracy for each difficulty category
        category_accuracy = df.groupby('score_category')['is_correct'].mean().reset_index()

        # 2. Define the correct order of the categories and apply it
        category_order = ['Easy (Score < -3)', 'Medium (Score -3 to 0)', 'Hard (Score 0 to 3)', 'Very Hard (Score > 3)']
        category_accuracy['score_category'] = pd.Categorical(
            category_accuracy['score_category'],
            categories=category_order,
            ordered=True
        )
        category_accuracy = category_accuracy.sort_values('score_category')

        # 3. Get the numerical ranks (Easy=0, Medium=1, etc.) and the accuracy values
        difficulty_ranks = category_accuracy['score_category'].cat.codes
        accuracy_values = category_accuracy['is_correct']

        # 4. Calculate Spearman's rank correlation
        rho, p_value = stats.spearmanr(difficulty_ranks, accuracy_values)

        # 5. Print the results
        print("\n" + "=" * 60)
        print("  Correlation: Difficulty Category vs. Average Model Accuracy")
        print("=" * 60)
        print(f"  Spearman's Rho (ρ): {rho:.4f}")
        print(f"  P-value: {p_value:.4f}")

        if p_value < 0.05 and rho < -0.9:
            print("  ✅ Interpretation: A strong, statistically significant negative correlation.")
            print(
                "     This confirms your difficulty categories are well-ordered and strongly predict a drop in performance.")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"   ❌ Error during category correlation analysis: {e}")


### EVALUATION ###

def generate_prompt(quote_text):
    """
    Generate the prompt for a given quote.
    """
    return f"From which book is the following quote? {quote_text}"

def evaluate_model(model_name, sampled_data):
    """
    Run comprehensive evaluation for a given model with expanded metadata analysis.
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'=' * 80}")

    results = []

    # all ratings for popularity categorization
    all_ratings = []
    for examples in sampled_data.values():
        for ex in examples:
            rating = ex.get("metadata", {}).get("ratings_count", 0)
            if rating:
                all_ratings.append(rating)

    metrics = {
        'by_difficulty': defaultdict(lambda: {'correct': 0, 'total': 0, 'times': []}),
        'by_quote_length': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_difficulty_score': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_book_popularity': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_named_entities': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_publication_year': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_rarity_ratio': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'errors': [],
        'response_times': [],
        'error_count': 0,
    }

    total_questions = sum(len(examples) for examples in sampled_data.values())
    question_num = 0

    for difficulty, examples in sampled_data.items():
        if not examples:
            continue

        print(f"\n   Testing {len(examples)} examples at '{difficulty}' difficulty level...")

        for ex in tqdm(examples, desc=f"   {difficulty.capitalize()}", unit="q"):
            question_num += 1

            # generate prompt from quote instead of using sample stored prompt
            prompt = generate_prompt(ex["quote"])
            prediction, duration, had_error = query_model(prompt, model_name)

            is_correct = eval(prediction, ex, OPENROUTER_API_KEY) if not had_error else False

            metadata = ex.get("metadata", {})
            quote_length = len(ex["quote"])

            difficulty_score = metadata.get("score", 0)
            book_popularity = metadata.get("ratings_count", 0)
            entities_count = metadata.get("entities_count", 0)
            publication_year = metadata.get("publication_year", None)
            rarity_ratio = metadata.get("rarity_ratio", None)

            # categorize all metadata
            length_category = categorize_quote_length(quote_length)
            score_category = categorize_difficulty_score(difficulty_score)
            popularity_category = categorize_book_popularity(book_popularity, all_ratings)
            entity_category = categorize_named_entities(entities_count)
            year_category = categorize_publication_year(publication_year)
            rarity_category = categorize_rarity_ratio(rarity_ratio)

            # update metrics
            metrics['by_difficulty'][difficulty]['correct'] += int(is_correct)
            metrics['by_difficulty'][difficulty]['total'] += 1
            metrics['by_difficulty'][difficulty]['times'].append(duration)

            metrics['by_quote_length'][length_category]['correct'] += int(is_correct)
            metrics['by_quote_length'][length_category]['total'] += 1

            metrics['by_difficulty_score'][score_category]['correct'] += int(is_correct)
            metrics['by_difficulty_score'][score_category]['total'] += 1

            metrics['by_book_popularity'][popularity_category]['correct'] += int(is_correct)
            metrics['by_book_popularity'][popularity_category]['total'] += 1

            metrics['by_named_entities'][entity_category]['correct'] += int(is_correct)
            metrics['by_named_entities'][entity_category]['total'] += 1

            metrics['by_publication_year'][year_category]['correct'] += int(is_correct)
            metrics['by_publication_year'][year_category]['total'] += 1

            metrics['by_rarity_ratio'][rarity_category]['correct'] += int(is_correct)
            metrics['by_rarity_ratio'][rarity_category]['total'] += 1

            metrics['response_times'].append(duration)

            if had_error:
                metrics['error_count'] += 1

            if not is_correct:
                metrics['errors'].append({
                    'id': ex['id'],
                    'quote': ex['quote'][:100] + '...' if len(ex['quote']) > 100 else ex['quote'],
                    'true_answer': ex['answer'],
                    'predicted_answer': prediction[:100] + '...' if len(prediction) > 100 else prediction,
                    'difficulty': difficulty,
                    'difficulty_score': difficulty_score,
                    'metadata': metadata
                })

            results.append({
                "question_num": question_num,
                "id": ex["id"],
                "model_name": model_name,
                "quote": ex["quote"],
                "true_answer": ex["answer"],
                "predicted_answer": prediction,
                "is_correct": is_correct,
                "had_error": had_error,
                "response_time_sec": duration,

                "difficulty": difficulty,
                "difficulty_score": difficulty_score,
                "score_category": score_category,

                "quote_length": quote_length,
                "length_category": length_category,

                "book_popularity": book_popularity,
                "popularity_category": popularity_category,

                "entities_count": entities_count,
                "entity_category": entity_category,

                "publication_year": publication_year,
                "year_category": year_category,

                "rarity_ratio": rarity_ratio,
                "rarity_category": rarity_category,

                "metadata": metadata
            })

            status = "✅" if is_correct else "❌"
            print(f"\n      {status} Q{question_num}/{total_questions} | "
                  f"Score: {difficulty_score:.2f} | "
                  f"Time: {duration:.2f}s")
            print(f"         Expected: {ex['answer']}")
            print(f"         Got: {prediction[:80]}...")

    return results, metrics

### VISUALIZATION ###

def create_accuracy_by_difficulty_chart(all_model_results, output_dir):
    """
    Create grouped bar chart with each model showing its performance across difficulties.
    Each model gets its own group with three bars (easy, medium, hard) side by side.
    """
    if not VIZ_AVAILABLE:
        return

    print("\nCreating Accuracy by Difficulty Chart...")

    try:
        fig, ax = plt.subplots(figsize=(14, 7))

        difficulties = ['easy', 'medium', 'hard']
        model_names = list(all_model_results.keys())
        x = np.arange(len(model_names))
        width = 0.25

        colors = {'easy': '#4CAF50', 'medium': '#FF9800', 'hard': '#F44336'}

        for idx, diff in enumerate(difficulties):
            accuracies = []
            for model_name in model_names:
                results = all_model_results[model_name]
                diff_results = [r for r in results if r['difficulty'] == diff]
                if diff_results:
                    acc = sum(r['is_correct'] for r in diff_results) / len(diff_results) * 100
                    accuracies.append(acc)
                else:
                    accuracies.append(0)

            bottom_padding = 2  # Define the padding amount
            bars = ax.bar(x + idx * width, accuracies, width, label=diff.capitalize(),
                          alpha=0.8, color=colors[diff], edgecolor='black', linewidth=1.2,
                          bottom=bottom_padding)  # Add bottom parameter to lift the bars

            # add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2.,
                            height + bottom_padding + 2,  # Adjust text position to match the new bar bottom
                            f'{height:.1f}%',
                            ha='center',
                            va='bottom',
                            fontsize=15,
                            fontweight='bold',
                            rotation=90)

        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title('Model Performance Across Difficulty Levels', fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend(title='Difficulty', fontsize=11, title_fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 110)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_difficulty.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   Created: accuracy_by_difficulty.png")

    except Exception as e:
        print(f"   Error creating the chart: {e}")

def create_average_accuracy_by_difficulty_chart(all_model_results, output_dir):
    """
    Create a simple bar chart showing the AVERAGE model performance across difficulties.
    This provides a high-level view of task difficulty.
    """
    if not VIZ_AVAILABLE:
        return

    print("\nCreating Average Accuracy by Difficulty Chart...")


    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        difficulties = ['easy', 'medium', 'hard']
        average_accuracies = []

        # Calculate the average accuracy for each difficulty level
        for diff in difficulties:
            # Collect the accuracy of every model for the current difficulty
            all_accuracies_for_diff = []
            for model_name, results in all_model_results.items():
                diff_results = [r for r in results if r['difficulty'] == diff]
                if diff_results:
                    acc = sum(r['is_correct'] for r in diff_results) / len(diff_results) * 100
                    all_accuracies_for_diff.append(acc)

            # Calculate the average of all model accuracies for this difficulty
            if all_accuracies_for_diff:
                avg_acc = sum(all_accuracies_for_diff) / len(all_accuracies_for_diff)
                average_accuracies.append(avg_acc)
            else:
                average_accuracies.append(0) # append 0 if no data

        # Plot the simple bar chart
        colors = ['#4CAF50', '#FF9800', '#F44336'] # Green, Orange, Red
        bars = ax.bar(difficulties, average_accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=18, fontweight='bold')

        ax.set_xlabel('Difficulty Level', fontsize=13, fontweight='bold')
        ax.set_ylabel('Average Accuracy Across All Models (%)', fontsize=13, fontweight='bold')
        ax.set_title('Average Model Performance vs. Task Difficulty', fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.4, linestyle='--')
        ax.set_ylim(0, 110)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_accuracy_by_difficulty.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   Created: average_accuracy_by_difficulty.png")

    except Exception as e:
        print(f"   Error creating average accuracy chart: {e}")



def create_performance_degradation_chart(all_model_results, output_dir):
    """
    Create simplified line chart showing AVERAGE accuracy decline from easy to hard across all models.
    """
    if not VIZ_AVAILABLE:
        return

    print("\nCreating Average Performance Degradation Chart...")

    try:
        fig, ax = plt.subplots(figsize=(10, 7))

        difficulties = ['Easy', 'Medium', 'Hard']
        avg_accuracies = []

        for diff in ['easy', 'medium', 'hard']:
            all_accuracies = []
            for model_name, results in all_model_results.items():
                diff_results = [r for r in results if r['difficulty'] == diff]
                if diff_results:
                    acc = sum(r['is_correct'] for r in diff_results) / len(diff_results) * 100
                    all_accuracies.append(acc)

            # Calculate average across all models
            if all_accuracies:
                avg_acc = sum(all_accuracies) / len(all_accuracies)
                avg_accuracies.append(avg_acc)
            else:
                avg_accuracies.append(0)

        # Plot the average line
        ax.plot(difficulties, avg_accuracies, marker='o', linewidth=4,
                markersize=15, label='Average Across All Models',
                color='#2E86AB', alpha=0.9)

        # Add shaded area under the line
        ax.fill_between(range(len(difficulties)), avg_accuracies, alpha=0.3, color='#2E86AB')

        # Add value labels
        for i, acc in enumerate(avg_accuracies):
            ax.text(i, acc + 3, f'{acc:.1f}%', ha='center', fontsize=13, fontweight='bold', color='#2E86AB')

        # Calculate and display the drop
        drop = avg_accuracies[0] - avg_accuracies[2]
        ax.annotate(f'Average Drop: {drop:.1f}pp',
                    xy=(1, (avg_accuracies[0] + avg_accuracies[2]) / 2),
                    fontsize=12, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Difficulty Level', fontsize=13, fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title('Average Performance Degradation: Easy → Medium → Hard',
                     fontsize=15, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='best')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0, 110)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_performance_degradation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   Created: average_performance_degradation.png")

    except Exception as e:
        print(f"   Error creating degradation chart: {e}")

def create_response_time_analysis(all_model_results, output_dir):
    """
    Create simplified bar chart showing average response time across all models.
    """
    if not VIZ_AVAILABLE:
        return

    print("\nCreating Average Response Time Chart...")

    try:
        fig, ax = plt.subplots(figsize=(12, 7))

        model_names = list(all_model_results.keys())
        avg_times = []

        for model_name in model_names:
            results = all_model_results[model_name]
            df = pd.DataFrame(results)
            avg_time = df['response_time_sec'].mean()
            avg_times.append(avg_time)

        bars = ax.bar(range(len(model_names)), avg_times, color='skyblue', alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, avg_times)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{time:.2f}s',
                    ha='center',
                    va='bottom',
                    fontsize=13,  # Increased font size
                    fontweight='bold')

        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Average Response Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_title('Average Response Time Across All Queries', fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_response_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   Created: average_response_time.png")

    except Exception as e:
        print(f"   Error creating response time chart: {e}")


def create_metadata_analysis_charts(results, output_dir, model_name):
    """
    Create 2x3 grid of charts analyzing performance across all key metadata.
    """
    if not VIZ_AVAILABLE:
        return

    print(f"\nCreating Metadata Analysis Charts for {model_name}...")

    try:
        fig, axes = plt.subplots(2, 3, figsize=(21, 12))
        axes = axes.flatten()
        df = pd.DataFrame(results)

        charts_config = [
            {
                'ax': axes[0],
                'category': 'length_category',
                'title': 'Performance by Quote Length',
                'color': 'skyblue',
                'order': ['very_short', 'short', 'medium', 'long', 'very_long'],
                'explanation': 'Quote Length: Longer quotes provide more context and unique phrases for matching.'
            },
            {
                'ax': axes[1],
                'category': 'score_category',
                'title': 'Performance by Difficulty Score',
                'color': 'lightcoral',
                # 'order': ['Below -3', 'Between -3 and 0', 'Above 0'],
                'order': ['Easy (Score < -3)', 'Medium (Score -3 to 0)', 'Hard (Score 0 to 3)',
                          'Very Hard (Score > 3)'],
                'explanation': 'Difficulty Score: Combines popularity, rarity, and other factors into overall difficulty.'
            },
            {
                'ax': axes[2],
                'category': 'popularity_category',
                'title': 'Performance by Book Popularity',
                'color': 'lightgreen',
                'order': ['Bottom Third (Unpopular)', 'Middle Third (Moderate)', 'Top Third (Popular)'],
                'explanation': 'Book Popularity: More popular books have better online coverage and training data presence.'
            },
            {
                'ax': axes[3],
                'category': 'entity_category',
                'title': 'Performance by Named Entities',
                'color': 'plum',
                'order': ['No Entities', 'Few Entities (1-2)', 'Many Entities (3+)'],
                'explanation': 'Named Entities: Names and places are highly distinctive and make quotes searchable.'
            },
            {
                'ax': axes[4],
                'category': 'year_category',
                'title': 'Performance by Publication Year',
                'color': 'gold',
                'order': ['Pre-1950', '1950-1980', '1980-2000', 'Post-2000'],
                'explanation': 'Publication Year: Newer books often have better digital availability and online discussion.'
            },
            {
                'ax': axes[5],
                'category': 'rarity_category',
                'title': 'Performance by Lexical Rarity',
                'color': 'lightpink',
                'order': ['Low Rarity (Generic)', 'Medium Rarity', 'High Rarity (Unique)'],
                'explanation': 'Lexical Rarity: Unique rare words make quotes stand out and easier to locate through search.'
            },
        ]

        for config in charts_config:
            ax = config['ax']
            category = config['category']

            print(f"   {config['explanation']}")

            if category not in df.columns:
                print(f"   Metadata key '{category}' not found, skipping chart.")
                continue

            stats = df.groupby(category)['is_correct'].agg(['mean', 'count'])
            if 'order' in config:
                stats = stats.reindex([s for s in config['order'] if s in stats.index])
            else:
                stats = stats.sort_index()

            bars = ax.bar(range(len(stats)), stats['mean'] * 100, alpha=0.7, color=config['color'],
                          edgecolor='black', linewidth=1)
            ax.set_xticks(range(len(stats)))
            ax.set_xticklabels(stats.index, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11)
            ax.set_title(config['title'], fontweight='bold', fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, 110)

            # Add count and percentage labels
            for i, (idx, row) in enumerate(stats.iterrows()):
                ax.text(i, row['mean'] * 100 + 2, f"{row['mean'] * 100:.1f}%\n(n={int(row['count'])})",
                        ha='center', fontsize=8, fontweight='bold')

        plt.suptitle(f'{model_name}: Detailed Metadata Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        safe_name = model_name.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f'{safe_name}_metadata_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Created: {safe_name}_metadata_analysis.png")

    except Exception as e:
        print(f"   Error creating metadata charts: {e}")

def create_comparative_metadata_charts(all_model_results, output_dir):
    """
    Creates one comparative chart per metadata feature, showing all models
    side-by-side with detailed labels on each bar.
    """
    if not VIZ_AVAILABLE:
        return

    print(f"\nCreating Comparative Metadata Analysis Charts...")

    try:
        # Combine all results into a single pandas DataFrame for easy plotting
        all_results_list = [item for results in all_model_results.values() for item in results]
        if not all_results_list:
            print("   No results to plot. Skipping metadata charts.")
            return

        df = pd.DataFrame(all_results_list)

        # Define the charts to create
        charts_to_create = {
            'length_category': {
                'title': 'Comparative Performance by Quote Length',
                'order': ['very_short', 'short', 'medium', 'long', 'very_long']
            },
            'year_category': {
                'title': 'Comparative Performance by Publication Year',
                'order': ['Pre-1950', '1950-1980', '1980-2000', 'Post-2000', 'Unknown']
            },
            'rarity_category': {
                'title': 'Comparative Performance by Lexical Rarity',
                'order': ['Low Rarity (Generic)', 'Medium Rarity', 'High Rarity (Unique)', 'Unknown']
            },
            'entity_category': {
                'title': 'Comparative Performance by Named Entities',
                'order': ['No Entities', 'Few Entities (1-2)', 'Many Entities (3+)']
            },
            'popularity_category': {
                'title': 'Comparative Performance by Book Popularity',
                'order': ['Bottom Third (Unpopular)', 'Middle Third (Moderate)', 'Top Third (Popular)', 'Unknown']
            },
            'score_category': {
                'title': 'Comparative Performance by Difficulty Score',
                'order': ['Easy (Score < -3)', 'Medium (Score -3 to 0)', 'Hard (Score 0 to 3)', 'Very Hard (Score > 3)']
            }
        }

        model_names = sorted(df['model_name'].unique())

        for category_col, config in charts_to_create.items():
            print(f"   • Generating chart: {config['title']}...")

            # Calculate the stats we need: mean accuracy and count for each group
            stats = df.groupby(['model_name', category_col])['is_correct'].agg(['mean', 'count']).reset_index()
            stats['mean'] *= 100  # Convert to percentage

            fig, ax = plt.subplots(figsize=(18, 9))

            num_models = len(model_names)
            num_categories = len(config['order'])
            bar_width = 0.8 / num_categories

            # Get colors from the predefined palette
            colors = sns.color_palette(METADATA_PALETTES.get(category_col, 'husl'), n_colors=num_categories)

            for i, category_name in enumerate(config['order']):
                category_stats = stats[stats[category_col] == category_name]
                model_map = {row['model_name']: (row['mean'], row['count']) for _, row in category_stats.iterrows()}

                accuracies = [model_map.get(model, (0, 0))[0] for model in model_names]
                counts = [model_map.get(model, (0, 0))[1] for model in model_names]

                index = np.arange(num_models)
                bar_positions = index + (i - num_categories / 2 + 0.5) * bar_width

                bars = ax.bar(bar_positions, accuracies, bar_width, label=category_name, color=colors[i],
                              edgecolor='black', linewidth=1)

                # Add the custom styled labels to each bar
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    if height > 0 or count > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 3,
                            rf"$\mathbf{{{height:.1f}\%}}$" + f" ({int(count)})",
                            ha='center',
                            va='bottom',
                            fontsize=10,  # Adjusted fontsize for better fit
                            rotation=90
                        )

            ax.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title(config['title'], fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(np.arange(num_models))
            ax.set_xticklabels(model_names, rotation=15, ha="right")
            ax.legend(title=category_col.replace('_', ' ').title(), bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.set_ylim(0, 120)  # Increased ylim to make space for labels
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()

            safe_filename = f"comparison_by_{category_col}.png"
            plt.savefig(os.path.join(output_dir, safe_filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"     ✅ Created: {safe_filename}")

    except Exception as e:
        print(f"   ❌ Error creating comparative metadata charts: {e}")

def create_response_time_by_difficulty_chart(all_model_results, output_dir):
    """
    Creates a grouped bar chart comparing average response times on
    'easy' vs. 'hard' samples for each model.
    """
    if not VIZ_AVAILABLE:
        print("Visualization libraries not available.")
        return

    print("\nCreating Response Time (Easy vs. Hard) Chart...")

    try:
        # Combine results into a single DataFrame
        all_results_list = [item for results in all_model_results.values() for item in results]
        if not all_results_list:
            print("   No results to plot.")
            return
        df = pd.DataFrame(all_results_list)

        # Filter for only easy and hard difficulties
        df_filtered = df[df['difficulty'].isin(['easy', 'hard'])]

        if df_filtered.empty:
            print("   No 'easy' or 'hard' samples found to compare response times.")
            return

        # Plotting with Seaborn
        plt.figure(figsize=(16, 8))
        ax = sns.barplot(
            data=df_filtered,
            x='model_name',
            y='response_time_sec',
            hue='difficulty',
            hue_order=['easy', 'hard'],
            palette={'easy': '#2ECC71', 'hard': '#E74C3C'},  # Green for easy, Red for hard
            edgecolor='black',
            linewidth=1.2,
            errorbar=None  # Add this line to remove confidence intervals
        )

        # Add value labels to each bar
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.2f}s",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points',
                            fontsize=16,
                            fontweight='bold')

        ax.set_title('Average Response Time: Easy vs. Hard Samples', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Response Time (seconds)', fontsize=1, fontweight='bold')
        plt.xticks(rotation=15, ha="right")
        # Set y-axis limit with some padding
        if not df_filtered.empty:
            fixed_limit = 30
            plt.ylim(0, fixed_limit)

        plt.legend(title='Difficulty')
        plt.tight_layout()

        # Save the chart
        safe_filename = "response_time_easy_vs_hard.png"
        plt.savefig(os.path.join(output_dir, safe_filename), dpi=300)
        plt.close()
        print(f"   ✅ Created: {safe_filename}")

    except Exception as e:
        print(f"   ❌ Error creating response time comparison chart: {e}")

### REPORTING ###

def export_overall_performance_table(all_model_results, output_dir):
    """
    Export overall performance comparison table to CSV.
    """
    if not VIZ_AVAILABLE:
        return

    print("\nExporting Overall Performance Comparison Table...")
    print("   TABLE PURPOSE: Summarizes accuracy and response time for each model.")
    print("   COLUMNS: Overall accuracy, accuracy by difficulty (Easy/Medium/Hard), avg response time.")

    try:
        os.makedirs(output_dir, exist_ok=True)

        summary_data = []
        for model_name, results in all_model_results.items():
            df = pd.DataFrame(results)
            overall_accuracy = df['is_correct'].mean() * 100
            avg_time = df['response_time_sec'].mean()

            difficulty_accuracy = df.groupby('difficulty')['is_correct'].mean() * 100

            summary_data.append({
                'Model': model_name,
                'Overall Accuracy (%)': overall_accuracy,
                'Easy (%)': difficulty_accuracy.get('easy', 0),
                'Medium (%)': difficulty_accuracy.get('medium', 0),
                'Hard (%)': difficulty_accuracy.get('hard', 0),
                'Avg Response Time (s)': avg_time
            })

        summary_df = pd.DataFrame(summary_data)
        filepath = os.path.join(output_dir, '1_overall_performance_comparison.csv')
        summary_df.to_csv(filepath, index=False, encoding='utf-8-sig', float_format='%.2f')
        print(f"   Saved: 1_overall_performance_comparison.csv")

        return summary_df

    except Exception as e:
        print(f"   Error exporting overall performance table: {e}")
        return None


def export_feature_impact_table(all_model_results, output_dir):
    """
    Export feature impact (performance gap) table to CSV.
    """
    if not VIZ_AVAILABLE:
        return

    print("\nExporting Feature Impact Comparison Table...")
    print("   TABLE PURPOSE: Shows the performance drop between best and worst cases for each feature.")
    print("   INTERPRETATION: Larger positive gaps = feature significantly impacts performance.")
    print("   USE CASE: Identify which features make queries hardest for each model.")

    try:
        os.makedirs(output_dir, exist_ok=True)

        impact_definitions = {
            'Lexical Rarity': ('rarity_category', 'Low Rarity (Generic)', 'High Rarity (Unique)'),
            'Publication Year': ('year_category', 'Post-2000', 'Pre-1950'),
            'Named Entities': ('entity_category', 'Many Entities (3+)', 'No Entities'),
            'Book Popularity': ('popularity_category', 'Top Third (Popular)', 'Bottom Third (Unpopular)'),
            'Quote Length': ('length_category', 'very_long', 'very_short'),
            'Difficulty Score': ('score_category', 'Easy (Score < -3)', 'Very Hard (Score > 3)')
        }

        impact_data = defaultdict(dict)
        model_names = sorted(list(all_model_results.keys()))

        for model_name in model_names:
            results = all_model_results[model_name]
            df = pd.DataFrame(results)

            for feature_name, (col, best_case, worst_case) in impact_definitions.items():
                if col not in df.columns:
                    continue

                best_df = df[df[col] == best_case]
                worst_df = df[df[col] == worst_case]

                acc_best = best_df['is_correct'].mean() * 100 if not best_df.empty else 0
                acc_worst = worst_df['is_correct'].mean() * 100 if not worst_df.empty else 0

                performance_gap = acc_best - acc_worst
                impact_data[feature_name][model_name] = performance_gap if not pd.isna(performance_gap) else 0

        impact_df = pd.DataFrame(impact_data).T
        impact_df.index.name = "Feature (Impact)"

        filepath = os.path.join(output_dir, '2_feature_impact_comparison.csv')
        impact_df.to_csv(filepath, encoding='utf-8-sig', float_format='%.1f')
        print(f"   Saved: 2_feature_impact_comparison.csv")

        return impact_df

    except Exception as e:
        print(f"   Error exporting feature impact table: {e}")
        return None


def export_detailed_metadata_performance(all_model_results, output_dir):
    """
    Export detailed performance breakdown for all metadata categories to CSV.
    """
    if not VIZ_AVAILABLE:
        return

    print("\nExporting Detailed Metadata Performance Table...")
    print("   TABLE PURPOSE: Shows performance for every sub-category of each feature.")
    print("   GRANULARITY: Full breakdown revealing performance trends across all metadata dimensions.")

    try:
        os.makedirs(output_dir, exist_ok=True)

        metadata_definitions = {
            'Publication Year': ('year_category', ['Pre-1950', '1950-1980', '1980-2000', 'Post-2000']),
            'Lexical Rarity': ('rarity_category', ['Low Rarity (Generic)', 'Medium Rarity', 'High Rarity (Unique)']),
            'Named Entities': ('entity_category', ['No Entities', 'Few Entities (1-2)', 'Many Entities (3+)']),
            'Book Popularity': (
            'popularity_category', ['Bottom Third (Unpopular)', 'Middle Third (Moderate)', 'Top Third (Popular)']),
            'Quote Length': ('length_category', ['very_short', 'short', 'medium', 'long', 'very_long']),
            # 'Difficulty Score': ('score_category', ['Below -3', 'Between -3 and 0', 'Above 0'])
            'Difficulty Score': ('score_category',
                                 ['Easy (Score < -3)', 'Medium (Score -3 to 0)', 'Hard (Score 0 to 3)',
                                  'Very Hard (Score > 3)'])
        }

        model_names = sorted(list(all_model_results.keys()))
        detailed_list = []

        for feature_name, (col, categories) in metadata_definitions.items():
            for category in categories:
                row_data = {'Feature': feature_name, 'Category': category}

                for model in model_names:
                    results = all_model_results[model]
                    df = pd.DataFrame(results)

                    if col in df.columns:
                        subset = df[df[col] == category]
                        accuracy = subset['is_correct'].mean() * 100 if not subset.empty else 0
                        count = len(subset)
                        row_data[f"{model} (%)"] = accuracy
                        row_data[f"{model} (n)"] = count
                    else:
                        row_data[f"{model} (%)"] = 0
                        row_data[f"{model} (n)"] = 0

                detailed_list.append(row_data)

        detailed_df = pd.DataFrame(detailed_list)
        filepath = os.path.join(output_dir, '3_detailed_metadata_performance.csv')
        detailed_df.to_csv(filepath, index=False, encoding='utf-8-sig', float_format='%.1f')
        print(f"   Saved: 3_detailed_metadata_performance.csv")

        return detailed_df

    except Exception as e:
        print(f"   Error exporting detailed metadata table: {e}")
        return None


def save_results_json(results, model_name, output_dir):
    """Save detailed results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    file_path = os.path.join(output_dir, f"{safe_name}_detailed_results.json")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"   Saved: {safe_name}_detailed_results.json")
    except Exception as e:
        print(f"   Error saving JSON: {e}")


def save_results_csv(results, model_name, output_dir):
    """Save results to CSV for easy analysis."""
    if not VIZ_AVAILABLE:
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = model_name.replace('/', '_').replace(' ', '_')
        file_path = os.path.join(output_dir, f"{safe_name}_results.csv")

        df = pd.DataFrame(results)
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"   Saved: {safe_name}_results.csv")
    except Exception as e:
        print(f"   Error saving CSV: {e}")


def generate_markdown_report(all_model_results, all_stats, dataset_stats, output_dir,
                             overall_perf_df, feature_impact_df, detailed_metadata_df):
    """Generate a comprehensive markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.md")

    print("   Generating report with all findings into a readable document.")

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Quote Attribution Challenge - Evaluation Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset statistics
            f.write("## Dataset Statistics\n\n")
            f.write(f"- **Total Questions:** {dataset_stats['total_questions']}\n")
            f.write(f"- **Difficulty Distribution:**\n")
            for diff, count in dataset_stats['difficulty_distribution'].items():
                f.write(f"  - {diff.capitalize()}: {count}\n")
            f.write(f"- **Average Quote Length:** {dataset_stats['avg_quote_length']:.1f} characters\n")
            f.write(f"- **Average Difficulty Score:** {dataset_stats['avg_difficulty_score']:.2f}\n")
            f.write(
                f"- **Quote Length Range:** {dataset_stats['quote_length_range'][0]} - {dataset_stats['quote_length_range'][1]}\n")
            f.write(
                f"- **Difficulty Score Range:** {dataset_stats['difficulty_score_range'][0]:.2f} - {dataset_stats['difficulty_score_range'][1]:.2f}\n\n")

            # overall performance comparison
            if overall_perf_df is not None:
                f.write("## Overall Performance Comparison\n\n")
                f.write(overall_perf_df.to_markdown(index=False, floatfmt=".2f"))
                f.write("\n\n")

            # feature impact comparison
            if feature_impact_df is not None:
                f.write("## Feature Impact Comparison (Performance Gap)\n\n")
                f.write(
                    "This table shows the performance drop (in percentage points) from the best case to the worst case for each feature.\n")
                f.write(
                    "**Interpretation:** Larger values indicate that the feature has a significant impact on model performance.\n\n")
                f.write(feature_impact_df.to_markdown(floatfmt=".1f"))
                f.write("\n\n")

            # Detailed metadata performance
            if detailed_metadata_df is not None:
                f.write("## Detailed Metadata Performance Breakdown\n\n")
                f.write(
                    "This table shows the accuracy for every sub-category of each feature, revealing the full performance trend.\n\n")
                f.write(detailed_metadata_df.to_markdown(index=False, floatfmt=".1f"))
                f.write("\n\n")

            # Detailed analysis per model
            f.write("## Detailed Analysis by Model\n\n")

            for model_name, results in all_model_results.items():
                stats = all_stats[model_name]
                f.write(f"### {model_name}\n\n")

                f.write(f"**Overall Statistics:**\n")
                f.write(f"- Accuracy: {stats['overall_accuracy']:.2f}%\n")
                f.write(f"- Correct Answers: {stats['total_correct']}/{stats['total_questions']}\n")
                f.write(f"- API Errors: {stats['total_errors']}\n")
                f.write(f"- Average Response Time: {stats['avg_response_time']:.3f}s\n")
                f.write(f"- Median Response Time: {stats['median_response_time']:.3f}s\n\n")

                if 'correlation_difficulty_accuracy' in stats:
                    f.write(f"**Correlations:**\n")
                    f.write(f"- Difficulty Score vs Accuracy: {stats['correlation_difficulty_accuracy']:.3f}\n")
                    f.write(f"- Quote Length vs Accuracy: {stats['correlation_length_accuracy']:.3f}\n")
                    f.write(f"- Book Popularity vs Accuracy: {stats['correlation_popularity_accuracy']:.3f}\n\n")

                f.write("---\n\n")

            # Key findings
            f.write("## Key Findings\n\n")

            best_model = max(all_stats.items(), key=lambda x: x[1]['overall_accuracy'])
            f.write(
                f"1. **Best Performing Model:** {best_model[0]} with {best_model[1]['overall_accuracy']:.2f}% accuracy\n\n")

            avg_drops = []
            for model_name, results in all_model_results.items():
                df = pd.DataFrame(results)
                easy_acc = df[df['difficulty'] == 'easy']['is_correct'].mean() * 100
                hard_acc = df[df['difficulty'] == 'hard']['is_correct'].mean() * 100
                drop = easy_acc - hard_acc
                avg_drops.append(drop)

            avg_drop = sum(avg_drops) / len(avg_drops)
            f.write(f"2. **Average Performance Drop (Easy → Hard):** {avg_drop:.2f} percentage points\n\n")

            model_drops = {}
            for model_name, results in all_model_results.items():
                df = pd.DataFrame(results)
                easy_acc = df[df['difficulty'] == 'easy']['is_correct'].mean() * 100
                hard_acc = df[df['difficulty'] == 'hard']['is_correct'].mean() * 100
                model_drops[model_name] = easy_acc - hard_acc

            most_consistent = min(model_drops.items(), key=lambda x: x[1])
            f.write(f"3. **Most Consistent Model:** {most_consistent[0]} with only {most_consistent[1]:.2f}pp drop\n\n")

            fastest_model = min(all_stats.items(), key=lambda x: x[1]['avg_response_time'])
            f.write(
                f"4. **Fastest Model:** {fastest_model[0]} with {fastest_model[1]['avg_response_time']:.3f}s average response time\n\n")

            f.write("## Generated Visualizations\n\n")
            f.write("The following charts have been generated in the `analysis/` directory:\n\n")
            f.write("- `accuracy_by_difficulty.png` - Comparison of accuracy across difficulty levels\n")
            f.write("- `performance_degradation.png` - Performance decline with increasing difficulty\n")
            f.write("- `response_time_by_difficulty.png` - Response time variation by difficulty\n")
            f.write("- `[model]_metadata_analysis.png` - Metadata-based performance for each model\n\n")

            f.write("## Generated CSV Reports\n\n")
            f.write("The following CSV files have been generated in the `analysis/csv_reports/` directory:\n\n")
            f.write("- `1_overall_performance_comparison.csv` - Overall model performance metrics\n")
            f.write("- `2_feature_impact_comparison.csv` - Performance gaps for each feature\n")
            f.write("- `3_detailed_metadata_performance.csv` - Detailed breakdown by all categories\n\n")

        print(f"   Generated: evaluation_report.md")

    except Exception as e:
        print(f"   Error generating report: {e}")


def print_comprehensive_summary(all_model_results, all_stats, dataset_stats):
    """Print a comprehensive summary to console with enhanced explanations."""
    print("\n" + "=" * 80)
    print(" " * 25 + "COMPREHENSIVE SUMMARY")
    print("=" * 80)

    # Dataset info
    print("\nDataset Information:")
    print(f"   Total Questions: {dataset_stats['total_questions']}")
    print(f"   Difficulty Distribution: {dataset_stats['difficulty_distribution']}")
    print(f"   Avg Quote Length: {dataset_stats['avg_quote_length']:.1f} chars")
    print(f"   Avg Difficulty Score: {dataset_stats['avg_difficulty_score']:.2f}")

    # Overall performance table
    print("\n" + "-" * 80)
    print("Overall Performance:")
    print("   (Higher accuracy = better; lower response time = faster)")
    print("-" * 80)
    print(f"{'Model':<30} | {'Accuracy':>10} | {'Correct':>10} | {'Errors':>8} | {'Avg Time':>10}")
    print("-" * 80)

    for model_name, stats in all_stats.items():
        print(f"{model_name:<30} | {stats['overall_accuracy']:>9.2f}% | "
              f"{stats['total_correct']:>3}/{stats['total_questions']:<3} | "
              f"{stats['total_errors']:>8} | {stats['avg_response_time']:>9.3f}s")

    print("-" * 80)

    # Performance by difficulty
    print("\n" + "-" * 80)
    print("Performance by Difficulty Level:")
    print("-" * 80)
    print(f"{'Model':<30} | {'Easy':>10} | {'Medium':>10} | {'Hard':>10} | {'Drop':>10}")
    print("-" * 80)

    for model_name, results in all_model_results.items():
        df = pd.DataFrame(results) if VIZ_AVAILABLE else None

        if df is not None:
            easy_acc = df[df['difficulty'] == 'easy']['is_correct'].mean() * 100
            medium_acc = df[df['difficulty'] == 'medium']['is_correct'].mean() * 100
            hard_acc = df[df['difficulty'] == 'hard']['is_correct'].mean() * 100
            drop = easy_acc - hard_acc

            print(f"{model_name:<30} | {easy_acc:>9.2f}% | {medium_acc:>9.2f}% | "
                  f"{hard_acc:>9.2f}% | {drop:>9.2f}pp")

    print("-" * 80)

    # Key insights
    print("\n" + "=" * 80)
    print(" " * 30 + "KEY INSIGHTS")
    print("=" * 80)

    best_model = max(all_stats.items(), key=lambda x: x[1]['overall_accuracy'])
    print(f"\nBest Model: {best_model[0]}")
    print(f"   Overall Accuracy: {best_model[1]['overall_accuracy']:.2f}%")
    print(f"   Correct Answers: {best_model[1]['total_correct']}/{best_model[1]['total_questions']}")
    print(f"   This model achieved the highest overall accuracy across all queries.")

    if VIZ_AVAILABLE:
        model_drops = {}
        for model_name, results in all_model_results.items():
            df = pd.DataFrame(results)
            easy_acc = df[df['difficulty'] == 'easy']['is_correct'].mean() * 100
            hard_acc = df[df['difficulty'] == 'hard']['is_correct'].mean() * 100
            model_drops[model_name] = easy_acc - hard_acc

        most_consistent = min(model_drops.items(), key=lambda x: x[1])
        print(f"\nMost Consistent Model: {most_consistent[0]}")
        print(f"   Performance Drop (Easy→Hard): {most_consistent[1]:.2f} percentage points")
        print(f"   Lower drop = more robust performance across difficulty levels.")
        print(f"   This model maintains its accuracy better when faced with challenging queries.")

    fastest_model = min(all_stats.items(), key=lambda x: x[1]['avg_response_time'])
    print(f"\n⚡ Fastest Model: {fastest_model[0]}")
    print(f"   Average Response Time: {fastest_model[1]['avg_response_time']:.3f}s")
    print(f"   Fastest models provide quicker responses, useful for real-time applications.")

    if VIZ_AVAILABLE and all_model_results:
        all_results_combined = []
        for results in all_model_results.values():
            all_results_combined.extend(results)

        df_all = pd.DataFrame(all_results_combined)

        print(f"\nDataset-Wide Statistics:")
        print(f"   Overall Accuracy (All Models): {df_all['is_correct'].mean() * 100:.2f}%")
        print(f"   Total Questions Answered: {len(df_all)}")
        print(f"   Total Correct: {df_all['is_correct'].sum()}")
        print(f"   Total API Errors: {df_all['had_error'].sum()}")

        if 'difficulty_score' in df_all.columns:
            corr = df_all['difficulty_score'].corr(df_all['is_correct'].astype(int))
            print(f"\nCorrelation Analysis:")
            print(f"   Difficulty Score ↔ Accuracy Correlation: {corr:.3f}")
            if corr < -0.3:
                print(f"   ✅ Strong negative correlation confirms difficulty metric is effective!")

    print("\n" + "=" * 80)
    print("\nAll results have been saved in structured formats for further analysis.")
    print("   CSV files provide tabular data for Excel/spreadsheet analysis")
    print("   PNG charts offer visual insights for presentations and reports")
    print("   JSON files contain complete detailed results for custom processing")
    print("   Markdown report consolidates all findings in a readable document")

def export_combined_results_csv(all_model_results, output_dir):
    """
    Combines results from all models into a single CSV file.
    """
    if not VIZ_AVAILABLE:
        return

    print("\nExporting Combined Results Table for All Models...")
    print("   TABLE PURPOSE: Provides a single, detailed sheet with every question and every model's response.")
    print("   USE CASE: Ideal for detailed filtering, sorting, and analysis in a spreadsheet application.")

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Create a single list by extending it with results from each model
        combined_results_list = []
        for model_name, results in all_model_results.items():
            # The results are already a list of dicts, so we can just add them
            combined_results_list.extend(results)

        if not combined_results_list:
            print("   No results to combine. Skipping combined CSV export.")
            return

        df = pd.DataFrame(combined_results_list)


        preferred_order = [
            "model_name", "question_num", "id", "is_correct", "had_error",
            "difficulty", "difficulty_score", "quote", "true_answer", "predicted_answer",
            "response_time_sec", "quote_length", "book_popularity", "entities_count",
            "publication_year", "rarity_ratio", "length_category", "score_category",
            "popularity_category", "entity_category", "year_category", "rarity_category",
            "metadata"
        ]
        existing_cols = [col for col in preferred_order if col in df.columns]
        df = df[existing_cols]

        filepath = os.path.join(output_dir, 'all_models_combined_results.csv')
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"   ✅ Saved: all_models_combined_results.csv")

    except Exception as e:
        print(f"   ❌ Error exporting combined results CSV: {e}")



def main():
    """Main execution function with comprehensive evaluation."""
    print("\n" + "=" * 80)
    print(" " * 15 + "STARTING EXPERIMENT PIPELINE")
    print("=" * 80)

    # step 1: prepare data
    print("\n" + "=" * 80)
    print("preparing data for analysis")
    print("=" * 80)
    sampled_test_set = load_and_sample_dataset(DATASET_PATH, N_PER_DIFFICULTY)
    if not sampled_test_set:
        print("No data loaded. pipeline stopped.")
        return
    dataset_stats = get_dataset_statistics(sampled_test_set)
    print(f"\nGathered dataset statistics.")

    # step 2: evaluating models
    print("\n" + "=" * 80)
    print("Starting to evaluate models")
    print("=" * 80)
    all_model_results = {}
    all_model_stats = {}
    for model_name in MODELS_TO_EVALUATE:
        try:
            results, _ = evaluate_model(model_name, sampled_test_set)
            stats = calculate_statistics(results)
            all_model_results[model_name] = results
            all_model_stats[model_name] = stats
            print(f"\nCompleted evaluation of {model_name}")
        except Exception as e:
            print(f"\nFailed to evaluate {model_name}: {e}")
            continue

    if not all_model_results:
        print("\nNo models were successfully evaluated. exit.")
        return

    # step 3: save results of each model
    print("\n" + "=" * 80)
    print("saving the results of each model's evaluation")
    print("=" * 80)
    for model_name, results in all_model_results.items():
        save_results_json(results, model_name, RESULTS_DIR)
        save_results_csv(results, model_name, RESULTS_DIR)
        # Add the function call here
        if VIZ_AVAILABLE:
            create_metadata_analysis_charts(results, ANALYSIS_DIR, model_name)

    # step 4: export eperformance and comparison tables to csv files
    print("\n" + "=" * 80)
    print("exporting eperformance and comparison tables to csv files")
    print("=" * 80)
    overall_perf_df = export_overall_performance_table(all_model_results, CSV_REPORTS_DIR)
    feature_impact_df = export_feature_impact_table(all_model_results, CSV_REPORTS_DIR)
    detailed_metadata_df = export_detailed_metadata_performance(all_model_results, CSV_REPORTS_DIR)
    combined_results_csv = export_combined_results_csv(all_model_results, CSV_REPORTS_DIR)

    # check if the performance of the models on the dataset differs in statistical significance
    if MODELS_TO_EVALUATE:
        analyze_feature_correlations(all_model_results, CSV_REPORTS_DIR, ANALYSIS_DIR)

    # step 5: generate charts for analysis
    print("\n" + "=" * 80)
    print("generating charts for analysis")
    print("=" * 80)
    if VIZ_AVAILABLE:
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        create_accuracy_by_difficulty_chart(all_model_results, ANALYSIS_DIR)
        create_performance_degradation_chart(all_model_results, ANALYSIS_DIR)
        create_response_time_analysis(all_model_results, ANALYSIS_DIR)
        create_comparative_metadata_charts(all_model_results, ANALYSIS_DIR)
        create_average_accuracy_by_difficulty_chart(all_model_results, ANALYSIS_DIR)
        print("\nCharts successfully generated")
    else:
        print("\nNo viz libraries available.")

    # generating comprehensive report
    print("\n" + "=" * 80)
    print("generating comprehensive report")
    print("=" * 80)
    generate_markdown_report(all_model_results, all_model_stats, dataset_stats, ANALYSIS_DIR,
                             overall_perf_df, feature_impact_df, detailed_metadata_df)

    # printing summary of all results
    print("\n" + "=" * 80)
    print("printing summary of all results")
    print("=" * 80)
    print_comprehensive_summary(all_model_results, all_model_stats, dataset_stats)


    print("\n" + "=" * 80)
    print(" " * 25 + "Pipeline completed.")
    print("=" * 80)
    print(f"\nIndividual model results (JSON/CSV) saved in: {RESULTS_DIR}/")
    print(f"Detailed comparison tables (CSV for Excel) saved in: {CSV_REPORTS_DIR}/")
    print(f"Visual analysis charts (PNG) saved in: {ANALYSIS_DIR}/")
    print(
        f"markdown report has been generated at: {os.path.join(ANALYSIS_DIR, 'evaluation_report.md')}")


if __name__ == "__main__":
    main()