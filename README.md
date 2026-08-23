# 📚 NLP PROJECT 


## 📝 Project Overview

This project tests the ability of **Large Language Models (LLMs)** to recognize the **source of literary quotes**.
Each example presents a quote from a well-known book and asks:   _"From which book is the following quote?_

The task is designed to be challenging, requiring models to rely solely on their **internal knowledge**, not external tools or search engines.
It assesses how well models handle **memorization, semantic comprehension, and contextual reasoning**. 

A key feature of this project is the **difficulty classification system** applied to every quote.
Each example is assigned a difficulty level using a creative and carefully tuned heuristic that combines multiple 
indicators - linguistic, popularity-based, and contextual. The scoring mechanism is not arbitrary; it produces clear and 
consistent accuracy differences between difficulty levels when tested on real LLMs, confirming that it effectively captures the true challenge of each example.

Overall, this project provides a rigorous and insightful benchmark for evaluating LLMs’ literary knowledge and contextual
reasoning, offering a nuanced understanding of their strengths and limitations in quote attribution

### Project Report

[View the full project report](docs/project_report.pdf)

## 🔀 Dataset Creation Pipeline 

Run the file: 

    python generate_data.py


The file performs two main steps to generate the dataset:

    if __name__ == "__main__":
        scraping_pipeline(output_file="raw_books_info.json") // Scraping Process
        build_dataset(input_file="raw_books_info.json", output_file="data/exampless.jsonl") // Dataset Building 


### 1️⃣ Scraping Process

1. Book discovery - Scrapes several Goodreads “Best Books” list pages.
2. Book metadata extraction - Collects title, author, publication year, page count, ratings, and reviews.
3. Quote extraction - Visits each book’s quotes page and saves all found quotes and their like counts.
4. Raw data saving - Stores structured results in raw_books_info.json.


   _raw_books_info.json structure_: 
    
          {
        
            "title": "string",          // The title of the book
        
            "author": "string",         // The author of the book
        
            "quotes": [                 // A list of quote objects from this book
              {
                "text": "string",       // The text of the quote
                "likes": number         // The number of likes this quote recieved
              }, 
              {
                "text": "string",       // The text of the quote
                "likes": number         // The number of likes this quote recieved
              }, 
                ..... ] , 
    
            "metadata": { // book's extracted metadata
                        "page_count": number , 
                        "publication_year": "string" (year), 
                        "ratings_count": number, 
                        "reviews_count": number 
                        }
            
          }
    


### 2️⃣ Dataset Building 

**Input:** raw_books_info.json. 

**Output:** examples.json - our final dataset ; This file will appear in the project root after successful execution.


   _examples.json structure_: 
    
        [
          {
            "id": "string",                   // "quote_{i}, i will be 0,1,2...
            "quote": "string",                // The quote text itself 
            "answer": "string",               // The book title - the correct answer 
            "author": number,                 // The author of the book 
            "metadata: { 
                        // Difficulty Level (Well explained below)
                        "difficulty": "easy|medium|hard",    // Difficulty level (categorical)
                        "score": number,                     // Final combined difficulty score
                        
                        // Length Features 
                        "length_words": number,              // Total number of words in the quote 
                        "length_bin": "string",              // Category of the previous attribute
                        "sentences": number,                 // Number of sentences in the quote

                        // Entity Features
                        "entities_count": number,            // Named entities count in the quote
                        "entities_bin": "string",           // Category of the previous attribute 

                        // Lexical Features 
                        "rarity_ratio": number,              // Rarity of words (custom metric)
                        "rarity_bin" : "string",             // Category of the previous attribute
                        "stopword_ratio": number,            // Ratio of stopwords to total words (custom metric)
                        "stopword_bin" : "string",           // Category of the previous attribute

                        // Popularity Features
                        "likes": number,                 // The number of likes this quote recieved
                        "likes_bin": "string",               // Category of the previous attribute
                        "book_popularity_norm": number,        // Normalized popularity score for the book
                        "book_popularity_adjustment": number,      // Popularity difficulty adjustment factor 

                        // Book's Metadata
                        "page_count": number , 
                        "publication_year": "string" (year), 
                        "ratings_count": number, 
                        "reviews_count": number
          },
          ...
        ] 

⚡ **Score Calculation** 
\\ Positive score → harder, negative score → easier
* **Quote Length:** Short quotes (<10 words) are harder; long quotes (>25 words) are easier. 
_Effect on Score:_ +1.5 (short), +0.5 (mid), -1.0 (long) 
* **Likes:** Less popular quotes (<1k likes) are harder; very popular quotes are easier. 
_Effect on Score:_ +1.5 (low), +0.5 (mid), -1.0 (high) 
* **Named Entities Count:** Quotes with more named entities are easier (provide context).
_Effect on Score:_ 0.5 (0), -1.5 (1-2), -2.5 (>2). 
* **Sentences:** Longer quotes with more sentences are easier. _Effect on Score:_ -0.25 (2), -0.5 (3+)
* **Stopword Ratio:** High stopword ratio (>0.62) makes quotes harder. 
STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","that","this","these","those","of","to","in","on","for",
    "from","by","with","as","at",......}. _Effect on Score:_ +0.5 (high), -0.5 (low <0.45) 
* **Lexical Rarity:** Higher proportion of rare content words makes quote easier for LLM to find out its specific source book.
_Effect on Score:_ -1.0 (high), -0.5 (mid), +0.5 (low) 
* **Book Popularity:** Popularity of the book, normalized across all books. Highly popular books (normalized score close to 1) make quotes easier.
_Effect on Score:_
  1. Take top 10 quotes by likes per book.
  2. Compute average likes of these quotes.
  3. Apply log1p transformation to tame large values.
  4. Normalize across all books using min-max scaling → book_pop_norm in [0,1]. 
  5. Score contribution: adjustment = W × (0.5 − book_pop_norm), where W=3.0




📌 **Difficulty Classification**

Once the score is computed, each quote is assigned a difficulty level:
* easy if score ≤ -3
* medium if -3 < score ≤ 0
* hard if score > 0 



## 🤖 Evaluation Framework 

Run the file: 
    
    python run_eval.py 

This script runs the **entire evaluation process**, testing several models across different difficulty levels 
and scoring their answers automatically.

Here are the specified steps:

1. Loads the dataset (data/examples.jsonl) and draws a fixed number (N_PER_DIFFICULTY) of samples from 
each difficulty level - easy, medium, and hard. This ensures that all difficulty types are **equally represented in the evaluation**.
2. Sends prompts to multiple models (MODELS_TO_EVALUATE) using **LiteLLM** via **OpenRouter API**. 
3. After collecting responses, the system checks whether each model’s answer is correct.
This is handled by the judge LLM (JUDGE_MODEL), defined in **evaluation.py**.
4. Logs detailed results including accuracy, latency, and metadata-based breakdowns. All metrics are computed 
automatically and summarized in both numeric and visual form. 

   
After running `run_eval.py`, you’ll find:

| Directory | Contents |
|------------|-----------|
| `results/` | Per-model raw responses and correctness results |
| `analysis/` | Charts showing accuracy by difficulty, quote length, rarity, etc. |
| `analysis/csv_reports/` | Summary tables of model performance |
| `evaluation_report.md` | A comprehensive report combining all results |
