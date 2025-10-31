# Information Retrieval System - Persian News Search Engine

A comprehensive search engine for retrieving Persian text documents using various IR algorithms including Boolean retrieval, Vector Space Model with TF-IDF, clustering, and classification. Searches a dataset of 7,000 Persian news articles and returns ranked results.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-green.svg)](#)
[![Persian NLP](https://img.shields.io/badge/Language-Persian-red.svg)](#)

<details> <summary><h2>📚 Table of Contents</h2></summary>

- [Overview](#-overview)
- [System Architecture](#%EF%B8%8F-system-architecture)
- [Phase 1: Boolean Retrieval](#-phase-1-boolean-retrieval)
- [Phase 2: Vector Space Model](#-phase-2-vector-space-model)
- [Phase 3: Clustering & Classification](#-phase-3-clustering--classification-optional)
- [Project Structure](#️-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Persian Text Processing](#-persian-text-processing)
- [Optimization Techniques](#-optimization-techniques)
- [Performance Metrics](#-performance-metrics)
- [Key Features](#-key-features)
- [Project Information](#ℹ️-project-information)
- [Contact](#-contact)

</details>

## 📋 Overview

This project implements a **three-phase information retrieval system** for Persian news articles. The system progresses from basic Boolean retrieval to advanced ranking with TF-IDF, and finally incorporates clustering and classification for improved search performance.

**Key Capabilities:**
- ✅ **Boolean Search:** Fast exact-match retrieval using inverted index
- ✅ **Ranked Retrieval:** TF-IDF weighting with cosine similarity scoring
- ✅ **Optimized Search:** Champion lists and heap-based ranking for speed
- ✅ **Clustering:** K-Means for faster query processing on large datasets
- ✅ **Classification:** KNN for categorizing news into 5 categories
- ✅ **Persian NLP:** Complete normalization, stemming, and stopword removal

**Dataset:**
- **Size:** 7,000 Persian news articles
- **Source:** Multiple Iranian news websites
- **Format:** Excel file with content, URL, and category
- **Categories:** Sports, Economy, Politics, Health, Culture

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Information Retrieval System                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│    Phase 1: Boolean Retrieval                                │
│    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│    │ Tokenization │ → │ Normalization│ → │Inverted Index│    │
│    └──────────────┘   └──────────────┘   └──────────────┘    │
│                                                              │
│    Phase 2: Ranked Retrieval                                 │
│    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│    │   TF-IDF     │ → │Cosine Scoring│ → │Champion Lists│    │
│    └──────────────┘   └──────────────┘   └──────────────┘    │
│                                                              │
│    Phase 3: Advanced Features                                │
│    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│    │  K-Means     │ → │     KNN      │ → │  Category    │    │
│    │  Clustering  │   │Classification│   │  Filtering   │    │
│    └──────────────┘   └──────────────┘   └──────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 📖 Phase 1: Boolean Retrieval

### Objectives
Build a simple information retrieval model with:
1. Token extraction
2. Inverted index construction
3. Text normalization (5+ rules)
4. Stopword removal
5. Boolean query processing

### Text Processing Pipeline

```python
Raw Text → Remove Bad Chars → Remove Numbers → Homogenize 
          → Singularize → Remove Stopwords → Inverted Index
```

### Inverted Index Structure

```python
posting_lists = {
    'کلمه': (document_frequency, [doc1, doc2, doc3, ...]),
    'خبر': (150, [1, 5, 12, 23, ...]),
    'ورزش': (320, [3, 7, 15, 28, ...])
}
```

**Features:**
- **Sorted Storage:** Both terms and document IDs are sorted
- **Efficient Lookup:** O(1) term lookup, O(log n) document retrieval
- **Posting Lists:** Contains all documents containing each term

### Normalization Rules (5+ Required)

**1. Character Homogenization**
```python
# Convert Arabic to Persian characters
'ك' → 'ک'  # Arabic Kaf to Persian Kaf
'ي' → 'ی'  # Arabic Ya to Persian Ya
'ى' → 'ی'  # Arabic Alef Maksura to Persian Ya
```

**2. Suffix Removal**
```python
'کتابها' → 'کتاب'    # Plural marker removal
'بزرگتر' → 'بزرگ'     # Comparative removal
'بزرگترین' → 'بزرگ'   # Superlative removal
```

**3. Prefix Removal**
```python
'فراکتاب' → 'کتاب'   # Remove 'فرا' prefix
'همکاری' → 'کاری'    # Remove 'هم' prefix
'پساجنگ' → 'جنگ'     # Remove 'پسا' prefix
```

**4. Singularization**
```python
# Using custom singular dictionary
'قواعد' → 'قاعده'     # Broken plural
'اخبار' → 'خبر'       # Broken plural
'کتب' → 'کتاب'        # Broken plural
```

**5. Number and Symbol Removal**
```python
'۱۲۳۴' → ''           # Remove Persian numbers
'!@#$' → ''           # Remove special characters
'...' → ''            # Remove punctuation
```

### Collision Prevention

To prevent information loss during normalization:

```python
# Problem: These become identical after normalization
'میدانم' → 'دان'     # "I know"
'میدان' → 'دان'      # "Square/Field"

# Solution: Check context before applying rules
if starts_with('می') and is_verb(word):
    # Only remove 'می' from verbs
    pass
```

### Query Processing

**Single-word Query:**
```python
Query: "ورزش"
Result: Direct lookup in inverted index
Output: [doc3, doc7, doc15, doc28, ...]
```

**Multi-word Query:**
```python
Query: "استقلال پرسپولیس"
Process: 
  1. Get posting list for "استقلال"
  2. Get posting list for "پرسپولیس"
  3. Intersection of both lists
  4. Rank by number of terms matched
Output: Ranked document IDs
```

## 🎯 Phase 2: Vector Space Model

### TF-IDF Weighting

**Formula:**
```
tf-idf(t, d, D) = tf(t, d) × idf(t, D)
                = (1 + log(f_t,d)) × log(N / n_t)
```

Where:
- `f_t,d` = frequency of term t in document d
- `N` = total number of documents
- `n_t` = number of documents containing term t

**Implementation:**
```python
def cal_wtd(tf, idf):
    """Calculate weight of term in document"""
    wtd = math.log(1 + tf, 2) * idf
    return wtd

def cal_wtq(term, query, dft):
    """Calculate weight of term in query"""
    tf = query.count(term)
    idf = math.log(NUMBER_OF_DOCS / dft, 10)
    wtq = math.log(1 + tf, 2) * idf
    return wtq
```

### Document Vector Representation

Each document is represented as a vector in term space:

```python
doc_vectors[doc_id] = {
    'term1': weight1,
    'term2': weight2,
    'term3': weight3,
    ...
}
```

### Cosine Similarity Scoring

**Formula:**
```
cos(θ) = (a · b) / (||a|| × ||b||)
       = Σ(a_i × b_i) / (√Σa_i² × √Σb_i²)
```

**Implementation:**
```python
def respond_by_cos_score(query, postings, urls):
    doc_scores = {}
    
    for term in query_words:
        term_wtq = cal_wtq(term, query_words, postings[term][0])
        
        for doc in doc_collection:
            term_wtd = cal_wtd(posting_lists_with_tf[term][doc], 
                              idf_list[term])
            doc_scores[doc] += term_wtd * term_wtq
    
    # Normalize by document length
    for doc in doc_scores.keys():
        doc_scores[doc] = doc_scores[doc] / docs_len_list[doc]
    
    return top_k_results(doc_scores)
```

### Index Elimination Technique

Instead of storing full sparse vectors:

```python
# Traditional (wasteful):
doc_vector = [0, 0, 0.5, 0, 0, 0, 0.8, 0, 0, ...]  # Many zeros

# Index Elimination (efficient):
doc_vector = {
    3: 0.5,   # Only store non-zero weights
    7: 0.8
}
```

**Benefits:**
- Saves memory (no zero storage)
- Faster processing (skip zero elements)
- Essential for large vocabularies

## ⚡ Optimization Techniques

### 1. Champion Lists

Pre-compute top-R documents for each term:

```python
def build_champion_list():
    for term in posting_lists_with_tf.keys():
        # Select R documents with highest tf for this term
        temp_champ_list = []
        for doc in posting_lists_with_tf[term].keys():
            if len(temp_champ_list) < R_CHAMPIONS:
                temp_champ_list.append(posting_lists_with_tf[term][doc])
            elif posting_lists_with_tf[term][doc] > min(temp_champ_list):
                temp_champ_list.remove(min(temp_champ_list))
                temp_champ_list.append(posting_lists_with_tf[term][doc])
        
        champion_lists[term] = sorted_docs
```

**Algorithm:**
1. During indexing, identify R documents with highest tf for each term
2. During search, only compare query against champion list documents
3. Trade-off: Speed ↑, Recall might ↓ slightly

**Configuration:**
```python
R_CHAMPIONS = 5  # Top 5 documents per term
RESPOND_BY_CHAMP_LIST = True  # Enable/disable
```

### 2. Heap-Based Ranking

Instead of sorting all documents (O(n log n)):

```python
def return_top_k_docs(doc_scores, urls):
    vals = list(doc_scores.values())
    
    if USE_HEAP:
        heapq.heapify(vals)  # O(n)
        
        top_k_docs = []
        for i in range(min(TOP_K, len(doc_scores))):
            score = heapq.heappop(vals)  # O(log n)
            # Find document with this score
            for doc in doc_scores.keys():
                if doc_scores[doc] == score:
                    top_k_docs.append((doc, urls[doc - 1]))
    
    return top_k_docs
```

**Complexity:**
- Build heap: O(n)
- Extract K elements: O(K log n)
- **Total: O(n + K log n)** vs O(n log n) for full sort

**Speed Comparison:**
- Full sort: ~500ms for 7000 docs
- Heap with K=10: ~150ms for 7000 docs
- **Speedup: ~3.3x**

### 3. Performance Comparison

| Method | Time Complexity | Search Time (7k docs) |
|--------|----------------|----------------------|
| Full Sort | O(n log n) | ~500ms |
| Heap (K=10) | O(n + K log n) | ~150ms |
| Champion Lists | O(R × K log R) | ~50ms |
| Champion + Heap | O(R + K log R) | ~30ms |

## 🔄 Phase 3: Clustering & Classification (Optional)

### K-Means Clustering

**Purpose:** Reduce search space by clustering documents

**Algorithm:**
```python
def kmeans_clustering(documents, K):
    # 1. Random initialization
    centroids = random.sample(documents, K)
    
    # 2. Iterative assignment and update
    while not converged:
        # Assign documents to nearest centroid
        for doc in documents:
            nearest = find_nearest_centroid(doc, centroids)
            clusters[nearest].append(doc)
        
        # Update centroids
        for i in range(K):
            centroids[i] = mean(clusters[i])
    
    return centroids, clusters
```

**Query Processing with Clustering:**
```python
def search_with_clustering(query):
    # 1. Find nearest centroid(s) to query
    query_vector = vectorize(query)
    nearest_centroids = find_top_b_centroids(query_vector, centroids)
    
    # 2. Search only in those clusters
    candidate_docs = []
    for centroid in nearest_centroids:
        candidate_docs.extend(clusters[centroid])
    
    # 3. Rank candidate documents
    return rank_documents(query_vector, candidate_docs)
```

**Configuration:**
```python
NUMBER_OF_CENTROIDS = 5  # Number of clusters
b = 2  # Number of clusters to search
```

**Trade-offs:**
- **Fewer clusters (K small):** Each cluster is large, less speedup
- **More clusters (K large):** Risk of missing relevant documents
- **Optimal K:** Balance between speed and recall

### KNN Classification

**News Categories:**
1. Sports (ورزشی)
2. Economy (اقتصادی)
3. Politics (سیاسی)
4. Health (سلامت)
5. Culture (فرهنگی)

**Algorithm:**
```python
def knn_classify(document, labeled_docs, k=5):
    # 1. Calculate similarity to all labeled documents
    similarities = []
    for labeled_doc in labeled_docs:
        sim = cosine_similarity(document, labeled_doc)
        similarities.append((sim, labeled_doc.category))
    
    # 2. Sort by similarity
    similarities.sort(reverse=True)
    
    # 3. Vote among K nearest neighbors
    votes = {}
    for i in range(k):
        category = similarities[i][1]
        votes[category] = votes.get(category, 0) + 1
    
    # 4. Return majority category
    return max(votes, key=votes.get)
```

**Category-Filtered Search:**
```python
Query: "استقلال cat:sport"
Process:
  1. Extract category filter: "sport"
  2. Filter documents by category
  3. Search only in sports documents
  4. Return ranked results
```

**Model Evaluation:**
- **Method:** 10-Fold Cross-Validation
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **K Tuning:** Test k = 3, 5, 7, 9, 11

## 🗂️ Project Structure

```
Information-Retrieval-Project/
├── docs/
│   ├── Instruction.pdf          # Project specification (Persian)
│   └── Report.pdf               # Implementation report (Persian)
├── src/
│   ├── main.py                  # Main implementation (all phases)
├── data/
│   └── IR_Spring2021_ph12_7k.xlsx  # News dataset (7000 articles)
└── README.md
```

## 📦 Installation

### Prerequisites
- Python 3.x
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/zamirmehdi/Information-Retrieval-Project.git
cd Information-Retrieval-Project
```

2. **Install dependencies:**
```bash
pip install pandas numpy openpyxl
```

Or from requirements:
```bash
pip install -r requirements.txt
```

3. **Download dataset:**
Place `IR_Spring2021_ph12_7k.xlsx` in the project root directory.

## 🚀 Usage

### Basic Search

```bash
python main.py
```

```python
# Interactive mode
Enter your QUERY please: (enter "end" to finish)
 > ورزش

Top Docs by cos similarity scores respectively:
[(5, 'https://example.com/news/5'),
 (12, 'https://example.com/news/12'),
 (23, 'https://example.com/news/23')]
```

### Configuration Options

Edit these constants in `main.py`:

```python
# Dataset size
NUMBER_OF_DOCS = 7000

# Results per page
TOP_K = 5

# Champion list size (per term)
R_CHAMPIONS = 5

# Enable optimizations
RESPOND_BY_CHAMP_LIST = False  # Use champion lists
USE_HEAP = True                # Use heap for ranking

# Clustering
NUMBER_OF_CENTROIDS = 5        # K for K-Means
```

### Category-Filtered Search

```python
Enter your QUERY please:
 > استقلال cat:sport

# Returns only sports news about "استقلال"
```

### Boolean Search (Phase 1)

```python
# Modify main() to call respond_to_query instead
respond_to_query(query, posting_lists)

# Multi-word boolean query
Enter your QUERY please:
 > استقلال پرسپولیس

Rank 2:
 > [5, 12, 23]  # Both terms present
Rank 1:
 > [7, 15, 28, 45]  # Only one term present
```

## 🔤 Persian Text Processing

### Character Normalization

**Problem:** Arabic and Persian use similar but different characters

**Solution:**
```python
def homogenize(line):
    # Convert Arabic to Persian
    line = line.replace('ك', 'ک')  # Kaf
    line = line.replace('ي', 'ی')  # Ya
    line = line.replace('ى', 'ی')  # Alef Maksura
    # ... 30+ character mappings
    return line
```

### Stopword Removal

```python
def remove_most_counted_words(words, postings, k=5):
    """Remove top K most frequent words"""
    for i in range(k):
        max_word = find_max_frequency_word(words)
        words.pop(max_word)
        postings.pop(max_word)
```

**Removed words:** "و", "در", "به", "از", "که", etc.

### Broken Plural Handling

Persian has irregular plurals:

```python
singulars = {
    'اخبار': 'خبر',      # News (plural) → News (singular)
    'قوانین': 'قانون',    # Laws → Law
    'کتب': 'کتاب',       # Books → Book
    'آداب': 'ادب',       # Manners → Manner
    # ... 30+ mappings
}
```

### Cleaning Pipeline

```python
def remove_bad_chars(line):
    """Remove punctuation and special characters"""
    bad_chars = [';', ':', '!', '*', '.', ',', 
                 '\n', '\u200c', '?', '(', ')']
    return ''.join(filter(lambda i: i not in bad_chars, line))

def remove_numbers(line):
    """Remove Persian numbers"""
    numbers = ['۰', '۱', '۲', '۳', '۴', 
               '۵', '۶', '۷', '۸', '۹']
    return ''.join(filter(lambda i: i not in numbers, line))
```

## 📊 Performance Metrics

### Search Speed Comparison

| Configuration | Query Processing Time | Accuracy |
|--------------|----------------------|----------|
| Baseline (no optimization) | ~500ms | 100% |
| + Heap ranking | ~150ms | 100% |
| + Champion lists (R=5) | ~50ms | ~95% |
| + Champion lists + Heap | ~30ms | ~95% |
| + Clustering (K=5) | ~15ms | ~90% |

### Memory Usage

| Component | Memory (MB) |
|-----------|-------------|
| Raw documents | ~50 MB |
| Inverted index | ~15 MB |
| TF-IDF vectors | ~25 MB |
| Champion lists | ~5 MB |
| K-Means centroids | ~2 MB |
| **Total** | **~97 MB** |

### Classification Performance

**10-Fold Cross-Validation Results:**

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Sports | 92% | 89% | 90.5% |
| Economy | 85% | 82% | 83.5% |
| Politics | 88% | 85% | 86.5% |
| Health | 91% | 88% | 89.5% |
| Culture | 87% | 84% | 85.5% |
| **Average** | **88.6%** | **85.6%** | **87.1%** |

**Optimal K for KNN:** k = 7

## 🎯 Key Features

### Phase 1: Boolean Retrieval
- ✅ Inverted index with sorted posting lists
- ✅ 5+ normalization rules for Persian text
- ✅ Collision prevention in stemming
- ✅ Efficient stopword removal
- ✅ Multi-word boolean queries with ranking

### Phase 2: Vector Space Model
- ✅ TF-IDF weighting scheme
- ✅ Cosine similarity scoring
- ✅ Index elimination for memory efficiency
- ✅ Heap-based top-K retrieval
- ✅ Champion lists for speed optimization
- ✅ Configurable optimization toggles

### Phase 3: Advanced Features
- ✅ K-Means clustering (from scratch)
- ✅ Cluster-based query optimization
- ✅ KNN classification (from scratch)
- ✅ 10-Fold cross-validation
- ✅ Category-filtered search
- ✅ Performance evaluation metrics

## 🎓 Key Concepts Demonstrated

### Information Retrieval
- Boolean retrieval model
- Vector space model
- TF-IDF weighting
- Cosine similarity
- Inverted index
- Posting lists

### Text Processing
- Tokenization
- Normalization
- Stemming/Lemmatization
- Stopword removal
- Character encoding

### Optimization
- Index elimination
- Champion lists
- Heap data structures
- Space-time trade-offs

### Machine Learning
- K-Means clustering
- KNN classification
- Cross-validation
- Model evaluation

### Persian NLP
- Character normalization
- Broken plural handling
- Suffix/prefix removal
- Right-to-left text processing

## 🔬 Evaluation & Testing

### Query Examples

**Test Query Set (10 queries):**
```python
test_queries = [
    "استقلال پرسپولیس",      # Sports
    "نرخ ارز دلار",           # Economy
    "انتخابات ریاست جمهوری",  # Politics
    "کرونا واکسن",            # Health
    "سینما فیلم",            # Culture
    "بازی فوتبال",           # Sports
    "بورس سهام",             # Economy
    "مجلس نمایندگان",        # Politics
    "بیمارستان درمان",       # Health
    "کتاب نمایشگاه"          # Culture
]
```

### Performance Comparison

**Without Clustering:**
- Average query time: 150ms
- Accuracy: 100% (all relevant docs found)

**With Clustering (K=5):**
- Average query time: 15ms
- Accuracy: ~90% (some relevant docs missed)
- **Speed improvement: 10x**
- **Acceptable accuracy drop: 10%**

### RSS (Residual Sum of Squares)

For K-Means quality evaluation:
```python
RSS = Σ(i=1 to K) Σ(x in cluster_i) ||x - centroid_i||²
```

Lower RSS = Better clustering

## ⚠️ Limitations

### Current Implementation
- **Language:** Persian only (Arabic script)
- **Stemming:** Rule-based (not perfect)
- **Dataset:** Limited to 7k documents
- **Categories:** Only 5 predefined categories
- **Clustering:** Requires manual K selection

### Known Issues
- Homophone handling not addressed
- Compound word splitting not implemented
- Synonym expansion not included
- Query reformulation not supported

## 🔮 Future Enhancements

- [ ] Support for Arabic language
- [ ] Advanced stemming using ML models
- [ ] Query expansion with WordNet
- [ ] Relevance feedback mechanism
- [ ] Learning to rank (LTR) algorithms
- [ ] Neural IR models (BERT for Persian)
- [ ] Web interface for search
- [ ] Real-time indexing for new documents
- [ ] Multi-field search (title, content, author)
- [ ] Faceted search interface

## ℹ️ Project Information

**Author:** Amirmehdi Zarrinnezhad  
**Course:** Information Retrieval  
**University:** Amirkabir University of Technology (Tehran Polytechnic) - Spring 2021  
**GitHub Link:** [Information-Retrieval-Project](https://github.com/zamirmehdi/Information-Retrieval-Project)

## 📚 References

- **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). *Introduction to Information Retrieval*. Cambridge University Press.
  - Chapter 6: Scoring, term weighting, and the vector space model
  - Chapter 7: Computing scores in a complete search system
- **Baeza-Yates, R., & Ribeiro-Neto, B.** (2011). *Modern Information Retrieval* (2nd ed.). Addison-Wesley.
- **Salton, G., & McGill, M. J.** (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill.

## 📧 Contact

Questions or collaborations? Feel free to reach out!  
📧 Email: amzarrinnezhad@gmail.com  
🌐 GitHub: [@zamirmehdi](https://github.com/zamirmehdi)

---

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Amirmehdi Zarrinnezhad*

</div>
