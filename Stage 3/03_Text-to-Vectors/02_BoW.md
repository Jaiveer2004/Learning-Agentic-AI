# 02 --- Bag of Words (BoW)

## 1. What is Bag of Words?

**Bag of Words (BoW)** is a text-vectorization technique that represents
a document using the frequency or presence of words from a predefined
vocabulary. It ignores the original order and grammar of the words.

In simple terms, BoW: - Builds a vocabulary of unique words/tokens from
a corpus. - Counts how often each vocabulary word appears in each
document. - Represents each document as a numerical vector.

### Why is it called a "Bag" of Words?

Imagine putting all the words from a sentence into a bag. The bag
contains the words, but their original order is lost.

For example:

``` text
Original sentence: I love Python
Bag of words:      Python, I, love
```

The order does not matter to a basic BoW representation.

> **Important:** BoW records word occurrence or frequency. It does not
> understand the meaning of words.

------------------------------------------------------------------------

## 2. Why do we need BoW?

Machine Learning algorithms generally perform mathematical operations on
numerical data. They cannot directly use raw text as ordinary numerical
features.

Text vectorization converts text into numerical representations that
models can process.

A simplified flow is:

``` text
Human language
      ↓
Text vectorization
      ↓
Numerical representation
      ↓
Machine Learning model
      ↓
Prediction
```

BoW is one of the simplest ways to convert text documents into numerical
features.

------------------------------------------------------------------------

## 3. How does BoW work?

Consider this small corpus:

``` text
Document 1: I love Python.
Document 2: I love Machine Learning.
Document 3: Python is amazing.
```

### Step 1: Create a corpus

A **corpus** is the complete collection of text documents being
processed.

Here, the corpus contains three documents.

### Step 2: Tokenize the documents

Break each document into tokens:

``` python
D1 = ["I", "love", "Python"]
D2 = ["I", "love", "Machine", "Learning"]
D3 = ["Python", "is", "amazing"]
```

### Step 3: Build the vocabulary

Collect all unique tokens from the corpus:

``` python
vocabulary = [
    "I",
    "love",
    "Python",
    "Machine",
    "Learning",
    "is",
    "amazing"
]
```

The vocabulary contains **7 unique words**.

Assign each word a fixed column position:

    Index Word
  ------- ----------
        0 I
        1 love
        2 Python
        3 Machine
        4 Learning
        5 is
        6 amazing

### Step 4: Count word frequencies

Count how many times each vocabulary word appears in each document.

For Document 1, `I love Python`:

-   I = 1
-   love = 1
-   Python = 1
-   Machine = 0
-   Learning = 0
-   is = 0
-   amazing = 0

Its BoW vector is:

``` python
[1, 1, 1, 0, 0, 0, 0]
```

For Document 2, `I love Machine Learning`:

``` python
[1, 1, 0, 1, 1, 0, 0]
```

For Document 3, `Python is amazing`:

``` python
[0, 0, 1, 0, 0, 1, 1]
```

### Step 5: Construct the BoW matrix

Rows represent documents; columns represent vocabulary terms.

  Document     I   love   Python   Machine   Learning   is   amazing
  ---------- --- ------ -------- --------- ---------- ---- ---------
  D1           1      1        1         0          0    0         0
  D2           1      1        0         1          1    0         0
  D3           0      0        1         0          0    1         1

This is a **3 × 7 BoW matrix**: - 3 rows = 3 documents. - 7 columns = 7
unique vocabulary terms. - Each cell = frequency of that term in that
document.

------------------------------------------------------------------------

## 4. Frequency-based BoW and binary BoW

BoW can represent documents in more than one way.

### A. Count / Frequency representation

Each value records how many times a word occurs in a document.

Example:

``` text
Document: Python Python Java
Vocabulary: [Python, Java]
BoW:        [2, 1]
```

### B. Binary / Presence representation

Each value records only whether a word is present or absent.

``` text
Document: Python Python Java
Vocabulary: [Python, Java]
Binary BoW: [1, 1]
```

-   `1` = word is present.
-   `0` = word is absent.

The standard `CountVectorizer` in scikit-learn produces frequency counts
by default. Binary BoW can be requested with `binary=True`.

------------------------------------------------------------------------

## 5. OHE vs BoW

One-Hot Encoding (OHE) and BoW are related, but they represent different
things.

Consider:

``` text
Sentence: I love Python
Vocabulary: [I, love, Python]
```

### One-Hot Encoding

OHE represents each individual token separately:

  Token    One-hot vector
  -------- ----------------
  I        `[1, 0, 0]`
  love     `[0, 1, 0]`
  Python   `[0, 0, 1]`

There are three separate vectors---one for each token.

### Bag of Words

BoW combines token occurrences into one document-level vector:

``` python
[1, 1, 1]
```

If the sentence is `I love Python Python`, its count-based BoW vector
is:

``` python
[1, 1, 2]
```

Conceptually, BoW can be understood as adding the one-hot vectors of the
tokens in a document. Repeated tokens increase the corresponding count.

**Key distinction:** - **OHE:** Represents an individual category/token
with a binary vector. - **BoW:** Aggregates token occurrences to
represent an entire document.

OHE is also used for categorical features outside NLP. BoW is
specifically a common document-representation approach.

------------------------------------------------------------------------

## 6. How is BoW useful compared with individual OHE vectors?

BoW provides a convenient document-level representation.

1.  **Document-level representation:** It combines all token occurrences
    into one vector for a document.
2.  **Captures frequency:** Repeated words receive higher counts.
3.  **Simple to implement:** It is easy to understand and works with
    many traditional ML algorithms.
4.  **Useful for text classification:** It provides numerical features
    for tasks such as spam detection and sentiment classification.

### What does BoW not solve?

BoW does **not** solve all the limitations associated with OHE-style
representations.

  -----------------------------------------------------------------------
  Issue                   Does BoW solve it?      Explanation
  ----------------------- ----------------------- -----------------------
  Representing a whole    Yes                     Aggregates token counts
  document                                        into one vector

  Capturing word          Yes                     Records how often each
  frequency                                       term occurs

  Artificial ordering of  Avoided                 Vocabulary positions
  categories                                      are features, not
                                                  meaningful ranks

  High dimensionality     No                      Usually one feature per
                                                  vocabulary term

  Sparsity                No                      Most terms are absent
                                                  from any one document

  Semantic meaning        No                      Does not learn
                                                  relationships between
                                                  words

  Word order              No                      Basic BoW ignores token
                                                  order
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 7. Limitations of Bag of Words

### 7.1 Ignores word order

Consider:

``` text
Sentence 1: Dog bites man.
Sentence 2: Man bites dog.
```

Both sentences contain the same words.

  Sentence          Dog   bites   man
  --------------- ----- ------- -----
  Dog bites man       1       1     1
  Man bites dog       1       1     1

Their BoW vectors are identical, even though their meanings differ.

### 7.2 Cannot understand semantics or context

Consider:

``` text
Document 1: I love this movie.
Document 2: I like this movie.
```

BoW treats `love` and `like` as separate vocabulary terms. It does not
inherently know that they have similar meanings.

It also does not naturally understand synonyms, antonyms, or context.

### 7.3 High dimensionality

If the corpus contains 50,000 unique terms, each document is represented
using 50,000 features.

Even a short document may therefore have a very large vector.

### 7.4 Sparse representation

A document usually contains only a small fraction of all terms in the
corpus. Most positions in its BoW vector are consequently zero.

Example:

``` python
[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
```

This is a sparse vector.

**Note:** Scikit-learn's `CountVectorizer` returns a sparse matrix by
default, which avoids storing every zero explicitly.

### 7.5 Common words may dominate

Words that occur frequently across many documents can receive large
counts, even if they are not useful for distinguishing one document from
another. TF-IDF addresses this issue by weighting terms according to
their importance across the corpus.

------------------------------------------------------------------------

## 8. N-Grams: Capturing some word-order information

Basic BoW ignores word order. **N-Grams** extend the vocabulary by
considering consecutive groups of tokens.

An N-Gram is a sequence of `N` consecutive tokens.

Consider:

``` text
I love Machine Learning
```

### Unigram (N = 1)

Individual tokens:

``` text
I
love
Machine
Learning
```

### Bigram (N = 2)

Groups of two consecutive tokens:

``` text
I love
love Machine
Machine Learning
```

### Trigram (N = 3)

Groups of three consecutive tokens:

``` text
I love Machine
love Machine Learning
```

  Type      Meaning                Example
  --------- ---------------------- --------------------
  Unigram   1 token                `Machine`
  Bigram    2 consecutive tokens   `Machine Learning`
  Trigram   3 consecutive tokens   `I love Machine`

N-Grams help preserve **local word-order patterns**. They do not fully
understand meaning or long-distance context.

For example:

``` text
I love this movie.
I hate this movie.
```

Unigrams identify individual words, while bigrams can capture phrases
such as `love this` and `hate this`.

### Important distinction

N-Grams can be used as features within a BoW approach. A document can be
represented by the frequencies of unigrams, bigrams, or higher-order
N-Grams.

------------------------------------------------------------------------

## 9. BoW vs N-Grams vs TF-IDF vs Word Embeddings

  -----------------------------------------------------------------------
  Technique               What it represents      Main purpose
  ----------------------- ----------------------- -----------------------
  OHE                     Individual              Binary category
                          categories/tokens       representation

  BoW                     Document-level word     Represent documents
                          counts                  using frequencies

  N-Grams                 Consecutive token       Capture local
                          sequences               word-order patterns

  TF-IDF                  Weighted document-term  Reduce the influence of
                          features                corpus-wide common
                                                  terms

  Word Embeddings         Dense numerical vectors Capture semantic and
                                                  syntactic relationships
  -----------------------------------------------------------------------

**Important:** N-Grams and TF-IDF are not necessarily alternatives to
BoW. N-Grams can expand the features used by BoW, and TF-IDF can replace
raw counts with weighted values.

------------------------------------------------------------------------

## 10. Implement BoW using Python

We can implement BoW using `CountVectorizer` from scikit-learn.

### Step 1: Install the library

``` bash
pip install scikit-learn
```

### Step 2: Import CountVectorizer

``` python
from sklearn.feature_extraction.text import CountVectorizer
```

### Step 3: Create documents

``` python
documents = [
    "I love Python",
    "I love Machine Learning",
    "Python is amazing"
]
```

### Step 4: Create the vectorizer

``` python
cv = CountVectorizer()
```

### Step 5: Fit and transform

``` python
bow_matrix = cv.fit_transform(documents)
```

-   `fit()` learns the vocabulary from the supplied documents.
-   `transform()` converts documents into word-count vectors.
-   `fit_transform()` performs both operations together.

### Step 6: Display the vocabulary

``` python
print(cv.get_feature_names_out())
```

Output:

``` text
['amazing' 'is' 'learning' 'love' 'machine' 'python']
```

**Why is `I` missing?**

By default, `CountVectorizer`: - Converts text to lowercase. - Uses a
default token pattern that ignores tokens containing fewer than two
characters.

Therefore, the single-character token `I` is excluded.

### Step 7: Display the BoW matrix

``` python
print(bow_matrix.toarray())
```

Output:

``` text
[[0 0 0 1 0 1]
 [0 0 1 1 1 0]
 [1 1 0 0 0 1]]
```

The columns correspond to:

``` text
['amazing', 'is', 'learning', 'love', 'machine', 'python']
```

  Document     amazing   is   learning   love   machine   python
  ---------- --------- ---- ---------- ------ --------- --------
  D1                 0    0          0      1         0        1
  D2                 0    0          1      1         1        0
  D3                 1    1          0      0         0        1

**Practical tip:** `CountVectorizer` returns a sparse matrix by default.
Use `.toarray()` only for small examples because converting a large
sparse matrix into a dense array can consume substantial memory.

------------------------------------------------------------------------

## 11. Where is BoW used?

  -----------------------------------------------------------------------
  Application                         How BoW helps
  ----------------------------------- -----------------------------------
  Spam detection                      Identifies word patterns associated
                                      with spam

  Sentiment analysis                  Represents words in positive and
                                      negative reviews

  Document classification             Provides frequency-based features
                                      for classifying documents

  News categorization                 Helps distinguish articles using
                                      their vocabulary

  Topic classification                Helps identify recurring vocabulary
                                      in documents
  -----------------------------------------------------------------------

For example, a spam classifier may learn that words such as `free`,
`winner`, `prize`, `claim`, and `offer` frequently occur in spam emails.

BoW provides numerical features that a classifier can learn from.

------------------------------------------------------------------------

## 12. The three things you should write in your notes

### 1. Bag of Words (BoW)

A text-vectorization technique that represents an entire document using
the frequency or presence of words from a predefined vocabulary,
ignoring word order and grammar.

### 2. How does BoW work?

**Corpus → Tokenization → Vocabulary Creation → Word Frequency Counting
→ Document Vectors → BoW Matrix.**

### 3. BoW vs OHE

-   **OHE:** Represents individual tokens using binary vectors.
-   **BoW:** Combines token occurrences into a document-level frequency
    vector.
-   **BoW captures word frequency but does not understand semantic
    meaning or word order.**

### Bonus: Limitations of BoW

-   High dimensionality.
-   Sparse representation.
-   Ignores word order.
-   Cannot understand semantic relationships.
-   Common words can dominate raw counts.

### Quick revision table

  Term              Remember it as
  ----------------- --------------------------------
  BoW               Bag of words
  Vocabulary        Unique words/tokens
  Frequency         Number of occurrences
  Unigram           One token
  Bigram            Two consecutive tokens
  Trigram           Three consecutive tokens
  Sparse Matrix     Matrix containing mostly zeros
  CountVectorizer   Python tool for BoW

------------------------------------------------------------------------

## 13. Knowledge Check

1.  What does Bag of Words represent?
2.  What does the "Bag" in Bag of Words mean?
3.  If `Python` appears three times in a document, what is its BoW
    count?
4.  Does BoW solve the high-dimensionality problem?
5.  What does a Bigram represent?
6.  Which technique assigns importance weights based on term frequency
    and document frequency?

### Answers

1.  An entire document using word frequencies or presence.
2.  Word order is ignored.
3.  3.  
4.  No. BoW still generally uses one feature per vocabulary term.
5.  Two consecutive tokens.
6.  TF-IDF.

------------------------------------------------------------------------

## 14. Learning Progression

1.  Tokenization
2.  One-Hot Encoding (OHE)
3.  Bag of Words (BoW)
4.  **TF-IDF (Next topic)**
5.  N-Grams (deeper understanding)
6.  Word Embeddings (Word2Vec, GloVe, FastText)

**Recommended next topic: TF-IDF (Term Frequency--Inverse Document
Frequency).**

It builds upon BoW and explains why counting words alone isn't always
enough, why frequently occurring words aren't necessarily important, and
how we can assign more meaningful weights to words.
