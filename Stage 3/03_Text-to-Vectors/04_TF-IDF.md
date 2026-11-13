# 04 --- TF-IDF (Term Frequency--Inverse Document Frequency)

## 1. Why do we need TF-IDF?

Machine Learning models need numerical representations of text. Earlier,
we learned:

-   **One-Hot Encoding (OHE):** represents individual categories or
    tokens using binary vectors.
-   **Bag of Words (BoW):** represents documents using word counts or
    presence.
-   **N-Grams:** extends word-based features to consecutive sequences,
    capturing some local word order.

A limitation of raw BoW counts is that every occurrence contributes to a
word's count, even when that word appears in almost every document. Such
common words may be less useful for distinguishing one document from
another.

For example, in product reviews, words such as `this` and `product` may
occur in many reviews, while words such as `excellent`, `damaged`, or
`overpriced` may be more distinctive.

**TF-IDF addresses this by weighting terms according to their frequency
in a document and how widely they occur across the corpus.**

## 2. What is TF-IDF?

**TF-IDF stands for Term Frequency--Inverse Document Frequency.**

It is a text-vectorization and term-weighting technique that estimates
how important a term is to a particular document relative to a
collection (corpus) of documents.

It combines two components:

1.  **TF (Term Frequency):** How often does the term occur in this
    document?
2.  **IDF (Inverse Document Frequency):** In how many documents across
    the corpus does the term occur?

The basic relationship is:

\[ `\boxed{TF\text{-}IDF(t,d)=TF(t,d)\times IDF(t)}`{=tex} \]

A term tends to receive a high TF-IDF weight when it occurs frequently
in a particular document but appears in relatively few documents
overall.

A term appearing in many documents receives a lower IDF weight.

## 3. Term Frequency (TF)

Term Frequency measures how frequently a term appears in a particular
document.

A common normalized-frequency formula is:

\[
TF(t,d)=`\frac{\text{Number of occurrences of term }t\text{ in document }d}`{=tex}
{`\text{Total number of terms in document }`{=tex}d} \]

Where:

-   \(t\) = term (word or feature)
-   \(d\) = document

### Example

Document:

> I love Python. Python is amazing. I love programming.

Ignoring punctuation, the document contains 9 tokens.

  Term            Count            TF
  ------------- ------- -------------
  I                   2   2/9 = 0.222
  love                2   2/9 = 0.222
  Python              2   2/9 = 0.222
  is                  1   1/9 = 0.111
  amazing             1   1/9 = 0.111
  programming         1   1/9 = 0.111

A term that occurs more frequently in a document receives a higher TF
under this formula.

**Note:** TF has multiple variants. Implementations may use raw counts,
normalized counts, or logarithmic scaling. Always check which formula a
tool uses.

## 4. Inverse Document Frequency (IDF)

IDF measures how widely a term is distributed across the corpus. It
gives less weight to terms that occur in many documents and more weight
to terms that occur in fewer documents.

The basic formula is:

\[ `\boxed{IDF(t)=\log\left(\frac{N}{DF(t)}\right)}`{=tex} \]

Where:

-   \(N\) = total number of documents in the corpus.
-   (DF(t)) = document frequency: number of documents containing term
    (t).
-   (`\log`{=tex}) = logarithm (often the natural logarithm).

### Example

Suppose a corpus contains five documents:

  Document   Content
  ---------- ------------------------------
  D1         I love Python programming
  D2         Python is amazing
  D3         I love machine learning
  D4         Machine learning is powerful
  D5         Python programming is fun

Here, (N=5).

  Term       Documents containing the term     DF
  ---------- ------------------------------- ----
  Python     D1, D2, D5                         3
  learning   D3, D4                             2
  I          D1, D3                             2

For Python:

\[ IDF(Python)=`\log`{=tex}(5/3)`\approx0.511`{=tex} \]

For learning:

\[ IDF(learning)=`\log`{=tex}(5/2)`\approx0.916`{=tex} \]

For I:

\[ IDF(I)=`\log`{=tex}(5/2)`\approx0.916`{=tex} \]

Python occurs in more documents, so its IDF is lower.

### Important observations

-   Higher document frequency (DF) → lower IDF.
-   Lower document frequency (DF) → higher IDF.
-   If a term appears in every document, its basic IDF is zero:

\[ IDF(t)=`\log`{=tex}(N/N)=`\log`{=tex}(1)=0 \]

This inverse relationship explains the word **Inverse** in Inverse
Document Frequency.

## 5. Combining TF and IDF

The basic TF-IDF formula is:

\[ `\boxed{TFIDF(t,d)=TF(t,d)\times IDF(t)}`{=tex} \]

### Worked example

Consider D1:

> Python is amazing. Python is powerful.

Ignoring punctuation, there are 6 tokens. Python occurs twice.

**Step 1 --- TF:**

\[ TF(Python,D1)=2/6`\approx0.333`{=tex} \]

**Step 2 --- IDF:**

Suppose the corpus contains 5 documents and Python appears in 3:

\[ IDF(Python)=`\log`{=tex}(5/3)`\approx0.511`{=tex} \]

**Step 3 --- TF-IDF:**

\[ TFIDF(Python,D1)=0.333`\times0.511`{=tex}`\approx0.170`{=tex} \]

This is the weight using the basic normalized-TF and unsmoothed-IDF
formulas.

### Interpretation

TF measures a term's frequency within one document. IDF adjusts that
frequency based on how common the term is across the corpus. Their
product gives the term's weight for that document.

A high score often indicates a term that is relatively distinctive to
the document. It does **not** necessarily mean the term is semantically
important or meaningful to a human.

## 6. BoW vs TF-IDF

BoW and TF-IDF can use the same vocabulary and produce document-level
feature vectors, but their values mean different things.

  -----------------------------------------------------------------------
  Aspect                  Bag of Words            TF-IDF
  ----------------------- ----------------------- -----------------------
  Main idea               Counts term occurrences Weights terms using TF
                          (or records presence)   and IDF

  Common terms            Can have high counts    Usually receive lower
                                                  IDF weights

  Rare terms              No special weighting by Usually receive higher
                          rarity                  IDF

  Output                  Count/presence vector   Weighted numerical
                                                  vector

  Word order              Ignored unless using    Ignored unless using
                          N-Gram features         N-Gram features

  Semantic understanding  No                      No
  -----------------------------------------------------------------------

**Main improvement:** TF-IDF reduces the influence of terms that occur
across many documents and gives more weight to terms that are relatively
distinctive to individual documents.

### What TF-IDF does not solve

-   It does not understand the meaning of words.
-   It does not inherently recognize synonyms (e.g. `good` and
    `excellent`).
-   It does not inherently understand context (e.g. `bank` by a river
    vs. a financial bank).
-   It does not preserve word order by itself.
-   It can still produce high-dimensional, sparse vectors.

## 7. TF-IDF with N-Grams

TF-IDF is not restricted to individual words. It can assign weights to
N-Gram features too.

For the sentence:

> I love machine learning

Possible features include:

**Unigrams:** - I - love - machine - learning

**Bigrams:** - I love - love machine - machine learning

**Trigrams:** - I love machine - love machine learning

A TF-IDF vectorizer can be configured to use unigrams, bigrams,
trigrams, or a combination.

For example, `ngram_range=(1, 2)` includes unigrams and bigrams.

N-Gram features capture local sequences, but they do not provide full
semantic understanding or long-range context.

## 8. Implementing TF-IDF in Python

Scikit-learn provides `TfidfVectorizer`, which learns a vocabulary and
IDF weights and transforms documents into TF-IDF feature vectors.

### Install

``` bash
pip install scikit-learn
```

### Basic example

``` python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love Python programming",
    "Python is amazing",
    "I love machine learning",
    "Machine learning is powerful",
    "Python programming is fun"
]

# Create the vectorizer
vectorizer = TfidfVectorizer()

# Learn vocabulary and IDF weights, then transform documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Display the learned feature names
print(vectorizer.get_feature_names_out())

# Display the matrix (suitable for this small example)
print(tfidf_matrix.toarray())
```

### Important methods

  -----------------------------------------------------------------------
  Method                              Purpose
  ----------------------------------- -----------------------------------
  `fit()`                             Learns the vocabulary and IDF
                                      weights from the supplied training
                                      corpus

  `transform()`                       Converts documents using the
                                      already-fitted vocabulary and IDF
                                      weights

  `fit_transform()`                   Performs fitting and transformation
                                      together

  `get_feature_names_out()`           Returns the learned feature names

  `toarray()`                         Converts a sparse matrix into a
                                      dense NumPy array
  -----------------------------------------------------------------------

**Practical ML rule:** Fit the vectorizer on the training documents
only, then use that same fitted vectorizer to transform validation and
test documents. Do not fit a new vectorizer on test data; doing so can
cause data leakage and inconsistent feature spaces.

### Using N-Grams

``` python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(documents)
```

`ngram_range=(1, 2)` includes:

-   Unigrams (one token)
-   Bigrams (two consecutive tokens)

### Useful parameters

  -----------------------------------------------------------------------
  Parameter                           Purpose
  ----------------------------------- -----------------------------------
  `max_features`                      Limits the number of selected
                                      features

  `min_df`                            Ignores terms occurring in fewer
                                      than the specified number/fraction
                                      of documents

  `max_df`                            Ignores terms occurring in more
                                      than the specified number/fraction
                                      of documents

  `stop_words`                        Removes specified stop words
                                      (e.g. `'english'`)

  `sublinear_tf`                      Applies logarithmic scaling to term
                                      frequency

  `ngram_range`                       Controls the range of N-Gram
                                      features

  `smooth_idf`                        Adds smoothing to the IDF
                                      calculation

  `norm`                              Controls vector normalization
                                      (`'l1'`, `'l2'`, or `None`)
  -----------------------------------------------------------------------

## 9. Scikit-learn's TF-IDF calculation

An important implementation detail: Scikit-learn's `TfidfVectorizer`
does not use exactly the basic formulas shown earlier.

By default, it uses smoothed IDF:

\[ `\boxed{IDF(t)=\log\left(\frac{1+N}{1+DF(t)}\right)+1}`{=tex} \]

It also uses L2 normalization by default.

Consequently, the final values from `TfidfVectorizer` may differ from
hand calculations using normalized term frequency and the basic
unsmoothed IDF formula.

Useful settings include:

-   `smooth_idf=True` (default): uses smoothed IDF.
-   `norm='l2'` (default): normalizes each document vector to unit L2
    length, when nonzero.
-   `use_idf=True` (default): applies inverse document frequency.

## 10. Applications of TF-IDF

1.  **Information retrieval and search:** Helps rank documents based on
    the importance of query terms.
2.  **Spam detection:** Provides weighted text features to a
    classification model.
3.  **Sentiment analysis:** Represents reviews as weighted features for
    a classifier.
4.  **Document similarity:** TF-IDF vectors can be compared using cosine
    similarity to identify similar documents.
5.  **Document classification:** Supplies numerical features to
    traditional ML classifiers.

TF-IDF is a feature representation method, not a classifier by itself.
It is often combined with algorithms such as Logistic Regression, Naive
Bayes, or Support Vector Machines.

## 11. Comparison: OHE, BoW, N-Grams and TF-IDF

  -----------------------------------------------------------------------
  Technique               What it does            Main purpose
  ----------------------- ----------------------- -----------------------
  One-Hot Encoding        Represents individual   Numerical category
                          categories/tokens with  representation
                          binary vectors          

  Bag of Words            Represents documents    Document-level
                          using word counts or    frequency features
                          presence                

  N-Grams                 Creates features from   Captures local word
                          consecutive token       order/phrases
                          sequences               

  TF-IDF                  Weights terms using     Highlights relatively
                          within-document         distinctive terms
                          frequency and           
                          corpus-wide document    
                          frequency               
  -----------------------------------------------------------------------

These techniques are related but not mutually exclusive. For example,
TF-IDF can weight unigram and N-Gram features.

## 12. The three things you should write in your notes

**1. TF (Term Frequency):** Measures how frequently a term appears in a
particular document.

**2. IDF (Inverse Document Frequency):** Measures how rare a term is
across the corpus. Common terms get lower IDF; terms found in fewer
documents get higher IDF.

**3. TF-IDF = TF × IDF:** Gives higher weight to terms that occur
frequently in a document but are relatively uncommon across the corpus.

### Quick revision

-   **BoW:** Tells us how frequently a word appears in a document.
-   **TF-IDF:** Weights that word according to its frequency in the
    document and rarity across the corpus.
-   **High TF-IDF:** Often indicates a term that is distinctive to a
    document.
-   **Low IDF:** Indicates a term found in many documents.
-   **N-Gram TF-IDF:** Can weight phrases as well as individual words.

> **Key takeaway: BoW tells us how frequently a word appears. TF-IDF
> tells us how important that word is to a document compared with the
> entire corpus.**

## 13. Knowledge check

1.  A word appears in every document in a corpus. What happens to its
    IDF using the basic formula?
    -   A. It becomes very high
    -   B. It becomes zero
    -   C. It becomes equal to TF
    -   D. It becomes negative
2.  A word appears frequently in one document but in very few documents
    overall. What would you generally expect?
    -   A. High TF-IDF
    -   B. Zero TF-IDF
    -   C. Always low TF
    -   D. No vocabulary entry
3.  Which limitation of BoW does TF-IDF primarily address?
    -   A. Loss of word order
    -   B. Lack of semantic understanding
    -   C. Overemphasis on terms common across documents
    -   D. Inability to tokenize text
4.  Can TF-IDF use bigrams as features?
    -   A. No, only individual words
    -   B. Yes, through `ngram_range`
    -   C. Only with Word2Vec
    -   D. Only with deep learning

**Answers:** 1-B, 2-A, 3-C, 4-B.

## 14. What to learn next

A natural next topic is **Word Embeddings**, including Word2Vec, GloVe,
and FastText. These move beyond simple counts and weights toward dense
vector representations that can capture semantic relationships between
words.
