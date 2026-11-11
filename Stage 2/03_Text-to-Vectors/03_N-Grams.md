# 03 --- N-Grams in NLP

## 1. What are N-Grams?

**N-Grams are continuous sequences of *N* consecutive items (usually
words or tokens) extracted from text.**

-   **N = 1:** Unigram --- one token at a time.
-   **N = 2:** Bigram --- two consecutive tokens.
-   **N = 3:** Trigram --- three consecutive tokens.
-   **N = 4:** Four-gram --- four consecutive tokens.

N-Grams are useful because they preserve some **local word-order
information**, unlike ordinary unigram Bag of Words (BoW).

### Example

Sentence:

> I love Machine Learning

  N-Gram type       N-Grams                                          Count
  ----------------- ---------------------------------------------- -------
  Unigram (N=1)     `I`, `love`, `Machine`, `Learning`                   4
  Bigram (N=2)      `I love`, `love Machine`, `Machine Learning`         3
  Trigram (N=3)     `I love Machine`, `love Machine Learning`            2
  Four-gram (N=4)   `I love Machine Learning`                            1

For a sequence containing **L tokens**, the number of N-Grams of a fixed
size N is:

\[ `\text{Number of N-Grams}`{=tex} = L - N + 1 \]

This assumes one uninterrupted token sequence and (N `\leq `{=tex}L).
For example, with 5 tokens, there are 4 bigrams and 3 trigrams.

------------------------------------------------------------------------

## 2. Types of N-Grams

### 2.1 Unigrams

A unigram is one individual token.

Example:

`Natural Language Processing is interesting`

Unigrams:

-   `Natural`
-   `Language`
-   `Processing`
-   `is`
-   `interesting`

Unigrams are the ordinary individual-word features used by a standard
unigram BoW model.

### 2.2 Bigrams

A bigram is a sequence of two consecutive tokens.

Example:

`I love Machine Learning`

Bigrams:

-   `I love`
-   `love Machine`
-   `Machine Learning`

Bigrams can represent common phrases such as `Machine Learning`,
`Credit Card`, and `not good`.

### 2.3 Trigrams

A trigram is a sequence of three consecutive tokens.

Example:

`I love Machine Learning`

Trigrams:

-   `I love Machine`
-   `love Machine Learning`

Trigrams capture longer local phrases, such as `New York City` or
`not very good`.

### 2.4 Higher-order N-Grams

Four-grams, five-grams, and larger N-Grams represent increasingly long
consecutive sequences. They can capture longer phrases, but they tend to
be rarer and increase vocabulary size.

------------------------------------------------------------------------

## 3. How N-Grams improve on ordinary BoW

### 3.1 The limitation of unigram BoW

Ordinary BoW with unigrams counts individual words and ignores their
order.

Consider:

-   `Dog bites man`
-   `Man bites dog`

Both produce the same unigram counts:

  Feature     Dog bites man   Man bites dog
  --------- --------------- ---------------
  `dog`                   1               1
  `bites`                 1               1
  `man`                   1               1

The two sentences have different meanings, but their unigram BoW vectors
are identical.

### 3.2 How bigrams help

Their bigram features differ:

  Sentence          Bigrams
  ----------------- --------------------------
  `Dog bites man`   `dog bites`, `bites man`
  `Man bites dog`   `man bites`, `bites dog`

The bigrams preserve local order and allow a model to distinguish these
sentences.

### 3.3 Negation example

Compare:

-   `I love this movie`
-   `I do not love this movie`

Unigrams record words such as `do` and `not`, but do not explicitly
encode the phrase `not love`.

Bigrams include:

-   First sentence: `i love`, `love this`, `this movie`
-   Second sentence: `i do`, `do not`, `not love`, `love this`,
    `this movie`

The feature `not love` can help a classifier learn a useful negation
pattern.

**Important:** N-Grams capture local sequences; they do not fully
understand meaning, sentiment, or long-distance context. A model still
has to learn how useful each feature is from training examples.

------------------------------------------------------------------------

## 4. N-Grams and Bag of Words work together

N-Grams are not necessarily a replacement for BoW. They are commonly
used as **features within a BoW-style representation**.

A document is represented by counting the occurrences of vocabulary
features. Those features can be:

-   Unigrams only.
-   Bigrams only.
-   Trigrams only.
-   A combination of unigrams, bigrams, and trigrams.

### Example: Unigram BoW

Documents:

1.  `I love Machine Learning`
2.  `I love Python`

Unigram vocabulary:

`i`, `love`, `machine`, `learning`, `python`

  Document     i   love   machine   learning   python
  ---------- --- ------ --------- ---------- --------
  D1           1      1         1          1        0
  D2           1      1         0          0        1

### Example: Unigram + Bigram features

The feature vocabulary can additionally include:

-   `i love`
-   `love machine`
-   `machine learning`
-   `love python`

  ----------------------------------------------------------------------------------------------
  Document         i    love   machine   learning   python  i love      love    machine     love
                                                                     machine   learning   python
  ---------- ------- ------- --------- ---------- -------- ------- --------- ---------- --------
  D1               1       1         1          1        0       1         1          1        0

  D2               1       1         0          0        1       1         0          0        1
  ----------------------------------------------------------------------------------------------

This representation contains both individual-word and phrase features.

### `ngram_range`

In scikit-learn's `CountVectorizer`:

-   `ngram_range=(1, 1)` --- unigrams only.
-   `ngram_range=(2, 2)` --- bigrams only.
-   `ngram_range=(3, 3)` --- trigrams only.
-   `ngram_range=(1, 3)` --- unigrams, bigrams, and trigrams.

------------------------------------------------------------------------

## 5. What problems do N-Grams solve---and what remains?

  -----------------------------------------------------------------------
  Limitation of unigram BoW           Does N-Gram BoW help?
  ----------------------------------- -----------------------------------
  Ignores all word order              **Partially.** It captures order
                                      inside each N-token window.

  Cannot represent phrases as         **Yes.** Phrases such as
  individual features                 `machine learning` become features.

  Cannot distinguish some reordered   **Often.** Different local
  sentences                           sequences produce different
                                      features.

  Cannot understand semantic          **No.** Similar words remain
  similarity                          separate features.

  High dimensionality                 **No.** N-Grams can make the
                                      vocabulary much larger.

  Sparse representation               **No.** Longer sequences are often
                                      rare, increasing sparsity.

  Long-distance context               **No.** A fixed N only captures a
                                      limited local window.
  -----------------------------------------------------------------------

### 5.1 Vocabulary explosion

Adding bigrams and trigrams introduces many additional features. The
vocabulary can become much larger than a unigram-only vocabulary.

### 5.2 Increased sparsity

Most documents contain only a small fraction of all possible N-Gram
features. Consequently, many matrix entries are zero.

### 5.3 Rare sequences and data sparsity

Longer N-Grams occur less frequently. A model may not see a particular
phrase often enough during training to learn a reliable association.

### 5.4 No inherent semantic understanding

For example, `love Python` and `enjoy Python` are different bigrams.
N-Grams do not inherently know that `love` and `enjoy` have related
meanings.

**Practical takeaway:** N-Grams add local sequence information, not
genuine language understanding.

------------------------------------------------------------------------

## 6. Word N-Grams vs Character N-Grams

N-Grams can be built from words/tokens or from characters.

### 6.1 Word N-Grams

Group consecutive words.

Example: `I love Python`

Bigrams:

-   `I love`
-   `love Python`

Useful for phrase detection, text classification, and sentiment
analysis.

### 6.2 Character N-Grams

Group consecutive characters.

Example: `PYTHON`

Character bigrams:

-   `PY`
-   `YT`
-   `TH`
-   `HO`
-   `ON`

Character trigrams:

-   `PYT`
-   `YTH`
-   `THO`
-   `HON`

Character N-Grams can capture overlapping spelling patterns. They are
useful for typo detection, spelling variation, language identification,
and text similarity.

  Word N-Grams                  Character N-Grams
  ----------------------------- ----------------------------------
  Group consecutive words       Group consecutive characters
  Capture word-level phrases    Capture character-level patterns
  Example: `machine learning`   Example: `mac`, `ach`, `chi`

------------------------------------------------------------------------

## 7. N-Grams in statistical language modelling

N-Grams are also used in **statistical language models**, which estimate
the probability of a word sequence or the next word.

For example:

> I want to drink \_\_\_\_\_\_

A language model estimates which next word is likely, such as `water`,
`tea`, or `coffee`.

A statistical N-Gram language model approximates the next-token
probability using a limited number of preceding tokens.

### Unigram model

Uses individual token probabilities:

\[ P(w_n) \]

### Bigram model

Uses the previous one token:

\[ P(w_n `\mid `{=tex}w\_{n-1}) \]

### Trigram model

Uses the previous two tokens:

\[ P(w_n `\mid `{=tex}w\_{n-2}, w\_{n-1}) \]

This is based on the **Markov assumption**: the next token is
approximated using only a limited preceding context.

For example:

-   Bigram: (P(`\text{Learning}`{=tex}
    `\mid `{=tex}`\text{Machine}`{=tex}))
-   Trigram: (P(`\text{Learning}`{=tex}
    `\mid `{=tex}`\text{love Machine}`{=tex}))

### Two uses---do not confuse them

1.  **N-Gram feature extraction:** Represents documents using counts of
    N-Gram features, often with BoW or TF-IDF.
2.  **N-Gram language modelling:** Estimates probabilities of token
    sequences or predicts the next token using a limited context.

Both use consecutive sequences, but their goals are different.

------------------------------------------------------------------------

## 8. Implementing N-Grams in Python

Use scikit-learn's `CountVectorizer`.

### Installation

``` bash
pip install scikit-learn
```

### Sample documents

``` python
documents = [
    "I love Machine Learning",
    "I love Python",
    "Machine Learning is amazing"
]
```

### 8.1 Unigrams

``` python
from sklearn.feature_extraction.text import CountVectorizer

unigram = CountVectorizer(ngram_range=(1, 1))

unigram_matrix = unigram.fit_transform(documents)

print(unigram.get_feature_names_out())
print(unigram_matrix.toarray())
```

### 8.2 Bigrams

``` python
bigram = CountVectorizer(ngram_range=(2, 2))

bigram_matrix = bigram.fit_transform(documents)

print(bigram.get_feature_names_out())
print(bigram_matrix.toarray())
```

### 8.3 Trigrams

``` python
trigram = CountVectorizer(ngram_range=(3, 3))

trigram_matrix = trigram.fit_transform(documents)

print(trigram.get_feature_names_out())
print(trigram_matrix.toarray())
```

### 8.4 Combine unigrams, bigrams, and trigrams

``` python
vectorizer = CountVectorizer(ngram_range=(1, 3))

matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(matrix.toarray())
```

### 8.5 Character N-Grams

``` python
char_vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(2, 3)
)

char_matrix = char_vectorizer.fit_transform(["Python"])

print(char_vectorizer.get_feature_names_out())
```

### Useful `CountVectorizer` parameters

  Parameter        Purpose
  ---------------- --------------------------------------------------
  `ngram_range`    Sets minimum and maximum N
  `analyzer`       Selects `word` or `char` analysis
  `max_features`   Limits the number of features
  `min_df`         Ignores features occurring in too few documents
  `max_df`         Ignores features occurring in too many documents
  `stop_words`     Removes selected stop words
  `binary=True`    Stores presence (1) rather than frequency counts

**Practical note:** `CountVectorizer` produces a sparse matrix by
default. Avoid converting large matrices to dense arrays with
`.toarray()` because that can consume substantial memory.

------------------------------------------------------------------------

## 9. Normal BoW vs N-Gram BoW

  Parameter                Unigram BoW        N-Gram BoW
  ------------------------ ------------------ -----------------------------
  Basic feature            Individual words   Consecutive token sequences
  Word order               Ignored            Partially captured
  Phrase features          No                 Yes
  Frequency counting       Yes                Yes
  Semantic understanding   No                 No
  Vocabulary size          Usually smaller    Can be much larger
  Sparsity                 Often high         Can be higher
  Computational cost       Usually lower      Can be higher

------------------------------------------------------------------------

## 10. The three things you should write in your notes

**1. N-Grams**

N-Grams are continuous sequences of N consecutive words or tokens
extracted from text to capture local word-order information.

**2. Types of N-Grams**

-   Unigram (N=1): One word.
-   Bigram (N=2): Two consecutive words.
-   Trigram (N=3): Three consecutive words.
-   Character N-Gram: Consecutive character sequences.

**3. N-Grams vs Normal BoW**

-   Normal BoW considers individual words and ignores their order.
-   N-Gram BoW considers consecutive word combinations and captures
    local word order.
-   N-Grams improve phrase representation but increase vocabulary size
    and sparsity.

### Quick revision table

  Term               Remember it as
  ------------------ ------------------------------------------
  Unigram            One word
  Bigram             Two consecutive words
  Trigram            Three consecutive words
  Word N-Gram        Groups of consecutive words
  Character N-Gram   Groups of consecutive characters
  N-Gram BoW         BoW with word-sequence features
  Language Model     Predicts probabilities of word sequences

------------------------------------------------------------------------

## 11. Knowledge check

1.  What is a Bigram?
    -   One word
    -   Two consecutive words
    -   Three consecutive words
    -   Two random words
2.  How many Trigrams can be generated from a sentence containing 6
    tokens?
    -   3
    -   4
    -   5
    -   6
3.  Which problem of normal BoW do N-Grams partially solve?
    -   High dimensionality
    -   Sparsity
    -   Loss of local word order
    -   Lack of numerical representation
4.  What does `ngram_range=(1,3)` mean in `CountVectorizer`?
    -   Only trigrams
    -   Unigrams, bigrams, and trigrams
    -   Only unigrams
    -   One to three documents
5.  What is a Character Bigram of `PYTHON`?
    -   `PYTHON`
    -   `PY`, `YT`, `TH`, `HO`, `ON`
    -   `PYT`, `YTH`
    -   `P`, `Y`, `T`, `H`, `O`, `N`
6.  Do N-Grams understand semantic relationships between words?
    -   Yes, completely
    -   No, not inherently
    -   Only trigrams
    -   Only character N-Grams
7.  In a Bigram language model, what context is used to predict the next
    word?
    -   Previous one word
    -   Previous two words
    -   Entire document
    -   No context

### Answers

1.  Two consecutive words.
2.  4 (using (L-N+1=6-3+1)).
3.  Loss of local word order.
4.  Unigrams, bigrams, and trigrams.
5.  `PY`, `YT`, `TH`, `HO`, `ON`.
6.  No, not inherently.
7.  Previous one word.

------------------------------------------------------------------------

## 12. NLP learning progression

1.  Tokenization
2.  One-Hot Encoding (OHE)
3.  Bag of Words (BoW)
4.  N-Grams
5.  **TF-IDF --- Next**
6.  Word Embeddings (Word2Vec, GloVe, FastText)
7.  RNN, LSTM, and GRU
8.  Attention Mechanism and Transformers

### What to learn next: TF-IDF

TF-IDF addresses another limitation of BoW and N-Grams: not every
frequently occurring word is equally important.

It helps assign importance weights to words based on their frequency in
a document and how common they are across the document collection.

> **Final takeaway:** BoW tells us which words appear and how
> frequently. N-Grams additionally tell us which words appear together
> in a particular sequence.
