# Bag of Words (BoW)

Bag of Words is a text-vectorization technique that represents a document by
the words it contains. It creates a vocabulary from a corpus and converts each
document into a fixed-length vector based on the words in that vocabulary.

The word **bag** is important: the method usually ignores grammar and word
order. A document is treated as an unordered collection of tokens.

## Why do we need it?

Machine-learning algorithms require numerical input, but raw text is made up of
words. BoW provides a simple way to transform text into a document-term matrix
that can be used for classification, clustering, search, or other ML tasks.

For example:

```text
S1: He is a good boy
S2: She is a good girl
S3: Boy and girl are good
```

After lowercasing and removing stopwords such as `he`, `she`, `is`, `a`, `and`,
and `are`, the remaining sentences are:

```text
S1: good boy
S2: good girl
S3: boy girl good
```

## Vocabulary and frequency

The vocabulary is the set of unique tokens in the corpus. For the example
above:

```text
Vocabulary: ["good", "boy", "girl"]
```

The frequency of each vocabulary word is:

| Word | Frequency |
|---|---:|
| good | 3 |
| boy | 2 |
| girl | 2 |

Using the column order `[good, boy, girl]`, the count-based BoW matrix is:

| Sentence | good | boy | girl | Output |
|---|---:|---:|---:|---:|
| S1: `good boy` | 1 | 1 | 0 | 1 |
| S2: `good girl` | 1 | 0 | 1 | 1 |
| S3: `boy girl good` | 1 | 1 | 1 | 1 |

Each sentence is therefore represented as a vector:

```text
S1 -> [1, 1, 0]
S2 -> [1, 0, 1]
S3 -> [1, 1, 1]
```

The shape of the document-term matrix is:

```text
(number of documents, vocabulary size)
```

The vocabulary order must remain fixed after fitting. Otherwise, a value in the
same column could refer to a different word during inference.

## How BoW is created

1. **Collect a corpus:** gather the documents used to build the representation.
2. **Tokenize:** split each document into words or tokens.
3. **Normalize:** commonly lowercase text and normalize punctuation.
4. **Remove stopwords:** optionally remove very common words such as `the`,
   `is`, and `a`.
5. **Build the vocabulary:** assign one column to each unique token.
6. **Count tokens:** fill each document's row with its token frequencies.

Stemming or lemmatization may also be applied before constructing the
vocabulary. For example, `connect`, `connected`, and `connecting` may be
normalized to a common form, depending on the preprocessing goal.

## Types of Bag of Words

### 1. Count-based BoW

Each cell stores the number of times a word appears in the document:

```text
Document: "good good food"
Vocabulary: ["good", "food", "bad"]
Vector:    [2,      1,      0]
```

This preserves frequency information. However, frequent words can dominate the
vector even when they are not useful for distinguishing documents.

### 2. Binary BoW

Each cell stores only whether a word appears:

```text
Document: "good good food"
Vocabulary: ["good", "food", "bad"]
Vector:    [1,      1,      0]
```

Binary BoW is sometimes described as a binary presence/absence version of BoW.
It is useful when repeated occurrences should not receive extra weight.

### 3. N-gram BoW

An n-gram is a contiguous sequence of `n` tokens:

- **Unigram:** one token, for example `food`
- **Bigram:** two consecutive tokens, for example `food good`
- **Trigram:** three consecutive tokens, for example `the food is`

For the sentences:

```text
S1: The food is good
S2: The food is not good
```

Useful bigrams include:

```text
S1: the food, food is, is good
S2: the food, food is, is not, not good
```

With the feature order:

```text
["food", "not", "good", "food good", "food not", "not good"]
```

the sentences can be represented as:

```text
S1 -> [1, 0, 1, 1, 0, 0]
S2 -> [1, 1, 1, 0, 1, 1]
```

N-grams preserve limited local order and are especially useful for phrases such
as `not good`, `machine learning`, or `New York`. Increasing `n` increases the
number of features and can make the matrix much sparser.

In scikit-learn, `(1, 1)` means unigrams, `(1, 2)` means unigrams plus
bigrams, and `(1, 3)` means unigrams, bigrams, and trigrams.

## Python implementation

### Manual count-based BoW

```python
from collections import Counter

documents = ["good boy", "good girl", "boy girl good"]
vocabulary = sorted(set(" ".join(documents).split()))

bow = []
for document in documents:
    counts = Counter(document.split())
    bow.append([counts[word] for word in vocabulary])

print(vocabulary)  # ['boy', 'girl', 'good']
print(bow)         # [[1, 0, 1], [0, 1, 1], [1, 1, 1]]
```

Sorting is only one possible ordering. The important requirement is that the
same vocabulary-to-column mapping is reused consistently.

### Scikit-learn `CountVectorizer`

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "He is a good boy",
    "She is a good girl",
    "Boy and girl are good",
]

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english",
)

matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(matrix.toarray())
```

`CountVectorizer` returns a sparse matrix by default, which is preferable for
large vocabularies because most document-word combinations are zero.

To create binary BoW:

```python
binary_vectorizer = CountVectorizer(binary=True)
binary_matrix = binary_vectorizer.fit_transform(documents)
```

To include unigrams and bigrams:

```python
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
ngram_matrix = ngram_vectorizer.fit_transform(documents)
```

Fit the vectorizer on training data only, then reuse it:

```python
vectorizer.fit(train_documents)
train_matrix = vectorizer.transform(train_documents)
test_matrix = vectorizer.transform(test_documents)
```

This prevents test-set vocabulary information from leaking into training.
Words not seen during fitting are ignored by `CountVectorizer` unless an
explicit vocabulary or unknown-token strategy is used.

## Similarity between BoW vectors

BoW vectors can be compared using measures such as cosine similarity. For
vectors `x` and `y`:

```text
cosine_similarity(x, y) = (x · y) / (||x|| ||y||)
```

Documents with many shared words often have a higher similarity. This is
lexical similarity, not true semantic understanding: `good movie` and
`excellent film` may be semantically similar but share no exact tokens.

## Advantages

1. **Simple and intuitive:** each column corresponds to a vocabulary word.
2. **Easy to implement:** `CountVectorizer` and similar utilities provide a
   practical implementation.
3. **Fixed-size input:** every document is converted to the same number of
   features after the vocabulary is fixed.
4. **Works well as a baseline:** it is often effective for small and
   medium-sized text-classification tasks.
5. **Frequency can be retained:** count-based BoW captures repeated words,
   unlike binary OHE.
6. **N-grams capture short phrases:** bigrams can distinguish expressions such
   as `good` and `not good`.

## Disadvantages and edge cases

1. **Sparse, high-dimensional matrix:** a large vocabulary produces mostly-zero
   vectors and can increase memory use and computation cost.
2. **No deep semantics:** words with similar meanings are unrelated unless they
   share tokens or appear in learned downstream patterns.
3. **Limited word-order information:** unigram BoW treats `dog bites man` and
   `man bites dog` identically. N-grams preserve only local order.
4. **Vocabulary-size dependency:** adding words or larger n-grams adds columns.
5. **Out-of-vocabulary (OOV) words:** tokens absent from the fitted vocabulary
   cannot contribute to a transformed vector.
6. **Stopword and preprocessing sensitivity:** removing a word such as `not`
   can destroy important sentiment information.
7. **Document-length bias:** count vectors from longer documents can have
   larger values simply because they contain more words. Binary BoW or TF-IDF
   can reduce this effect.
8. **Potential overfitting:** very large vocabularies and n-gram ranges can
   memorize training-specific phrases.

## BoW compared with other representations

| Representation | Main idea | Uses frequency | Captures semantics |
|---|---|---:|---:|
| One-hot encoding | One vector per token/category | No | No |
| Binary BoW | Word presence in each document | No | No |
| Count BoW | Word frequency in each document | Yes | No |
| TF-IDF | Frequency weighted by document rarity | Yes, weighted | No, but highlights informative words |
| Word embeddings | Dense learned vector per word | Indirectly | Approximately |

BoW is a strong first representation for understanding text vectorization. For
larger corpora or tasks that need semantic relationships, TF-IDF, Word2Vec, or
contextual embeddings are usually more appropriate.