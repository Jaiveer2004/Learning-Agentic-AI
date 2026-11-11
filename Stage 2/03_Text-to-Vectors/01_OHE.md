# One-Hot Encoding (OHE)

One-hot encoding converts a categorical value into a binary vector. Exactly one
position is `1` (the position assigned to that category) and every other
position is `0`.

In NLP, it can represent a word, a token, or an entire document as a vector so
that a machine-learning algorithm can work with numerical input.

## Why do we need it?

Machine-learning algorithms cannot directly process text such as:

```text
The food is good
```

OHE converts text into numbers without assigning an artificial order to words.
For example, assigning `food = 1`, `pizza = 2`, and `burger = 3` would wrongly
suggest that `burger` is greater than `pizza`. One-hot vectors do not introduce
this false ordering.

## Vocabulary and vector representation

The **vocabulary** is the set of unique words in the corpus. Suppose the
corpus is:

| Document | Text | Output |
|---|---|---:|
| D1 | `The food is good` | 1 |
| D2 | `The food is bad` | 0 |
| D3 | `Pizza is amazing` | 1 |
| D4 | `Burger is bad` | 0 |

After tokenization and normalization, the vocabulary might be:

```text
["the", "food", "is", "good", "bad", "pizza", "amazing"]
```

The vocabulary has size 7, so each word is represented by a vector of length
7:

```text
the     -> [1, 0, 0, 0, 0, 0, 0]
food    -> [0, 1, 0, 0, 0, 0, 0]
good    -> [0, 0, 0, 1, 0, 0, 0]
```

The position of a `1` is determined by the vocabulary index. The index must be
kept consistent between training and inference.

## Document-level OHE

A document can be represented by combining the one-hot vectors of its words.
For example, with the vocabulary above:

```text
"The food is good" -> [1, 1, 1, 1, 0, 0, 0]
"Pizza is amazing" -> [0, 0, 1, 0, 0, 1, 1]
```

This produces a document-term matrix whose shape is:

```text
(number of documents, vocabulary size)
```

The exact representation depends on the chosen convention. A binary
document-level representation records whether a word is present (`1`) or
absent (`0`); it does not record how many times the word occurs. For example,
`"good good food"` still has `good = 1` in binary OHE. Count-based
representations such as Bag of Words are more suitable when frequency matters.

## Numerical example

For the vocabulary `["food", "pizza", "burger"]`:

| Word | food | pizza | burger |
|---|---:|---:|---:|
| food | 1 | 0 | 0 |
| pizza | 0 | 1 | 0 |
| burger | 0 | 0 | 1 |

These vectors are orthogonal: the dot product between two different words is
0. Therefore, basic OHE does not express semantic similarity. The words
`good` and `excellent` are just as unrelated numerically as `good` and
`pizza`, even if their meanings are similar.

## Categorical-data example

OHE is also commonly used for ordinary categorical features:

```text
Colour: ["red", "blue", "green"]

red   -> [1, 0, 0]
blue  -> [0, 1, 0]
green -> [0, 0, 1]
```

For a category with `k` possible values, the one-hot vector has `k` columns.
For many categories, one column can be omitted (`drop="first"`) in a linear
model to avoid perfect multicollinearity. Do not do this automatically when
you need the complete, directly interpretable encoding.

## Python implementation

### Manual implementation

```python
vocabulary = {"food": 0, "pizza": 1, "burger": 2}

def one_hot(word: str) -> list[int]:
    vector = [0] * len(vocabulary)
    if word in vocabulary:
        vector[vocabulary[word]] = 1
    return vector

one_hot("pizza")  # [0, 1, 0]
```

For production code, an unknown word should be handled deliberately. It can be
mapped to an `<UNK>` token, ignored, or rejected with a clear error; silently
returning an all-zero vector can hide data-quality problems.

### Scikit-learn

For categorical values:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = encoder.fit_transform([["red"], ["blue"], ["green"]])

print(encoder.categories_)
print(encoded)
```

`handle_unknown="ignore"` makes an unseen category all zeros at transform time.
The encoder should be fitted on the training data only and reused for
validation, test, and production data:

```python
encoder.fit(train_categories)
train_encoded = encoder.transform(train_categories)
test_encoded = encoder.transform(test_categories)
```

For a text corpus, `CountVectorizer(binary=True)` is usually more convenient
than manually creating one vector per token:

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["the food is good", "the food is bad", "pizza is amazing"]
vectorizer = CountVectorizer(binary=True)
matrix = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())
print(matrix.toarray())
```

The result is a sparse document-term matrix. `binary=True` records presence or
absence; without it, the matrix stores token counts and is a Bag-of-Words
representation rather than strict binary OHE.

## Advantages

1. **Simple and intuitive:** each dimension has a clear category/word meaning.
2. **Easy to implement:** it is available in libraries such as
   `OneHotEncoder` and `CountVectorizer`.
3. **Fixed-size input:** once the vocabulary is fixed, every input has the same
   number of features and can be passed to an ML algorithm.
4. **No artificial ordinal relationship:** categories are not treated as
   numerically ordered.
5. **Useful baseline:** it works well for small vocabularies and simple
   classification or regression problems.

## Disadvantages and edge cases

1. **Sparse, high-dimensional vectors:** a large vocabulary creates mostly-zero
   matrices, increasing memory and computation cost. Keep these matrices sparse
   where possible.
2. **Vocabulary-size dependency:** adding words increases the number of
   features. A vocabulary of 100,000 words creates 100,000 dimensions.
3. **No semantic meaning:** similar words do not receive similar vectors.
4. **No word-order information:** `"dog bites man"` and `"man bites dog"`
   contain the same words and can receive the same binary representation.
5. **Out-of-vocabulary (OOV) words:** a word absent from the training
   vocabulary has no known column. Use an `<UNK>` token or an explicit unknown
   handling strategy.
6. **Possible overfitting:** many features and relatively few training
   examples can make a model memorize the training data.
7. **Repeated words are collapsed in binary form:** frequency information is
   lost. Use counts or TF-IDF when frequency is important.

## OHE compared with other text representations

| Representation | Main idea | Semantic similarity | Word order |
|---|---|---:|---:|
| One-hot encoding | One active position per category/token | No | No |
| Bag of Words | Token counts or binary presence | No | No |
| TF-IDF | Frequency weighted by document rarity | No | No |
| Word2Vec / other embeddings | Dense learned vectors | Yes, approximately | Limited/context-dependent |

OHE is therefore a useful introductory and baseline technique, but large NLP
systems generally use TF-IDF, embeddings, or contextual language-model
representations when sparsity and meaning become important.