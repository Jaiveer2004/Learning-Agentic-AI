# One-Hot Encoding (OHE)

## 1. What is One-Hot Encoding?

**One-Hot Encoding (OHE)** is a technique that converts categorical
values or tokens into binary vectors containing only `0`s and `1`s.
Exactly one position is `1`, while all other positions are `0`.

-   **One:** Exactly one position is active (`1`).
-   **Hot:** The active position is switched ON.

### Example

Suppose the vocabulary contains three programming languages:

  Language   One-Hot Vector
  ---------- ----------------
  Python     `[1, 0, 0]`
  Java       `[0, 1, 0]`
  C++        `[0, 0, 1]`

Each category receives a unique vector.

## 2. Why Do We Need OHE?

Machine Learning algorithms perform mathematical operations such as
addition, multiplication, distance calculations, and optimization. They
generally cannot perform these operations directly on raw text labels.

Text vectorization converts text or categories into numerical
representations that models can process.

OHE is one simple way to do this.

### Main reasons

1.  **Numerical representation:** Converts categorical values into
    numbers that ML algorithms can process.
2.  **Avoids artificial ordering:** Does not imply that one category is
    greater than another.
3.  **Unique representation:** Gives every category its own distinct
    vector.
4.  **Supports mathematical operations:** Encoded values can be used as
    model inputs.

### Why not simply assign numbers?

Consider this encoding:

  Language     Label
  ---------- -------
  Python           1
  Java             2
  C++              3

A model might interpret these numbers as ordered or assume meaningful
distances between them. But programming languages do not have this
numerical ordering.

OHE avoids that implication:

  Language   One-Hot Vector
  ---------- ----------------
  Python     `[1, 0, 0]`
  Java       `[0, 1, 0]`
  C++        `[0, 0, 1]`

## 3. OHE in NLP: A Worked Example

Suppose a corpus contains these documents:

``` text
Document 1: I love Python.
Document 2: I love Java.
Document 3: Python is amazing.
```

### Step 1: Create the vocabulary

Extract the unique tokens:

``` python
vocabulary = [
    "I",
    "love",
    "Python",
    "Java",
    "is",
    "amazing"
]
```

Vocabulary size = **6**.

### Step 2: Assign each token a position

    Index Token
  ------- ---------
        0 I
        1 love
        2 Python
        3 Java
        4 is
        5 amazing

### Step 3: Create one-hot vectors

  Token     One-Hot Vector
  --------- ----------------------
  I         `[1, 0, 0, 0, 0, 0]`
  love      `[0, 1, 0, 0, 0, 0]`
  Python    `[0, 0, 1, 0, 0, 0]`
  Java      `[0, 0, 0, 1, 0, 0]`
  is        `[0, 0, 0, 0, 1, 0]`
  amazing   `[0, 0, 0, 0, 0, 1]`

**Important:** The length of every one-hot vector equals the vocabulary
size.

If the vocabulary has 6 unique tokens, each vector has 6 positions. If
it has 10,000 unique tokens, each vector has 10,000 positions.

### Step 4: Represent a sentence

Sentence:

``` text
I love Python
```

Its one-hot representation is a matrix containing one row per token:

$$
\begin{bmatrix}
1&0&0&0&0&0\\
0&1&0&0&0&0\\
0&0&1&0&0&0
\end{bmatrix}
$$

A single token is represented by a vector; a sequence of tokens can
therefore be represented by a matrix.

## 4. Understanding OHE Mathematically

Consider two one-hot vectors:

$$
A = [1, 0, 0]
$$

$$
B = [0, 1, 0]
$$

Their dot product is:

$$
A \cdot B = (1 \times 0) + (0 \times 1) + (0 \times 0) = 0
$$

Different one-hot vectors are orthogonal: their dot product is zero.

This means OHE does not encode semantic similarity between categories.
For example, it does not know that Python and Java are both programming
languages.

## 5. Implementing OHE with Python

Scikit-learn provides `OneHotEncoder` for encoding categorical features.

### Install

``` bash
pip install scikit-learn
```

### Example

``` python
import numpy as np
from sklearn.preprocessing import OneHotEncoder

languages = np.array([
    ["Python"],
    ["Java"],
    ["C++"],
    ["Python"],
    ["Java"]
])

encoder = OneHotEncoder(sparse_output=False)

encoded_data = encoder.fit_transform(languages)

print(encoded_data)
print(encoder.categories_)
```

The exact column order is determined by the categories learned by the
encoder.

### Understanding the methods

-   `fit()`: Learns the categories from the data.
-   `transform()`: Converts categories into one-hot vectors.
-   `fit_transform()`: Performs both operations.

**ML practice:** Fit the encoder on training data, then use that same
fitted encoder to transform validation and test data. Do not
independently fit it on each dataset; that can produce inconsistent
columns and data leakage.

> Compatibility note: In older scikit-learn versions, use
> `OneHotEncoder(sparse=False)` instead of `sparse_output=False`.

## 6. Limitations of One-Hot Encoding

### 6.1 High dimensionality

The vector length equals the number of unique categories or tokens.

For a vocabulary of 50,000 tokens, every one-hot vector has 50,000
positions.

This can create large feature spaces.

### 6.2 Sparse representation

A one-hot vector contains exactly one `1` and all remaining positions
are `0`.

For a vocabulary of 50,000 tokens:

-   1 position contains `1`.
-   49,999 positions contain `0`.

This is called a **sparse representation**.

### 6.3 No semantic meaning

Consider:

``` text
King   -> [1, 0, 0, 0]
Queen  -> [0, 1, 0, 0]
Apple  -> [0, 0, 1, 0]
Car    -> [0, 0, 0, 1]
```

OHE treats each token as a separate category. It does not capture
relationships such as:

-   King and Queen are related.
-   Apple and Banana are related.
-   Car and Bus are related.

In terms of dot product, all distinct one-hot vectors are equally
dissimilar.

This is one reason dense word embeddings are useful.

## 7. OHE vs Bag of Words vs Word Embeddings

These techniques all represent text numerically, but they do different
jobs.

  -------------------------------------------------------------------------
  Parameter         One-Hot Encoding    Bag of Words      Word Embeddings
                                        (BoW)             
  ----------------- ------------------- ----------------- -----------------
  Represents        Individual          A document using  Tokens using
                    categories/tokens   token counts      dense vectors

  Output            Binary vector       Frequency vector  Dense numerical
                                                          vector

  Captures          No                  Yes               Not directly
  frequency?                                              

  Captures semantic No                  No                Can capture them
  relationships?                                          

  Typical           Sparse              Sparse            Dense
  representation                                          
  -------------------------------------------------------------------------

### Key distinction

-   **OHE:** Gives each token or category its own binary vector.
-   **BoW:** Combines token counts into one document-level vector.
-   **Word Embeddings:** Learn dense vector representations that can
    capture semantic and syntactic relationships.

## 8. Important Terms

  -----------------------------------------------------------------------
  Term                                Meaning
  ----------------------------------- -----------------------------------
  Vocabulary                          Collection of unique tokens

  One-Hot Encoding                    One position ON, all others OFF

  Vector                              Ordered numerical representation

  Sparse vector                       A vector containing mostly zeros

  Dimensionality                      Number of positions/features in a
                                      vector

  Word embeddings                     Dense vectors that can capture
                                      semantic relationships
  -----------------------------------------------------------------------

## 9. The Three Things You Should Write in Your Notes

1.  **One-Hot Encoding (OHE):** Converts categorical values or tokens
    into binary vectors, with exactly one `1` and all other positions
    `0`.
2.  **Why use OHE?** Converts categories into numerical representations
    and avoids artificial numerical ordering.
3.  **Limitations:** High dimensionality, sparse vectors, and no
    semantic understanding of relationships between words.

### Quick revision

-   Vector length = vocabulary size.
-   One category/token = one unique one-hot vector.
-   Different one-hot vectors have a dot product of zero.
-   OHE represents identity, not meaning.

## 10. Knowledge Check

Try answering these questions without looking at the notes.

1.  Why do we convert text into numerical vectors?
2.  If the vocabulary contains 5 unique tokens, what is the length of
    each one-hot vector?
3.  What does "one-hot" mean?
4.  What is the main semantic limitation of OHE?
5.  What happens to vector dimensionality when vocabulary size
    increases?

### Answers

1.  ML models need numerical representations for mathematical
    operations.
2.  5 positions.
3.  Exactly one position is `1`; all others are `0`.
4.  OHE does not capture semantic relationships between words.
5.  Vector dimensionality increases because vector length equals
    vocabulary size.

------------------------------------------------------------------------

## Final Takeaway

**We convert text into numbers so that machines can perform mathematical
operations on language. OHE is one of the simplest ways to represent
categorical values or individual tokens numerically.**
