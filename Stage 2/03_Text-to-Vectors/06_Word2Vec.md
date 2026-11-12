# 06 --- Word2Vec

## 1. What is Word2Vec?

**Word2Vec** is a neural-network-based technique that learns dense
vector representations (word embeddings) by predicting relationships
between words and their surrounding context in a text corpus.

It was introduced by Google researchers, including Tomas Mikolov, in
2013.

Word2Vec is based on the **distributional hypothesis**:

> Words that occur in similar contexts tend to have similar meanings.

For example:

-   "The king rules the kingdom."
-   "The queen rules the kingdom."
-   "The king wears a crown."
-   "The queen wears a crown."

Because *king* and *queen* occur in similar contexts, Word2Vec can learn
similar vector representations for them.

**Important:** Word2Vec does not receive a dictionary of word meanings.
It learns statistical patterns from text.

## 2. Why do we need Word2Vec?

Traditional representations have limitations:

  -----------------------------------------------------------------------
  Technique               What it represents      Main limitation
  ----------------------- ----------------------- -----------------------
  One-Hot Encoding (OHE)  Individual words as     Does not express
                          binary vectors          similarity between
                                                  words

  Bag of Words (BoW)      Documents using word    Ignores word order and
                          counts                  semantics

  TF-IDF                  Documents using         Does not inherently
                          weighted word counts    learn semantic
                                                  relationships

  Word2Vec                Words as learned dense  Learns useful
                          vectors                 relationships from
                                                  context, but is static
  -----------------------------------------------------------------------

For example, OHE might represent a vocabulary as:

``` text
king  = [1, 0, 0, 0]
queen = [0, 1, 0, 0]
man   = [0, 0, 1, 0]
woman = [0, 0, 0, 1]
```

These vectors identify words but do not encode that *king* and *queen*
are related.

A trained Word2Vec model may instead learn vectors such as:

``` text
king  = [0.85, 0.20, 0.75]
queen = [0.83, 0.25, 0.78]
```

These are illustrative values, not universal or predetermined vectors.
Similarity is learned from usage patterns.

## 3. What are word embeddings?

A **word embedding** is a dense, low-dimensional numerical vector
representing a word.

Example:

``` python
king = [0.85, 0.20, 0.75]
```

-   **Vector:** A list of numerical values.
-   **Dense:** Most values are non-zero.
-   **Low-dimensional:** Usually far fewer dimensions than the
    vocabulary size.
-   **Learned:** Values are adjusted during training.
-   **Semantic/syntactic patterns:** The combined vector can encode
    useful relationships learned from word usage.

### Dimensions

Each number is one dimension (or learned feature) of the vector.

For example:

``` text
king = [0.85, 0.20, 0.75]
         D1    D2    D3
```

Dimensions are generally **latent features**. The model is not told that
Dimension 1 means "gender" or Dimension 2 means "royalty." Individual
dimensions usually do not have simple human-readable meanings; their
combined values help represent patterns.

Common embedding sizes include 50, 100, 300, and other chosen
dimensions.

### Dense vs. sparse representations

-   OHE is sparse: one value is 1 and the rest are 0.
-   BoW and TF-IDF are usually sparse: most vocabulary features are zero
    for a particular document.
-   Word2Vec embeddings are dense: most coordinates contain non-zero
    real values.

A 50,000-word vocabulary would require a 50,000-dimensional OHE vector
for each word. A 300-dimensional embedding represents each word with
only 300 learned values.

## 4. How does Word2Vec learn the values?

The model does **not** know in advance that *king* should have a
particular value. It learns the values through training.

1.  The model initializes trainable embedding weights, usually with
    small random values.
2.  It reads tokenized training text.
3.  It creates training examples from target words and their surrounding
    context.
4.  It predicts context or target words, depending on the architecture.
5.  It compares predictions with the actual training targets using a
    loss function.
6.  Backpropagation calculates gradients.
7.  Gradient descent updates the weights to reduce the loss.
8.  Repeating this process across many examples produces learned
    embeddings.

The basic gradient-descent update is:

$$
\theta_{\text{new}} =
\theta_{\text{old}} -
\eta\frac{\partial L}{\partial \theta}
$$

Where:

-   \(\theta\) = trainable parameter (such as an embedding value)
-   \(L\) = loss
-   \(\eta\) = learning rate
-   \(\frac{\partial L}{\partial \theta}\) = gradient of
    the loss with respect to the parameter

Different initializations, corpora, dimensions, and training settings
can produce different vectors. There is no single correct numerical
vector for a word.

## 5. The context window

A **context window** specifies how many neighbouring words around a
target word are considered during training.

Example:

``` text
The king is powerful
```

If the target is `king` and the window size is 2, the nearby context may
include:

``` text
The   king   is   powerful
  \     |     |      /
       target
```

The precise number of available context words depends on the target's
position and sentence boundaries.

The context window controls how much local context contributes to
training.

## 6. The two Word2Vec architectures

Word2Vec has two main architectures:

-   **CBOW (Continuous Bag of Words):** Context → Target
-   **Skip-Gram:** Target → Context

### 6.1 CBOW --- Continuous Bag of Words

CBOW predicts a target word from its surrounding context words.

Example:

``` text
The _____ is powerful.
```

Expected target:

``` text
king
```

Context words:

``` text
The, is, powerful
```

Typical simplified workflow:

1.  Select the context words.
2.  Convert them to token IDs (or conceptually one-hot vectors).
3.  Look up their embeddings in the input embedding matrix.
4.  Combine the context embeddings, commonly by averaging.
5.  Use the combined representation to score candidate target words.
6.  Convert scores to probabilities (with a softmax in the basic
    full-softmax formulation).
7.  Compare predictions with the actual target and calculate loss.
8.  Update model weights.

If the context vectors are:

$$
v_{\text{the}} = [0.2, 0.1, 0.3]
$$

$$
v_{\text{is}} = [0.4, 0.2, 0.1]
$$

$$
v_{\text{powerful}} = [0.6, 0.3, 0.2]
$$

Their average is:

$$
h = \frac{v_{\text{the}} + v_{\text{is}} + v_{\text{powerful}}}{3}
$$

$$
h = [0.4, 0.2, 0.2]
$$

This combined context representation is used to predict the target word.

**Why "Bag of Words"?** The context words are combined without
preserving their order in the averaged context representation.

### 6.2 Skip-Gram

Skip-Gram predicts surrounding context words from a target word.

Example:

``` text
The king is powerful
```

Target:

``` text
king
```

Context:

``` text
The, is, powerful
```

The training pairs are:

  Input (target)   Expected context
  ---------------- ------------------
  king             the
  king             is
  king             powerful

Typical simplified workflow:

1.  Select a target word.
2.  Look up its input embedding.
3.  Use the embedding to calculate scores for possible context words.
4.  Convert scores into probabilities in the full-softmax version.
5.  Compare predictions with actual context words.
6.  Calculate loss and update weights.

### CBOW vs. Skip-Gram

  -----------------------------------------------------------------------
  Feature                 CBOW                    Skip-Gram
  ----------------------- ----------------------- -----------------------
  Direction               Context → Target        Target → Context

  Input                   Multiple context words  One target word

  Prediction              One target word         Context words, through
                                                  target-context pairs

  Context handling        Combines context        Learns from
                          vectors                 target-context pairs

  Training speed          Often faster            Often slower

  Rare words              Often effective for     Often useful for rare
                          frequent words          words
  -----------------------------------------------------------------------

**Revision hook:** CBOW fills in the missing word; Skip-Gram predicts
the neighbours of a given word.

## 7. Inside the Skip-Gram neural network

A simplified Skip-Gram model uses two trainable matrices.

Suppose:

-   Vocabulary size \(V=6\)
-   Embedding dimension \(D=3\)

### Input embedding matrix

$$
W_{\text{in}} \in \mathbb{R}^{V \times D}
$$

Its shape is \(6 \times 3\). Each row is the input embedding for one
vocabulary word.

### Output weight matrix

$$
W_{\text{out}} \in \mathbb{R}^{D \times V}
$$

Its shape is \(3 \times 6\). It helps calculate scores for candidate
context words.

### Step 1: One-hot input and embedding lookup

Suppose `king` has token ID 1:

$$
x = [0, 1, 0, 0, 0, 0]
$$

The input embedding is:

$$
h = xW_{\text{in}}
$$

Because the one-hot vector has a single 1, this selects the row for
`king` from \(W_{\text{in}}\).

### Step 2: Calculate output scores (logits)

$$
z = hW_{\text{out}}
$$

The result contains one score for each vocabulary word.

### Step 3: Convert scores to probabilities

In the full-softmax formulation:

$$
P(w_i \mid \text{king}) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Softmax converts scores into probabilities that sum to 1.

### Step 4: Calculate loss

If the actual context word is `the`, and the model predicts:

$$
P(\text{the} \mid \text{king}) = 0.20
$$

The cross-entropy loss for this one target-context pair is:

$$
L = -\log\left(P(\text{the} \mid \text{king})\right)
$$

$$
L = -\log(0.20) \approx 1.609
$$

A higher probability for the correct context word produces a lower loss.

### Step 5: Update the weights

Backpropagation calculates gradients, and gradient descent updates the
trainable weights. Both input and output weights are adjusted during
training.

After training, the rows of \(W_{\text{in}}\) are commonly used
as the learned word embeddings.

## 8. Negative Sampling

Calculating a full probability distribution over a very large vocabulary
can be expensive.

**Negative Sampling** makes training more efficient by training the
model to distinguish:

-   Positive pairs: actual target-context pairs found in the corpus.
-   Negative pairs: randomly sampled words paired with the target that
    are not the observed context for that training example.

Example sentence:

``` text
The king wears a crown.
```

Positive pair:

``` text
(king, crown)
```

Illustrative negative pairs:

``` text
(king, banana)
(king, airplane)
(king, computer)
```

The model learns to give higher compatibility scores to positive pairs
and lower scores to sampled negative pairs.

The negative-sampling objective for one positive pair and (k) negative
samples is commonly expressed as:

$$
L = -\log \sigma(v_c'^{T}v_w)
- \sum_{i=1}^{k}\log \sigma(-v_{n_i}'^{T}v_w)
$$

Where:

-   \(v_w\) = input vector for the target word
-   \(v_c'\) = output vector for the positive context word
-   \(v_{n_i}'\) = output vector for a negative sample
-   \(k\) = number of negative samples
-   \(\sigma\) = sigmoid function

A negative sample is a training negative, not necessarily a word that
could never appear near the target in natural language.

## 9. Hierarchical Softmax

Another strategy for efficient Word2Vec training is **Hierarchical
Softmax**.

-   Organizes vocabulary words into a binary tree.
-   Predicts a word by following a path through the tree.
-   Can reduce the cost of computing probabilities over a large
    vocabulary.

  -----------------------------------------------------------------------
  Method                              Main idea
  ----------------------------------- -----------------------------------
  Standard Softmax                    Calculates probabilities across the
                                      vocabulary

  Negative Sampling                   Uses positive pairs and a small set
                                      of sampled negative pairs

  Hierarchical Softmax                Uses a binary tree to calculate
                                      word probabilities
  -----------------------------------------------------------------------

Negative Sampling and Hierarchical Softmax are alternative training
strategies; both are not required simultaneously.

## 10. Similarity between word vectors

Once words have embeddings, their vectors can be compared.

A common measure is **cosine similarity**, which measures the angle
between two vectors:

$$
\operatorname{cosine}(A,B) =
\frac{A \cdot B}{\lVert A\rVert\lVert B\rVert}
$$

General interpretation:

    Score Interpretation
  ------- -----------------------
       +1 Same direction
        0 Perpendicular vectors
       -1 Opposite directions

A higher cosine similarity often indicates more similar usage patterns
in an embedding space. It does not guarantee that two words are
synonyms.

### Vector arithmetic

A famous Word2Vec illustration is:

$$
\operatorname{vec}(\text{king}) - \operatorname{vec}(\text{man})
+ \operatorname{vec}(\text{woman})
\approx \operatorname{vec}(\text{queen})
$$

This demonstrates that some relationships can be reflected in vector
directions. It is not a universal rule or proof of human-like reasoning.
Results depend on the corpus and training method, and embeddings can
reflect training-data biases.

## 11. Visualizing word embeddings

A word embedding is a numerical vector. A graph is a way for humans to
inspect relationships between vectors.

-   In a 2D embedding, the two values can be plotted directly as X and
    Y.
-   In a 300-dimensional embedding, the original vectors cannot be
    directly plotted on a 2D screen.
-   Dimensionality-reduction methods such as PCA, t-SNE, and UMAP can
    project high-dimensional vectors into 2D or 3D for visualization.

  -----------------------------------------------------------------------
  Method                              Purpose
  ----------------------------------- -----------------------------------
  PCA                                 Projects data into fewer dimensions
                                      while preserving as much variance
                                      as possible

  t-SNE                               Helps visualize local
                                      neighbourhoods and clusters

  UMAP                                Often preserves local and some
                                      global structure in a
                                      low-dimensional visualization
  -----------------------------------------------------------------------

**Important:** The visualization is not required for Word2Vec training.
The model learns vectors; dimensionality reduction is a separate step
used to display them.

## 12. Complete Word2Vec workflow

``` text
Raw text corpus
      |
      v
Preprocessing and tokenization
      |
      v
Vocabulary creation
      |
      v
Assign token IDs
      |
      v
Create target-context training examples
      |
      v
Initialize embedding and output weights
      |
      v
CBOW or Skip-Gram prediction
      |
      v
Calculate loss
      |
      v
Backpropagation and gradient descent
      |
      v
Repeat over training examples and epochs
      |
      v
Learned word embeddings
      |
      v
Similarity search / downstream NLP task
      |
      v
Optional 2D or 3D visualization
```

## 13. Python implementation using Gensim

Install:

``` bash
pip install gensim
```

Train a Skip-Gram model:

``` python
from gensim.models import Word2Vec

sentences = [
    ["the", "king", "is", "powerful"],
    ["the", "queen", "is", "powerful"],
    ["the", "king", "wears", "a", "crown"],
    ["the", "queen", "wears", "a", "crown"],
    ["the", "man", "is", "strong"],
    ["the", "woman", "is", "strong"]
]

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=2,
    min_count=1,
    sg=1,
    workers=1,
    epochs=100
)

# Retrieve a word embedding
print(model.wv["king"])

# Find the most similar words
print(model.wv.most_similar("king", topn=5))

# Calculate cosine similarity
print(model.wv.similarity("king", "queen"))

# Inspect vocabulary
print(model.wv.index_to_key)

# Check vocabulary membership
print("king" in model.wv)
```

### Important parameters

  Parameter           Meaning
  ------------------- -----------------------------------------
  `sentences`         Tokenized training corpus
  `vector_size=100`   Embedding dimension
  `window=2`          Maximum context distance on either side
  `min_count=1`       Ignore words occurring fewer than once
  `sg=1`              Skip-Gram; `sg=0` selects CBOW
  `workers=1`         Number of worker threads
  `epochs=100`        Number of training passes

### Useful methods

``` python
model.wv["king"]
model.wv.most_similar("king", topn=5)
model.wv.similarity("king", "queen")
model.wv.index_to_key
"king" in model.wv
```

**Practical note:** The small example corpus is for learning the API,
not for producing high-quality embeddings. Useful general-purpose
embeddings require a much larger and more varied corpus.

## 14. Limitations of Word2Vec

1.  **Static embeddings:** A word generally has one vector, even when it
    has multiple meanings. For example, *bank* in "river bank" and "bank
    account" gets the same standard Word2Vec vector.
2.  **Out-of-vocabulary words:** Standard Word2Vec cannot directly
    retrieve a learned vector for a word absent from its vocabulary.
3.  **Limited context window:** It learns from a local context, not the
    full sentence or document.
4.  **No contextual sentence understanding:** It creates word vectors
    rather than context-dependent token representations.
5.  **Training-data bias:** Learned associations can reflect biases in
    the training corpus.

FastText addresses some out-of-vocabulary and rare-word issues using
subword information. Contextual models such as ELMo and BERT address the
limitation of one static vector per word.

## 15. Word2Vec vs. other embedding techniques

  -----------------------------------------------------------------------
  Technique                           Main approach
  ----------------------------------- -----------------------------------
  Word2Vec                            Predicts target words or context
                                      words

  GloVe                               Learns from global word
                                      co-occurrence statistics

  FastText                            Uses character N-Grams and subword
                                      information

  ELMo                                Generates contextual word
                                      representations using a
                                      bidirectional language model

  BERT                                Generates contextual token
                                      representations using a Transformer
                                      encoder
  -----------------------------------------------------------------------

## 16. The three things you should write in your notes

1.  **Word2Vec:** A neural-network-based technique that learns dense
    word embeddings by predicting relationships between words and their
    surrounding context.
2.  **Two architectures:** CBOW predicts a target word from context
    words; Skip-Gram predicts context words from a target word.
3.  **Learning process:** Initialize weights → predict context →
    calculate loss → backpropagate → update weights using gradient
    descent. Negative Sampling improves training efficiency by using
    positive and sampled negative pairs.

### One-line revision

> CBOW fills in the missing word; Skip-Gram predicts the neighbours of a
> given word.

## 17. Quick revision questions

1.  What is Word2Vec, and what is the distributional hypothesis?
2.  What does an embedding dimension represent?
3.  What is a context window?
4.  How does CBOW differ from Skip-Gram?
5.  Why does Word2Vec use input and output weight matrices?
6.  What is the role of Softmax?
7.  What is the loss function trying to minimize?
8.  How do backpropagation and gradient descent update embeddings?
9.  What is Negative Sampling, and why is it useful?
10. What is Hierarchical Softmax?
11. How is cosine similarity used with embeddings?
12. Why are standard Word2Vec embeddings called static?
13. How does FastText help with unseen words?

## 18. Recommended learning progression

1.  Word2Vec --- CBOW and Skip-Gram (**current topic**)
2.  GloVe --- global word co-occurrence statistics
3.  FastText --- character N-Grams and subword embeddings
4.  ELMo and contextual embeddings
5.  BERT and Transformer embeddings
