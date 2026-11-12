# 53. Word Embeddings in NLP

## 1. Why Word Embeddings?

Earlier text representations have limitations:

| Technique | Represents | Key limitation |
|---|---|---|
| One-Hot Encoding (OHE) | Individual tokens as binary vectors | Different tokens are equally distant; no learned similarity |
| Bag of Words (BoW) | Documents as token counts | Ignores order and does not inherently capture meaning |
| N-Gram BoW | Documents as counts of consecutive token sequences | Only captures local order; vocabulary can grow rapidly |
| TF-IDF | Documents as frequency-and-rarity-weighted features | Doesn't inherently understand synonyms or context |

**Motivation:** We want numerical representations in which related words can have related vectors, rather than treating every word as an independent category.

## 2. What Are Word Embeddings?

**Word embeddings are learned, usually dense numerical vectors that represent words and can capture semantic and syntactic relationships from patterns in text.**

For illustration (not pretrained values):

```text
king  → [ 0.82, 0.75, 0.12]
queen → [ 0.80, 0.72, 0.18]
apple → [-0.20, 0.10, 0.85]
```

Here, `king` and `queen` have similar representations. Real embeddings often have hundreds of dimensions, and their coordinates typically do **not** have simple human-readable meanings.

### Dense vs. sparse

- **Sparse vector:** Mostly zeros. OHE for a vocabulary of 50,000 words has 50,000 positions with just one `1`.
- **Dense vector:** Many nonzero values. An embedding might have 100 or 300 learned coordinates.
- **Lower dimensionality:** A 300-dimensional word embedding is much shorter than a 50,000-dimensional one-hot vector, although an entire embedding table still stores vectors for many words.

## 3. How Do Embeddings Learn Relationships?

The **distributional hypothesis** says that words appearing in similar contexts tend to have similar meanings or uses.

Examples:

```text
The king wore a crown.
The queen wore a crown.
The cat sat on the sofa.
The dog sat on the sofa.
```

A training algorithm observes target words and surrounding context words, adjusts numerical parameters to improve a prediction or co-occurrence objective, and gradually learns useful vectors. It is learning statistical regularities, **not** reading dictionary definitions or acquiring guaranteed human-like understanding.

### Context window

In `The cat sat on the sofa`, if the target is `sat` and the window size is 2, the nearby context is `The`, `cat`, `on`, `the` (subject to preprocessing and tokenization). Window size controls how much local context the learning task considers.

## 4. Vector Space and Similarity

Each embedding is a point/vector in a multidimensional space. Related words may appear close together.

### Cosine similarity

A common way to compare embeddings is:

$$
\operatorname{cosine}(A,B)=\frac{A\cdot B}{\lVert A\rVert\lVert B\rVert}
$$

- `+1`: Same direction.
- `0`: Orthogonal (perpendicular).
- `−1`: Opposite directions.

A higher cosine score often suggests more similar representations, but it does not guarantee identical meanings. Cosine similarity compares **direction**, not vector magnitude.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

king = np.array([[0.8, 0.7, 0.1]])
queen = np.array([[0.8, 0.6, 0.2]])
apple = np.array([[-0.3, 0.1, 0.9]])

print(cosine_similarity(king, queen))
print(cosine_similarity(king, apple))
```

These are illustrative vectors, not real learned word embeddings.

### Vector arithmetic

A famous demonstration from some Word2Vec spaces is:

$$
\operatorname{vec}(king)-\operatorname{vec}(man)+\operatorname{vec}(woman)\approx\operatorname{vec}(queen)
$$

This is an **illustrative tendency**, not a universal law. Analogies depend on the model, corpus, and evaluation method, and can reflect biases in training data.

## 5. Major Traditional Embedding Techniques

### 5.1 Word2Vec

Word2Vec is a neural prediction-based family of methods popularized in 2013. It learns word vectors from local contexts.

**CBOW (Continuous Bag of Words): Context → Target**

```text
Context: The cat ___ on the sofa
Target:  sat
```

CBOW uses surrounding words to predict the center word.

**Skip-Gram: Target → Context**

```text
Target:  sat
Context: The, cat, on, the
```

Skip-Gram uses the center word to predict surrounding words.

| CBOW | Skip-Gram |
|---|---|
| Predicts center word from context | Predicts context words from center word |
| Often faster to train | Often useful for rare-word representations |
| Aggregates context | Learns from target-context pairs |

Training optimizations include **negative sampling** and **hierarchical softmax**; these are good next-step topics.

### 5.2 GloVe (Global Vectors)

GloVe learns embeddings using **global word co-occurrence statistics**. It uses patterns of how often words occur together across the corpus, rather than relying solely on local prediction examples.

For instance, `ice` may co-occur with `cold`, while `steam` may co-occur with `hot`. Co-occurrence relationships help shape the learned vector space.

### 5.3 FastText

FastText incorporates **character N-Grams** into word representations. For `playing`, character trigrams might include `pla`, `lay`, `ayi`, `yin`, `ing` (actual implementations can add boundary markers).

This can help with:

- Rare words and some unseen words.
- Spelling and morphological variations.
- Languages with rich word inflections.

## 6. Static vs. Contextual Embeddings

### Static embeddings

Traditional Word2Vec, GloVe, and FastText generally give a word the same learned representation regardless of the sentence.

```text
I deposited money in the bank.
I sat beside the river bank.
```

The word `bank` has different senses, but a static word embedding normally uses one vector for both.

### Contextual embeddings

Models such as **BERT** and other Transformers generate token representations that depend on the surrounding text. `bank` in a financial sentence can therefore have a different representation from `bank` in a river sentence.

**Important:** Transformer tokenizers may split words into subwords. Contextual models therefore often produce *token* vectors, not necessarily one vector per whole word.

## 7. How Embeddings Fit Into NLP

A simplified neural NLP pipeline:

```text
Raw text
   ↓
Preprocessing / tokenization
   ↓
Token IDs
   ↓
Embedding lookup (learned dense vectors)
   ↓
Neural model (e.g., RNN or Transformer)
   ↓
Prediction / classification / generation
```

For traditional machine learning, an alternative pipeline might use **TF-IDF → Logistic Regression**. These are different representation choices, not mandatory consecutive steps.

A document representation can also be created by pooling word vectors or by using a model trained to embed whole sentences/documents.

## 8. Comparing the Techniques

| Property | OHE | BoW | TF-IDF | Word embeddings |
|---|---|---|---|---|
| Typical unit | Token | Document | Document | Word/token |
| Values | Binary | Counts | Weights | Learned real numbers |
| Typical sparsity | Sparse | Sparse | Sparse | Dense |
| Semantic relationships | Not inherent | Not inherent | Not inherent | Can be learned |
| Word order | Not encoded | Ignored | Ignored | Depends on training/model |
| Context-dependent | No | No | No | Static: no; contextual: yes |

N-Gram features can be used with **both BoW and TF-IDF** to capture local word sequences.

## 9. Limitations of Word Embeddings

1. **Polysemy:** Static embeddings struggle with multiple meanings of the same word.
2. **Training-data dependence:** Rare or poorly represented concepts may have weak embeddings.
3. **Bias:** Learned vectors may reflect social or other biases in text.
4. **Sentence meaning:** A word vector is not automatically a complete sentence representation.
5. **Long-range context:** Traditional static embeddings do not dynamically incorporate full-sentence context.

Contextual embeddings address some of these issues, but are not flawless.

## 10. Practical Applications

- Semantic word similarity and related-term discovery.
- Features for text classification and sentiment analysis.
- Search, recommendation, and information retrieval (often using sentence/document embeddings).
- Inputs to neural networks for NLP.
- Foundation for understanding contextual representations in Transformers and LLMs.

## 11. The Three Things You Should Write in Your Notes

**1. Word Embeddings:** Learned, dense numerical vectors representing words; they can capture semantic and syntactic relationships.

**2. Core intuition:** Words used in similar contexts tend to develop similar vectors. Similarity is often measured using cosine similarity.

**3. Main techniques:** **Word2Vec** learns from context prediction (CBOW: context → word; Skip-Gram: word → context); **GloVe** uses global co-occurrence statistics; **FastText** uses character N-Grams. **Static** vectors stay the same across sentences; **contextual** vectors depend on the sentence.

### Quick recall

| Term | Memory trigger |
|---|---|
| Sparse | Mostly zeros |
| Dense | Many nonzero values |
| Distributional hypothesis | Similar contexts → related representations |
| Cosine similarity | Angle between vectors |
| CBOW | Context predicts center word |
| Skip-Gram | Center word predicts context |
| GloVe | Global co-occurrence |
| FastText | Character subwords |
| Static | Same word vector across contexts |
| Contextual | Vector changes with context |

## 12. Check Your Understanding

1. Why can't OHE directly express that `love` and `enjoy` are similar?
2. What does it mean for an embedding to be dense?
3. How does the distributional hypothesis motivate embedding training?
4. What does cosine similarity measure?
5. How do CBOW and Skip-Gram differ?
6. Why can FastText handle some unseen words better than a basic word-level model?
7. Why is the word `bank` a problem for static embeddings?

## 13. What to Learn Next

1. **Word2Vec in depth:** context windows, CBOW, Skip-Gram, training objective, negative sampling, hierarchical softmax.
2. **GloVe:** co-occurrence matrices and global statistics.
3. **FastText:** character N-Grams and out-of-vocabulary handling.
4. **Contextual embeddings:** ELMo, BERT, and Transformer representations.
5. **Hands-on:** Load pretrained vectors, compare cosine similarities, and use embeddings for a text classification task.

---

**Final takeaway:** OHE identifies a word, BoW counts words, TF-IDF weights terms by document distinctiveness, and **word embeddings learn vector representations that can encode relationships between words**.
