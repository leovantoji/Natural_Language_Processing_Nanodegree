## Intro to NLP
- Even though computers cannot understand unstructured texts like humans, computers can analyse documents to identify: 
  - Frequent & rare words.
  - Tone & sentiment of the message.
  - Clusters of documents.
- **Context** is everything when it comes to NLP.
- Common **NLP pipeline** includes:
  - Text processing.
  - Feature extraction.
  - Modelling.

## Text processing
- Text processing generally includes:
  - Cleaning (e.g. remove HTML tags, etc.).
  - Normalisation (e.g. convert texts to lower case, remove punctuations and extra spaces, etc.).
  - Stop word removal (e.g. remove words that are too common, etc.).
  - Identify different parts of speech (e.g. which words are nouns, verbs, or named entity, and convert words into canonical forms using stemming and lemmatisation).
- Remove HTML tags using **Beautiful Soup** library.
  ```python
  from bs4 import BeautifulSoup
  soup = BeautifulSoup(r.text, 'html5lib')
  print(soup.get_text())
  ```
- It's better to **remove punctuation characters with a space**. This approach makes sure that words don't get concatenated together, in case the original text did not have a space before or after the punctuation.
- **Tokenisation** is simply splitting each sentence into a sequence of words.
  ```python
  words = text.split()
  
  # use NLTK
  from nltk.tokenize import word_tokenize
  words = word_tokenize(text)
  ```
- Sentence tokenisation with `nltk`.
  ```python
  from nltk.tokenize import sent_tokenize
  sentences = sent_tokenize(text)
  ```
- List of **stop words** in English from `nltk`.
  ```python
  from nltk.corpus import stopwords
  print(stopwords.words('english'))
  
  # remove stop words
  words = [w for w in words if w not in stopwords.words('english')]
  ```
- **Part-of-speech tagging** with `nltk`.
  ```python
  from nltk import pos_tag
  # tag parts of speech
  sentence = word_tokenize('I always lie down to tell a lie.')
  pos_tag(sentence)
  ```
- Visualise the **parse tree**.
  ```python
  for tree in parser.parse(sentence):
    tree.draw()
  ```
- **Named entity recognition** is often used to index and search for news articles.
  ```python
  from nltk import pos_tag, ne_chunk
  from nltk.tokenize import word_tokenize
  
  # recognise named entities in a tagged sentence
  ne_chunk(pos_tag(word_tokenize('Antonio joined Udacity Inc. in California.')))
  ```
- **Stemming** is the process of reducing a word to its stem or root form. Stemming is meant to be a **fast and crude operation** carried out by applying very simple search and replace style rules. `nltk` has language-specific stemmers.
  ```python
  from nltk.stem.porter import PorterStemmer
  
  # reduce words to their stems
  stemmed = [PorterStemmer().stem(w) for w in words]
  ```
- **Lemmatisation** is another technique to reduce words to a normalised form using a dictionary to map different variants of a word back to its root. For example, `was`, `were`, and `is` can be reduced to `be`.
  ```python
  from nltk.stem.wordnet import WordNetLemmatizer
  
  # reduce words to their root form. Default pos is noun.
  lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
  
  # lemmatise verbs by specifying pos
  lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
  ```
- Summary:
  ![text_processing_summary](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/text_processing_summary.png)

## Spam Classifer
- The basic idea of **Bag of Words (BoW)** is to take a piece of text and count the frequency of words in that text. It is important to note that the BoW concept **treats each word individually** and the **order in which the words occur doesn't matter**.

## Part of Speech Tagging with Hidden Markov Models
- [From Wikipedia](https://en.wikipedia.org/wiki/Part-of-speech_tagging). In corpus linguistics, **part-of-speech-tagging (POS tagging)** is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context - i.e., its relationship with adjacent and related words in a phrase, sentence, or paragraph. For example:
  - Sentence: Mary has a little lamb.
  - Noun (N): Mary
  - Verb (V): has
  - Determinant (Dt): a
  - Adjective (Ad): little
  - Noun (N): lamb
- Part-of-speech tagging is often used to help disambiguate natural language phrases because it can be done quickly with high accuracy. Tagging can be used for many NLP tasks like determining correct pronunciation during speech synthesis (for example, _dis_-count as a noun vs dis-_count_ as a verb), for information retrieval, and for word sense disambiguation.
- **Lookup tables** don't work very well with words having different tags.
- [From Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model). **Hidden Markov Model (HMM)** is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobservable (i.e. hidden) states.
- `λ = (A,B)` specifies an HMM in terms of an **emission** probability distribution `A` and a **state transition** probability distribution `B`.
  - The emission probabilities give the conditional probability of observing evidence values for each hidden state.
  - The transition probabilities give the conditional probability of moving between states during the sequence. 
- [From Wikipedia](https://en.wikipedia.org/wiki/Viterbi_algorithm). The **Viterbi algorithm** is a dynamic programming algorithm for finding the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events, especially in the context of Markov information sources and hidden Markov models (HMM). 

## Feature extraction and embeddings
- A **document-term matrix** is a mathematical matrix that describes the frequency of terms that occur in a collection of documents. Each document is a row, and each term is a column.
- **Count of common words** is a commonly used approach to match similar documents. Nonetheless, this approach has an **inherent flaw**. As the size of the document increases, the number of common words tend to increase even if the documents talk about different topics.
- **Cosine similarity** is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. This metric is used to measure how **similar the documents** are irrespective of their size. [Machine Learning Plus](https://www.machinelearningplus.com/nlp/cosine-similarity/).
![cosine_similarity_formula](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/Cosine-Similarity-Formula-1.png)
- **TF-IDF** (term frequency - inverse document frequency): *tfidf(t, d, D) = tf(t, d) x idf(t, D)*. TF-IDF is an innovative approach to assigning weights to words that signify their relevance in the document.
  - Term frequency *tf(t,d)* is the ratio between the raw count of a term, *t*, in a document, *d*, divided by the total number of terms in *d*.
  - Inverse document frequency *idf(t, D)* is the logarithm of the total number of documents in the collection, *D*, divided by the number of documents where *t* is present.
  - It is a way to score the importance of words (or "terms") in a document based on how frequently they appear across multiple documents.
  - If a word appears frequently in a document, it's important. Give the word a high score. But if a word appears in many documents, it's not a unique identifier. Give the word a low score. Therefore, common words like `the` and `for`, which appear in many documents, will be scaled down. Words that appear frequently in a single document will be scaled up.
- **Word embedding** is the collective name for a set of language modelling and feature learning techniques in natural language processing where words or phrases from the vocabulary are mapped to vectors of real numbers.
- The core idea of **Word2Vec** is that a model, which is able to predict a given word given neighbouring words or vice versa, is likely to capture the contextual meanings of words very well.
  - Neighbouring words: **Continuous Bag of Words**.
  - Middle word: **Continuous Skip-gram**.
- Properties of **Word2Vec**:
  - Robust, distributed representation.
  - Vector size independent of vocabulary.
  - Train once, store in lookup table.
  - Deep learning ready.
- **GloVe** (Global vectors for word representation) is an approach that tries to **optimise the vector representation of each word** by using **co-occurence statistics**.
- **t-SNE** (t-Distributed Stochastic Neighbouring Embedding) is a **dimensionality reduction technique** that can map high dimensional vectors to a lower dimensional space.

## Topic Modelling
- **Topic modelling** is a type of statistical modelling for discovering the abstract topics that occur in a collection of documents. **Latent Dirichlet Allocation (LDA)** is an example of topic model and is used to classify text in a document to a particular topic. [From MonkeyLearn](https://monkeylearn.com/blog/introduction-to-topic-modeling/)
- In statistics, **latent variables** (hidden variables) are variables that are **not directly observed** but are rather **inferred from other observed variables**. 
  - *BoW model* without latent variables has 500K parameters.
  ![Bag_of_Words](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/bag-of-words-quiz.png)
  - *Latent variable model* has only 15K parameters.
  ![Latent_variables](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/how-many-parameters-quiz.png)
- An **LDA** model **factors the BoW** model into **2 matrices**:
  - The first matrix indexes **documents by topic**.
  - The other matrix indexes **topics by word**.
- LDA is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modelled as Dirichlet distributions.
  - Each document is modelled as a multinomial distribution of topics and each topic is modelled as a multinomial distribution of words.
  - LDA assumes that the every chunk of text we feed into it will contain words that are somehow related. Therefore choosing the right corpus of data is crucial.
  - It also assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution.
- [From Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution). In probability theory and statistics, the **beta distribution** is a family of continuous probability distributions defined on the interval *\[0, 1\]* parametrised by two **positive shape parametres**, denoted by *α* and *β*, that appear as exponents of the random variable and control the shape of the distribution. The **generalisation to multiple variables** is called a **Dirichlet distribution**. 
  ![Dirichlet_distributions](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/dirichlet_distributions.png)
  ![LDA](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/lda.png)
  ![Topic_model](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/topic_model.png)

## Sequence to Sequence
- [Keras blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html): **Sequence-to-Sequence (seq2seq)** learning is about training models to convert sequences from one domain (e.g. sentences in English) to sequences in another domain (e.g. the same sentences translated to French). For example: `the cat sat on the mat` → seq2seq → `le chat etait assis sur le tapis`.
  ![seq2seq](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/seq2seq.png)
- **Applications** of seq2seq: Machine translation, question-answer, news-summary, captioning text, etc.
  ![captioning_text](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/captioning_text.png)
- **Seq2Seq Architecture**:
  - The inference process is done by handling inputs to the encoder.
  - The encoder summarises the inputs into a context variable or state.
  - The decoder proceeds to take the context and generate the output sequence.
  ![seq2seq_architecture](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/seq2seq_architecture.png)
  ![seq2seq_architecture_in_depth](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/seq2seq_architecture_in_depth.png)
- A seq2seq model works by **feeding one element of the input sequence at a time to the encoder**.

## Deep Learning Attention
- [LiLianWeng's github](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) Attention can be broadly interpreted as a vector of importance weights. In order to predict one element, such as a pixel in an image or a word in a sentence, we estimate using the attention vector how strongly it is correlated with other elements and take the sum of their values weighted by the attention vector as the approximation of the target.
- Limitation of seq2seq models which can be solved using attention methods:
  - The **fixed size of the context matrix** passed from the encoder to the decoder is a **bottleneck**.
  - **Difficulty of encoding long sequences** and **recalling long-term dependancies**.
- The **size of the context matrix** in an attention seq2seq model depends on the **length of the inpput sequence**.
- Seq2Seq **without Attention**:
![seq2seq_wo_attention](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/seq2seq_wo_attention.png)
- Seq2Seq **with Attention**:
  ![seq2seq_w_attention](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/seq2seq_w_attention.png)
- Every time step in the decoder requires calculating an attention vector in a seq2seq model with attention.
- **Attention Encoder**:
  ![attention_encoder_unrolled_view](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/attention_encoder_unrolled_view.png)
- **Attention Decoder**:
  ![attention_decoder_unrolled_view](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/attention_decoder_unrolled_view.png)
- **Context Vector Generation**:
  ![attention_context_vector_generation](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/attention_context_vector_generation.png)
- **Additive attention** (Bahdanau et al., 2015) uses a one-hidden layer feed-forward network to calculate the attention alignment where **v<sub>a</sub>** and **W<sub>a</sub>** are learned attention parameters. Analogously, we can also use matrices **W<sub>1</sub>** and **W<sub>2</sub>** to learn separate transformations for **h<sub>i</sub>** and **s<sub>j</sub>** respectively, which are then summed:
  ![additive_attention_formula1](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/additive_attention_formula1.png)
  ![additive_attention_formula2](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/additive_attention_formula2.png)
- **Multiplicative attention** (Luong et al., 2015) simplifies the attention operation by calculating the following function:
  ![multiplicative_attention_formula](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/multiplicative_attention_formula.png)
  ![multiplicative_attention_detail1](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/multiplicative_attention_detail1.png)
  ![multiplicative_attention_detail2](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/multiplicative_attention_detail2.png)
- The **intuition behind using dot product** as a scoring method is that the dot product of two vectors  in word-embedding space is a **measure of the similartiy** between them.
- The simplicity of **not having a weight vector** in multiplicative attention comes the drawback of assuming the encoder and decoder have the same embedding space. Thus, this might work for text summarisation where both the encoder and decoder use the same language and the same embedding space. For machine translation, since each language tends to have its own embedding space, we might want to use a second scoring method.
  ![multiplicative_attention_detail3](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/multiplicative_attention_detail3.png)
- Full view of multiplicative attention decoding phase:
  ![multiplicative_attention_decoding](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/multiplicative_attention_decoding.png)
- Computer Vision applications:
  ![cv_attention_application](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/cv_attention_application.png)
  ![cv_attention_application2](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/cv_attention_application2.png)
- **Self-attention**: Without any additional information, however, we can still extract relevant aspects from the sentence by allowing it to attend to itself using self-attention (Lin et al., 2017)
- The **Transformer**:
  ![transformer](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/transformer.png)

## Speech Recognition
- **VUI pipeline** includes:
  - Speech recognition: Voice to text.
    - Acoustic Model → Language Model → Accent Model
  - Text to text: Text input reasoned to text output.
  - Text to speech.
- **Automatic Speech Recognition (ASR)**: The goal is to input any continuous audio speech and output the text equivalent. ASR should be **speaker independent** and have **high accuracy**.
- **Models in speech recognition** can be divided into acoustic models and language models.
  - The **acoustic model** solves the problem of **turning sound signals** into some kind of **phonetic representation**.
  - The **language model** houses the **domain knowledge of words, grammar and sentence structure** for the language.
- The **challenges of ASR** are: 
  - Noises.
  ![noise](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/asr_challenge_noise.png)
  - Variability in pitch, volume and speed.
  ![asr_challenge_variability](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/asr_challenge_variability.png)
  ![asr_challenge_word_speed](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/asr_challenge_word_speed.png)
  - Ambiguity specific to languages.
  ![asr_challenge_language_knowledge](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/asr_challenge_language_knowledge.png)
  ![asr_challenge_spoken_vs_written](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/asr_challenge_spoken_vs_written.png)  
- **Signal analysis**

## Extracurricular
### Hyperparameters
- There are generally two types of hyperparameters:
  - Optimiser hyperparameters which are related to the optimisation and training process than to the model itself. These include learning rate, mini-batch size, and the number of training iterations or epochs.
  - Model hyperparameters which are involved in the structure of the model. These include the number of layers or model specific parameters for architecture.
- The **learning rate** is the single most important hyperparameter.
  ![learning_rate](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/images/learning_rate.png)
- If the **mini-batch size** variable is too small, the training might be too slow. If the **mini-batch size** is too large, it could be computationally taxing (i.e. needs more memory) and could result in worse accuracy. Generally, 32, 64, 128, and 256 are potentially good starting values.
- The **number of training iterations** is a hyperparameter we can optimise automatically using a technique called **early stopping** (also "early termination").
- RNN Hyperparameters: cell structure like LSTM and GRU should both be tried in the task and compare. 
