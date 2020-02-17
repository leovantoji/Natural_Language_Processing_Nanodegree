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
  ![text_processing_summary](https://github.com/leovantoji/Natural_Language_Processing_Nanodegree/blob/master/text_processing_summary.png)

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
