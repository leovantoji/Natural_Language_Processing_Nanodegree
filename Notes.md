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
- It's better to remove **punctuation** characters with a space. This approach makes sure that words don't get concatenated together, in case the original text did not have a space before or after the punctuation.
