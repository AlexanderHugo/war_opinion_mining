from collections import defaultdict
import re
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
import pandas as pd
from dataStructures import PostDataStructure

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")

class CorpusManager:
    _instance = None  # Singleton instance

    def __new__(cls):
        """ Implements a Singleton pattern to ensure only one CorpusManager exists. """
        if cls._instance is None:
            cls._instance = super(CorpusManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """ Initializes data structures and preprocessing tools. """
        self.texts: List[PostDataStructure] = []  # Raw text data
        self.tokenized_texts = []  # Tokenized texts for Sentiment Analysis
        self.dictionary = None
        self.corpus = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words("english")
        self.words_to_rem = ['http', 'com', 'www']
        self.cooccurrences = []

    def clean_and_tokenize(self, text):
        """ Cleans and tokenizes the given text. """
        text = text.lower()

        # Remove various alphanumeric patterns
        text = re.sub(r'\b_+[a-z0-9]+\b', ' ', text)  # Remove patterns starting with underscore(s)
        text = re.sub(r'\b\d+[a-z][a-z0-9_]*\b', ' ', text)  # Remove patterns like '003dl0nlraie1'
        text = re.sub(r'\b[a-z0-9]+_[a-z0-9_]+\b', ' ', text)  # Remove patterns with underscores
        text = re.sub(r'\b\d+s\b', ' ', text)  # Remove patterns like '000s'
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'\b[0-9a-f]+\b', ' ', text)  # Remove hexadecimal numbers
        text = re.sub(r'\b_*\d+\b', ' ', text)  # Remove numbers with optional leading underscores
        text = re.sub(r'_\w+', ' ', text)  # Remove any word that starts with underscore
        text = re.sub(r'_{1,}', ' ', text)  # Remove underscores (one or more)
        text = re.sub(r'\W+', ' ', text)  # Remove remaining special characters

        words = word_tokenize(text)
        combined_words_to_remove = set(self.stop_words + self.words_to_rem)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in combined_words_to_remove]  # Stopword removal & Lemmatization
        return words

    def update_corpus(self, new_posts: List[PostDataStructure]):
        """ Updates corpus with new text entries. """
        new_posts = [new_posts] if not isinstance(new_posts, list) else new_posts
    
        # Handle the text extraction
        if len(new_posts) > 1:
            all_texts_from_new_post = []
            for post in new_posts:
                all_texts_from_new_post.extend([post.selftext] + post.comments)
        else:
            post = new_posts[0]
            all_texts_from_new_post = [post.selftext] + post.comments
        
        self.texts.extend(all_texts_from_new_post)  # Store raw text
        self.tokenized_texts = [self.clean_and_tokenize(text) for text in self.texts]  # Preprocessed texts
        self.dictionary = Dictionary(self.tokenized_texts)  # Create Gensim dictionary
        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokenized_texts]  # Convert to Bag-of-Words

        print("\n📢 Corpus Updated Successfully!\n")

    def get_tokenized_texts(self):
        """ Returns tokenized texts for Sentiment Analysis. """
        return self.tokenized_texts

    def get_gensim_data(self):
        """ Returns dictionary and corpus for Topic Modeling with Gensim. """
        return self.dictionary, self.corpus
