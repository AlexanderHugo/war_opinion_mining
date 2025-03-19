from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import numpy as np

class VaderSentimentAnalyzer:
    def __init__(self, corpus_manager):
        """
        Initializes the VADER Sentiment Analyzer with shared data.
        :param corpus_manager: The shared CorpusManager instance.
        """
        self.corpus_manager = corpus_manager
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        """ Analyzes sentiment of the given text. """
        tokens = self.corpus_manager.clean_and_tokenize(text)
        joined_text = ' '.join(tokens)
        sentiment_scores = self.analyzer.polarity_scores(joined_text)
        compound = sentiment_scores["compound"]
        pos = sentiment_scores["pos"]
        neu = sentiment_scores["neu"]
        neg = sentiment_scores["neg"]
        classifier = "Positive" if compound >= 0.5 else "Negative" if compound <= -0.5 else "Neutral"
        return classifier, sentiment_scores

    def analyze_overall_post_sentiment(self, text_post, comments, ratio=0.5):
        """ Creates a compound score for all the comments in a post. """
        _, main_post_sentiment_scores = self.analyze_sentiment(text_post)

        comment_scores = []
        for comment in comments:
            _, scores = self.analyze_sentiment(comment)
            comment_scores.append(scores)

        num_items = len(comment_scores)

        if num_items == 0:
            classifier = "Positive" if main_post_sentiment_scores["compound"] >= 0.5 else "Negative" if main_post_sentiment_scores["compound"] <= -0.5 else "Neutral"       
            return classifier, main_post_sentiment_scores

        comments_compound = sum(x["compound"] for x in comment_scores) / num_items
        comments_pos = sum([x["pos"] for x in comment_scores]) / num_items
        comments_neu = sum([x["neu"] for x in comment_scores]) / num_items
        comments_neg = sum([x["neg"] for x in comment_scores]) / num_items

        overall_compound = main_post_sentiment_scores["compound"] * ratio + comments_compound * (1 - ratio)
        overall_pos = main_post_sentiment_scores["pos"] * ratio + comments_pos * (1 - ratio)
        overall_neu = main_post_sentiment_scores["neu"] * ratio + comments_neu * (1 - ratio)
        overall_neg = main_post_sentiment_scores["neg"] * ratio + comments_neg * (1 - ratio)

        classifier = "Positive" if overall_compound >= 0.5 else "Negative" if overall_compound <= -0.5 else "Neutral"       
        return classifier, {
            "neg": overall_neg,
            "neu": overall_neu,
            "pos": overall_pos,
            "compound": overall_compound
        }



       