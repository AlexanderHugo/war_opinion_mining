import gensim
from gensim.models import CoherenceModel
import pickle

class GensimLDA:
    def __init__(self, corpus_manager, num_topics=5):
        """
        Initializes LDA model with a shared corpus.
        :param corpus_manager: The shared CorpusManager instance.
        :param num_topics: Number of topics for LDA.
        """
        self.corpus_manager = corpus_manager
        self.num_topics = num_topics
        self.lda_model = None
        self.coherence_model = None

    def train_gensim(self):
        """ Fetch latest corpus from CorpusManager and train the LDA model. """
        dictionary, corpus = self.corpus_manager.get_gensim_data()
        if corpus and dictionary:
            self.lda_model = gensim.models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=self.num_topics,
                random_state=100,
                update_every=1,
                chunksize=100,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )
            self.coherence_model = CoherenceModel(model=self.lda_model, texts=self.corpus_manager.get_tokenized_texts(), dictionary=dictionary, coherence='c_v', processes=1)
            print("✅ LDA Model trained successfully!")
        else:
            print("⚠ No sufficient data to train LDA. Please update the corpus.")

    def update_model(self, new_article):
        """
        Updates the LDA model with a new article.
        :param new_article: New text to be included in the LDA model.
        """
        new_article = self.corpus_manager.clean_and_tokenize(new_article)
        bow_new_article = [self.corpus_manager.dictionary.doc2bow(new_article)]
        self.lda_model.update(bow_new_article)

    def get_topic_keywords(self, topic_id, num_words=5):
        try:
            # Get topic terms with probabilities
            topic_terms = self.lda_model.show_topic(topic_id, num_words)
            # Extract just the words (without probabilities)
            keywords = [word for word, prob in topic_terms]
            return keywords
        
        except Exception as e:
            print(f"⚠ Error getting keywords for topic {topic_id}: {str(e)}")
            return []

    def predict_topic(self, article):
        """
        Predicts the topic of a given article.
        :param article: Article to be predicted by the LDA model.
        """
        bow_article = self.corpus_manager.dictionary.doc2bow(self.corpus_manager.clean_and_tokenize(article))
        topic_predictions = self.lda_model[bow_article]

        dominant_topic = max(topic_predictions[0], key=lambda x: x[1])
        return {
            "topic_id": dominant_topic[0],
            "probability": dominant_topic[1],
            "keywords": self.get_topic_keywords(dominant_topic[0])
        }

    def save(self, filename):
        """ Saves the LDA model to a file. """
        with open(filename, 'wb') as f:
            pickle.dump({'dictionary': self.corpus_manager.dictionary, 'lda_model': self.lda_model}, f)

    def load_model(self, filename):
        """ Loads the LDA model from a file. """
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.corpus_manager.dictionary = data['dictionary']
            self.lda_model = data['lda_model']

    def print_topics(self, num_words=5):
        """ Prints extracted topics from the trained LDA model. """
        if self.lda_model:
            for idx, topic in self.lda_model.show_topics(num_topics=self.num_topics, num_words=num_words, formatted=False):
                print(f"Topic {idx}: {[word for word, _ in topic]}")
        else:
            print("⚠ No trained LDA model found.")

    def get_coherence(self):
        """ Returns the coherence score of the LDA model. """
        return self.coherence_model.get_coherence()

    def get_perplexity(self):
        """ Returns the perplexity of the LDA model. """
        return self.lda_model.log_perplexity(self.corpus_manager.corpus)
