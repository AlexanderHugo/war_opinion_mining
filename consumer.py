import time
import pika
from graph import Graph
from mongo import WarOpMongoDB
import json
from gensimLDA import GensimLDA
from corpus import CorpusManager
import pickle
from dataStructures import EnhancedPostDataStructure, PostDataStructure
from vaderSentimentAnalysis import VaderSentimentAnalyzer
from shared_state import load_state, update_state


# Load the trained LDA model and corpus_manager
with open('trained_lda_model.pkl', 'rb') as f:
    lda, corpus_manager = pickle.load(f)

class RabbitMQConsumer:
    def __init__(self, callback, db, lda, corpus_manager) -> None:
        self.__host = "localhost"
        self.__port = 5672
        self.__username = "guest"
        self.__password = "guest"
        self.__queue = "mensagens_3"
        self.__callback = callback
        self.__channel = self.__create_channel()
        self.db = db
        self.lda = lda
        self.corpus_manager = corpus_manager

    def __create_channel(self):
        conections_parameters = pika.ConnectionParameters(
            host=self.__host,
            port=self.__port,
            credentials=pika.PlainCredentials(
                username=self.__username,
                password=self.__password
            )
        )

        channel = pika.BlockingConnection(conections_parameters).channel()

        # Criando a fila do tipo "stream"
        channel.queue_declare(
            queue=self.__queue,
            durable=True,
            arguments={'x-queue-type': 'stream'}
        )

        # ✅ Configurar PEFETCH COUNT (necessário para fila tipo Stream)
        channel.basic_qos(prefetch_count=10)  # Ajuste conforme o necessário

        # ✅ Consumir mensagens da fila
        channel.basic_consume(
            queue=self.__queue,
            auto_ack=False,  # Pode ser False se precisar de confirmação manual
            on_message_callback=self.__callback
        )

        return channel

    def start(self):
        print(f"Listening to RabbitMQ on port {self.__port}, queue: {self.__queue}")
        self.__channel.start_consuming()

message_count = 0
UPDATE_THRESHOLD = 10  # Update after every 20 new messages

def callback(ch, method, properties, body):
    global message_count
    bucket_posts: list[PostDataStructure] = []
    
    message = body.decode()
    try:
        json_message = json.loads(message)
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {message}")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    print(f"📩 Received: {json_message.get('id', 'No ID')}")
    existing_post = db.findPostById(json_message.get('id'))
    
    if existing_post is None:
        print(f" ID [{json_message.get('id', 'No ID')}] is new. Processing and adding to the database.")
        
        # Insert into MongoDB collection
        inserted_id = db.insert_new_post(json_message)
        
        if inserted_id:
            # Process the post
            id = json_message.get('id', '')
            title = json_message.get('title', '')
            upvote_ratio = json_message.get('upvote_ratio', 0.0)
            author = json_message.get('author', '')
            created_utc = json_message.get('created_utc', '')
            score = json_message.get('score', 0)
            url = json_message.get('url', '')
            selftext = json_message.get('selftext', '')
            num_comments = json_message.get('num_comments', [])
            comments = json_message.get('comments', [])
            
            topic = lda.predict_topic(article=selftext)
            sentiment, sentiment_probs = sentiment_analyzer.analyze_sentiment(selftext)
            overall_sentiment, overall_sentiment_probs = sentiment_analyzer.analyze_overall_post_sentiment(
                selftext, comments, ratio=0.7
            )
            
            # Create EnhancedPostDataStructure
            enhanced_post = EnhancedPostDataStructure(
                id=id,
                title=title,
                upvote_ratio=upvote_ratio,
                author=author,
                created_utc=created_utc,
                score=score,
                url=url,
                selftext=selftext,
                num_comments=num_comments,
                comments=comments,
                sentiment_score={
                    "label": sentiment,
                    "probs": sentiment_probs
                },
                overall_sentiment_score={
                    "label": overall_sentiment,
                    "probs": overall_sentiment_probs
                },
                topic=topic
            )
            
            # Save to local storage
            db.saveEnhancedPost(enhanced_post)
            
            message_count += 1

            bucket_posts.append(
                PostDataStructure(
                    id=id,
                    title=title,
                    upvote_ratio=upvote_ratio,
                    author=author,
                    created_utc=created_utc,
                    score=score,
                    url=url,
                    selftext=selftext,
                    num_comments=num_comments,
                    comments=comments
                )
            )
            
            if message_count >= UPDATE_THRESHOLD:
                print("Updating LDA model and Graph")
                lda.train_gensim()
                
                corpus_manager.update_corpus(bucket_posts)
                graph = Graph()
                graph.create_network_graph(corpus_manager.get_tokenized_texts(), min_weight=100)

                # Update shared state with version and timestamp
                current_state = load_state()
                current_version = current_state.get('version', 0) + 1
                
                update_state('version', current_version)
                update_state('last_update', time.time())
                update_state('total_posts', len(db.getAllEnhancedPost()))
                
                bucket_posts = []
                message_count = 0 # Reset counter
        else:
            print(f"Failed to insert post with ID: {json_message.get('id', 'No ID')}")
    else:
        print(f" ID [{json_message.get('id', 'No ID')}] already in the database.")

    # Manually acknowledge the message
    ch.basic_ack(delivery_tag=method.delivery_tag)

db = WarOpMongoDB()
sentiment_analyzer = VaderSentimentAnalyzer(corpus_manager)

# Criando e iniciando o consumidor
DBconsumer = RabbitMQConsumer(callback=callback, db=db, lda=lda, corpus_manager=corpus_manager)
DBconsumer.start()