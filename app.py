import streamlit as st
import numpy as np
import os
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
import time
import pickle
from typing import List, Tuple
from dataclasses import dataclass
from mongo import WarOpMongoDB
from corpus import CorpusManager
from gensimLDA import GensimLDA
from vaderSentimentAnalysis import VaderSentimentAnalyzer
from graph import Graph
from shared_state import get_state_version, load_state
from dataStructures import PostDataStructure, EnhancedPostDataStructure
import pathlib

# Constants
UPDATE_INTERVAL = 10 * 60  # seconds
CACHE_KEY = "war_op_data"

@dataclass
class SessionState:
    graph: Graph
    lda: GensimLDA
    corpus_manager: CorpusManager
    db: WarOpMongoDB
    last_update: float = 0
    version: int = 0

@st.cache_resource
def initialize_session(use_saved_model=False) -> SessionState:
    get_cached_graph_html.clear()  # Clear the graph HTML cache
    graph = Graph()
    _, lda, corpus_manager, db = process_data(use_saved_model)

    # Create the initial graph
    tokenized_texts = corpus_manager.get_tokenized_texts()
    graph.create_network_graph(tokenized_texts, min_weight=100)

    return SessionState(graph, lda, corpus_manager, db)

@st.cache_data
def get_cached_graph_html(nodes, edges, node_sizes, min_weight=10, n_components=None):
    graph = Graph.from_graph_data({'nodes': nodes, 'edges': edges, 'node_sizes': node_sizes})
    return graph.prune_graph(min_weight).draw_graph_pyvis(n_components=n_components)

def check_for_updates(session_state: SessionState) -> bool:
    current_version = get_state_version()
    if current_version > session_state.version:
        return True
    return False

def process_data(use_saved_model=False) -> Tuple[Graph, GensimLDA, CorpusManager, WarOpMongoDB]:
    db = WarOpMongoDB()
    corpus_manager = CorpusManager()
    lda = GensimLDA(corpus_manager)
    sentiment_analyzer = VaderSentimentAnalyzer(corpus_manager)

    all_db_posts: List[PostDataStructure] = db.findAllPosts()
    corpus_manager.update_corpus(all_db_posts)

    if use_saved_model == True and os.path.isfile('trained_lda_model.pkl'):
        with open('trained_lda_model.pkl', 'rb') as f:
            lda, corpus_manager = pickle.load(f)
    else:
        lda.train_gensim()
    
        with open('trained_lda_model.pkl', 'wb') as f:
            pickle.dump((lda, corpus_manager), f)
    
    graph = Graph()
    graph.create_network_graph(corpus_manager.get_tokenized_texts(), min_weight=100)

    for post in all_db_posts:
        topic = lda.predict_topic(article=post.selftext)
        sentiment, sentiment_probs = sentiment_analyzer.analyze_sentiment(post.selftext)
        overall_sentiment, overall_sentiment_probs = sentiment_analyzer.analyze_overall_post_sentiment(post.selftext, post.comments, ratio=0.7)
        
        enhanced_post = EnhancedPostDataStructure(
            id=post.id,
            title=post.title,
            upvote_ratio=post.upvote_ratio,
            author=post.author,
            created_utc=post.created_utc,
            score=post.score,
            url=post.url,
            selftext=post.selftext,
            num_comments=post.num_comments,
            comments=post.comments,
            sentiment_score={"label": sentiment, "probs": sentiment_probs},
            overall_sentiment_score={"label": overall_sentiment, "probs": overall_sentiment_probs},
            topic=topic
        )

        db.saveEnhancedPost(enhanced_post)

    return graph, lda, corpus_manager, db

def render_post_card(post: EnhancedPostDataStructure):
    sentiment = post.sentiment_score['label'].lower()
    st.html(f'<div class="{sentiment}">')
    with st.expander(f"{post.title} - [{post.sentiment_score['label']}]"):
        st.write(f"**ID:** {post.id}")
        st.write(f"**Title:** {post.title}")
        st.write(f"**Sentiment Score:** {post.sentiment_score['label']} (Probability: {post.sentiment_score['probs']['compound']:.2f})")
        st.write(f"**Overall Sentiment Score:** {post.overall_sentiment_score['label']} (Probability: {post.overall_sentiment_score['probs']['compound']:.2f})")
        st.write(f"**Topic:** {post.topic['topic_id']} (Probability: {post.topic['probability']:.2f})")
        st.write("**Keywords:**", ", ".join(post.topic['keywords']))
        st.write(f"**Text** {post.selftext}")
    st.html('</div>')

def render_post_cards(posts: List[EnhancedPostDataStructure]):
    col1, col2 = st.columns(2)
    for i, post in enumerate(posts):
        with col1 if i % 2 == 0 else col2:
            render_post_card(post)

def render_topic_distribution(db: WarOpMongoDB, lda: GensimLDA):
    enhanced_posts = db.getAllEnhancedPost()
    topic_counts = {}
    for post in enhanced_posts:
        topic_id = post.topic['topic_id']
        topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
    
    df = pd.DataFrame(list(topic_counts.items()), columns=['Topic', 'Count'])
    fig = px.bar(df, x='Topic', y='Count', title='Number of Documents per Topic')
    st.plotly_chart(fig)
    
    st.subheader("Top Keywords per Topic")
    for topic_id in topic_counts.keys():
        keywords = lda.get_topic_keywords(topic_id)
        st.write(f"**Topic {topic_id}:** {', '.join(keywords)}")

def main():
    st.set_page_config(page_title="War Operation Mining Dashboard", layout="wide")

    if CACHE_KEY not in st.session_state:
        st.session_state[CACHE_KEY] = initialize_session(use_saved_model=True)

    session_state = st.session_state[CACHE_KEY]
    
    # Check for updates
    if check_for_updates(session_state):
        st.cache_resource.clear()
        st.cache_data.clear()
        session_state = initialize_session(use_saved_model=True)
        session_state.version = get_state_version()
        st.session_state[CACHE_KEY] = session_state
        st.rerun()

    # Display last update information
    shared_state = load_state()
    st.sidebar.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(shared_state.get('last_update', 0)))}")
    st.sidebar.write(f"Total posts: {shared_state.get('total_posts', 0)}")
    st.sidebar.write(f"Version: {shared_state.get('version', 0)}")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Post Cards", "Dynamic Graph", "Topic Distribution"])

    if page == "Post Cards":
        st.title("Post Cards")
        enhanced_posts = session_state.db.getAllEnhancedPost()
        render_post_cards(enhanced_posts)

    elif page == "Dynamic Graph":
        st.title("Dynamic Word Co-occurrence Graph")
   
        col1, col2 = st.columns(2)        
        with col1:
            min_weight = st.slider("Minimum edge weight", 1, 1000, 100)
        
        with col2:
            # New slider for number of components
            max_components = session_state.graph.get_max_components_num()  # You might want to adjust this based on your data
            n_components = st.slider("Number of connected components to display", 1, max_components, max_components)
        
        # Get the graph data in a hashable format
        graph_data = session_state.graph.get_graph_data()
        
        # Use the cached function with the graph data
        graph_html = get_cached_graph_html(
            graph_data['nodes'],
            graph_data['edges'],
            graph_data['node_sizes'],
            min_weight,
            n_components
        )

        st.components.v1.html(graph_html, height=1200)

    elif page == "Topic Distribution":
        st.title("Topic Distribution")
        render_topic_distribution(session_state.db, session_state.lda)

if __name__ == "__main__":
    main()
