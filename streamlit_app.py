import streamlit as st
import pickle
from graph import Graph
from mongo import WarOpMongoDB
from pyvis.network import Network
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

# Load the trained LDA model and corpus_manager
with open('trained_lda_model.pkl', 'rb') as f:
    lda, corpus_manager = pickle.load(f)

# Initialize MongoDB connection
db = WarOpMongoDB()

# Streamlit app
st.set_page_config(page_title="War Operation Mining Dashboard", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Post Cards", "Dynamic Graph", "Topic Distribution"])

if page == "Post Cards":
    st.title("Post Cards")
    
    # Get all enhanced posts
    enhanced_posts = db.getAllEnhancedPost()
    
    for post in enhanced_posts:
        with st.expander(f"Post: {post.id}"):
            st.write(f"**Title:** {post.title}")
            st.write(f"**Sentiment Score:** {post.sentiment_score['label']} (Probability: {post.sentiment_score['probs']:.2f})")
            st.write(f"**Overall Sentiment Score:** {post.overall_sentiment_score['label']} (Probability: {post.overall_sentiment_score['probs']:.2f})")
            st.write(f"**Topic:** {post.topic['topic_id']} (Probability: {post.topic['probability']:.2f})")
            st.write("**Keywords:**", ", ".join(post.topic['keywords']))

elif page == "Dynamic Graph":
    st.title("Dynamic Word Co-occurrence Graph")
    
    # Create graph
    graph = Graph(corpus_manager.get_tokenized_texts())
    G = graph.create_network_graph(min_weight=50)
    
    # Convert to PyVis network
    net = Network(notebook=True, width="100%", height="600px", bgcolor="#222222", font_color="white")
    for node in G.nodes():
        net.add_node(node, label=node)
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], value=edge[2]['weight'])
    
    # Save and display the graph
    net.save_graph("graph.html")
    HtmlFile = open("graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=600)

elif page == "Topic Distribution":
    st.title("Topic Distribution")
    
    # Get all enhanced posts
    enhanced_posts = db.getAllEnhancedPost()
    
    # Count documents per topic
    topic_counts = {}
    for post in enhanced_posts:
        topic_id = post.topic['topic_id']
        if topic_id in topic_counts:
            topic_counts[topic_id] += 1
        else:
            topic_counts[topic_id] = 1
    
    # Create DataFrame for plotting
    df = pd.DataFrame(list(topic_counts.items()), columns=['Topic', 'Count'])
    
    # Create bar chart
    fig = px.bar(df, x='Topic', y='Count', title='Number of Documents per Topic')
    st.plotly_chart(fig)
    
    # Display top keywords for each topic
    st.subheader("Top Keywords per Topic")
    for topic_id in topic_counts.keys():
        keywords = lda.get_topic_keywords(topic_id)
        st.write(f"**Topic {topic_id}:** {', '.join(keywords)}")
