import streamlit as st
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Page Configuration
st.set_page_config(page_title="Ekana AI - Phishing Threat Intelligence", layout="wide")
st.title("🛡️ Ekana AI: Real-Time Phishing Threat Intelligence")
st.markdown("Analyze suspicious SMS or Email messages against the live vector database.")

st.sidebar.markdown("### Project Details")
st.sidebar.markdown("**Pipeline:** PySpark + Redpanda + ChromaDB")
st.sidebar.markdown("**Lead Developer:** Sani Yadav")

# 2. Load the AI Model and Database Connection (Cached for speed)
@st.cache_resource
def load_model_and_db():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    collection = chroma_client.get_or_create_collection(name="phishing_signatures")
    return model, collection

model, collection = load_model_and_db()

# 3. User Interface
st.subheader("Semantic Threat Search")
suspicious_message = st.text_area(
    "Paste the suspicious message here:", 
    height=150, 
    placeholder="e.g., Your Amazon account is locked. Click here to update your billing info..."
)

if st.button("Analyze Threat", type="primary"):
    if suspicious_message.strip():
        with st.spinner("Analyzing semantics and querying vector database..."):
            # Convert the input to a mathematical vector using the exact same AI model
            query_vector = model.encode([suspicious_message], normalize_embeddings=True).tolist()
            
            # Search ChromaDB for the top 3 closest semantic matches
            results = collection.query(
                query_embeddings=query_vector,
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            
            if results['distances'] and results['distances'][0]:
                st.markdown("### 🔍 Threat Analysis Results")
                
                # Process and display the results
                for i in range(len(results['distances'][0])):
                    distance = results['distances'][0][i]
                    # Convert cosine distance to a similarity percentage
                    similarity_score = (1 - distance) * 100 
                    
                    doc = results['documents'][0][i]
                    meta = results['metadatas'][0][i]
                    
                    # Display as an expandable card
                    with st.expander(f"Match {i+1} | Semantic Similarity: {similarity_score:.2f}% | Known Label: {meta['label']}"):
                        st.write(f"**Database Signature:** {doc}")
                        st.write(f"**Source:** {meta['source']}")
                        st.write(f"**Timestamp Indexed:** {meta['timestamp']}")
            else:
                st.info("No signatures found in the database. Ensure the PySpark stream is running and data is injected.")
    else:
        st.warning("Please enter a message to analyze.")
