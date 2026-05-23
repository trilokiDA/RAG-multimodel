import streamlit as st
import os
from main import setup_rag_pipeline, build_graph

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multimodal RAG Assistant")
st.markdown("Ask questions about your data, and I'll provide answers with relevant charts!")

# Initialize the pipeline only once
if "app" not in st.session_state:
    with st.spinner("Setting up RAG Pipeline... this may take a minute."):
        try:
            retrieval_pipeline = setup_rag_pipeline()
            st.session_state.app = build_graph(retrieval_pipeline)
            st.success("RAG System Ready!")
        except Exception as e:
            st.error(f"Error setting up pipeline: {e}")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img_path in message["images"]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Source: {os.path.basename(img_path)}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            inputs = {"question": prompt}
            # We use the compiled app from session state
            result = st.session_state.app.invoke(inputs)
            
            # The result from our graph contains 'generation' and 'documents'
            full_response = result["generation"]
            
            # Extract image paths for st.image
            image_paths = []
            for doc in result["documents"]:
                if doc.metadata.get("is_multimodal") and doc.metadata.get("image_path"):
                    image_paths.append(doc.metadata.get("image_path"))
            
            # Remove duplicates
            image_paths = list(set(image_paths))

            # Clean the text response (remove the "Relevant Visuals Found" text since we'll show them as st.image)
            clean_response = full_response.split("## 🖼️ Relevant Visuals Found")[0]
            
            st.markdown(clean_response)
            
            # Display images if found
            found_images = []
            if image_paths:
                st.subheader("📊 Relevant Charts & Visuals")
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"Source: {os.path.basename(img_path)}")
                        found_images.append(img_path)
            
            # Save assistant message to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": clean_response,
                "images": found_images
            })
