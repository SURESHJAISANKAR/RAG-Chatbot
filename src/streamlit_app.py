import streamlit as st
from main import RAGPipeline
from llm import generate_answer

st.title("📄 RAG Q&A System")

# Input box
query = st.text_input("Enter your question")

if st.button("Submit"):
    from main import RAGPipeline
    rag = RAGPipeline()
    rag.load() 
    if query.strip() == "":
        st.warning("Please enter a question")
    else:
        context = rag.query(query)
        answer = generate_answer(query, context)
        st.subheader("Answer:")
        st.write(answer)

        # with st.expander("Retrieved Context"):
        #     st.write(context)
        # st.text_input = ''