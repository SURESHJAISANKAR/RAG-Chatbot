import streamlit_app as st
from main import RAGPipeline
from llm import generate_answer


rag = RAGPipeline()
rag.load() 

st.title("📄 RAG Q&A System")

# Input box
query = st.text_input("Enter your question")

if st.button("Submit"):

    if query.strip() == "":
        st.warning("Please enter a question")
    else:
        context = rag.query(query)

        answer = generate_answer(query, context)

        st.subheader("Answer:")
        st.write(answer)


        with st.expander("Retrieved Context"):
            st.write(context)