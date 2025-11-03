import openai
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter as CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain


from htmlTemplate import css, bot_template, user_template
#from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def get_summary(prompt):
    messages = [{"role": "user", "content": prompt}]
    llm = openai()
    response = openai.ChatCompletion.create(
        model= llm,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]

def main():
    load_dotenv()
    st.set_page_config(page_title="UniBuddy - Chat with your PDF",)
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your syllabus PDFs")
    user_question = st.text_input("Ask question from your PDFs : ")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        
        if st.button("Start Processing"):
            raw_text = get_pdf_text(pdf_docs)
            
        if st.button("Chat"):
            with st.spinner("Chatting..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

        if st.button("Summarise"):
            with st.spinner("Summarising..."):
                raw_text = get_pdf_text(pdf_docs)
                prompt =  f"""
                            Your task is to generate a short summary of educational \
                            content from an educational PDF.

                            Summarize the review below, delimited by triple
                            backticks covering all key points of the PDF

                            Review: ```{raw_text}```
                        """
                summarize = get_summary(prompt)
                st.write(bot_template.replace("{{MSG}}", summarize.content), unsafe_allow_html=True)
            
if __name__ == '__main__':
    main()

# import openai
# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter as CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_community.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplate import css, bot_template, user_template



# # ---------- Extract PDF Text ----------
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             content = page.extract_text()
#             if content:
#                 text += content
#     return text


# # ---------- Split Text ----------
# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# # ---------- Create Vector Store ----------
# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# # ---------- Create Conversation Chain ----------
# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI(model="gpt-3.5-turbo")
#     retriever = vectorstore.as_retriever()

#     prompt = ChatPromptTemplate.from_template("""
#     Use the following context to answer the user's question.
#     If the answer is not available in the context, say "I don't know."

#     Context: {context}
#     Question: {input}
#     """)

#     combine_docs_chain = create_stuff_documents_chain(llm, prompt)
#     conversation_chain = create_retrieval_chain(retriever, combine_docs_chain)

#     return conversation_chain


# # ---------- Handle Chat Messages ----------
# def handle_userinput(user_question):
#     response = st.session_state.conversation.invoke({"input": user_question})
#     st.session_state.chat_history.append({"user": user_question, "bot": response["answer"]})

#     for chat in st.session_state.chat_history:
#         st.write(user_template.replace("{{MSG}}", chat["user"]), unsafe_allow_html=True)
#         st.write(bot_template.replace("{{MSG}}", chat["bot"]), unsafe_allow_html=True)


# # ---------- Summarization ----------
# def get_summary(prompt):
#     client = OpenAI()  # ✅ use new OpenAI client
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0,
#     )
#     return response.choices[0].message.content  # ✅ updated access


# # ---------- Streamlit App ----------
# def main():
#     load_dotenv()
#     st.set_page_config(page_title="UniBuddy - Chat with your PDF")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     st.header("Chat with your syllabus PDFs")
#     user_question = st.text_input("Ask question from your PDFs : ")
#     if user_question and st.session_state.conversation:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        
#         if st.button("Start Processing"):
#             raw_text = get_pdf_text(pdf_docs)
            
#         if st.button("Chat"):
#             with st.spinner("Chatting..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 vectorstore = get_vectorstore(text_chunks)
#                 st.session_state.conversation = get_conversation_chain(vectorstore)
#                 st.success("You can now chat with your PDFs!")

#         if st.button("Summarise"):
#             with st.spinner("Summarising..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 prompt = f"""
#                 Your task is to generate a short summary of educational
#                 content from an educational PDF.

#                 Summarize the review below, delimited by triple
#                 backticks covering all key points of the PDF.

#                 Review: ```{raw_text}```
#                 """
#                 summarize = get_summary(prompt)
#                 st.write(bot_template.replace("{{MSG}}", summarize), unsafe_allow_html=True)
            

# if __name__ == '__main__':
#     main()
