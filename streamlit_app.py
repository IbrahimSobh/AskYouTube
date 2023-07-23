import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader


# https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/youtube_audio

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

def generate_response(youtube_url, google_api_key, query_text):
    #Directory to save audio files
    save_dir = "./youtube"

    try:
        # Use the YoutubeLoader to load and parse the transcript of a YouTube video
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        docs = loader.load()
    
        # Combine doc
        combined_docs = [doc.page_content for doc in docs]
        text = " ".join(combined_docs)
        
        # Split them
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32, separators=["\n\n", "\n", ",", " ", "."])
        pages = text_splitter.create_documents([text])

        # Select embeddings
        embeddings = GooglePalmEmbeddings(google_api_key=google_api_key)
        
        # Create a vectorstore from documents
        db = Chroma.from_documents(pages, embeddings) 
        
        # Create retriever interface
        retriever = db.as_retriever(k=3)
        # retriever = db.as_retriever(k=2, fetch_k=4)
        # retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .9})
    
    except:
        st.write("An error occurred")
        return   
    
    # Create QA chain
    #qa = RetrievalQA.from_chain_type(llm=GooglePalm(google_api_key=google_api_key, temperature=0.1, max_output_tokens=128), chain_type="stuff", retriever=retriever)
    qa = RetrievalQA.from_chain_type(llm=GooglePalm(google_api_key=google_api_key, temperature=0.1, max_output_tokens=128),
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    try:
      res = qa({"query": query_text})
      return res  
    except:
      st.write("An error occurred")

    return


# Page title
st.set_page_config(page_title='Ask YouTube 🎥 via PaLM🌴 Model , LangChain 🦜🔗 and Chroma vector DB. By: Ibrahim Sobh')
st.title('Ask YouTube 🎥 via PaLM🌴 Model , LangChain 🦜🔗 and Chroma vector DB. By: Ibrahim Sobh')

# File upload
#uploaded_file = st.file_uploader('Upload text file', type='txt')
# uploaded_file = st.file_uploader('Upload pdf file', type='pdf')
youtube_url = st.text_input('Enter YouTube url:', placeholder = 'https://youtu.be/kCc8FmEb1nY')

# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not youtube_url)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    google_api_key = st.text_input('Google PaLM🌴 API Key', type='password', disabled=not (youtube_url and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(youtube_url and query_text))
    if submitted and google_api_key:
        with st.spinner('Calculating (This may take a couple of minutes depending on the video) ...'):
            response = generate_response(youtube_url, google_api_key, query_text)
            if response:
                result.append(response)
            del google_api_key


if len(result):
    st.markdown('**Answer:** **:blue[' + response['result'] + "]**")
    st.markdown('---')
    st.markdown('**References:** ')
    for i, sd in enumerate(response['source_documents']):
        st.markdown('**Ref ' + str(i) + '** :green[' + sd.page_content[:70] + "... ]")   
