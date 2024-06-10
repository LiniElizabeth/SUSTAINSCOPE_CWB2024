import streamlit as st
import openai
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from dotenv import load_dotenv
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import AzureAISearchRetriever
import seaborn as sns



#Loading Environment Variables
load_dotenv()

#Obtaining Azure AI Search Credentials 
AZURE_AI_SEARCH_SERVICE_NAME=os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
AZURE_AI_SEARCH_ENDPOINT=os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_AI_SEARCH_API_KEY=os.getenv("AZURE_AI_SEARCH_API_KEY")  
AZURE_AI_SEARCH_INDEX_NAME=os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
search_client = SearchClient(endpoint=AZURE_AI_SEARCH_ENDPOINT, index_name=AZURE_AI_SEARCH_INDEX_NAME, credential=AzureKeyCredential("AZURE_AI_SEARCH_API_KEY"))

#Obtaining OpenAI Credentials 
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

#Initialise OpenAI Model
llm = OpenAI(temperature=0.9, max_tokens=500, model = "gpt-3.5-turbo-instruct")
#Initialise OpenAI Embeddings Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Initialising the Azure AI Search retriever
retriever = AzureAISearchRetriever(content_key="content", top_k=1, index_name=AZURE_AI_SEARCH_INDEX_NAME,service_name=AZURE_AI_SEARCH_SERVICE_NAME, api_key=AZURE_AI_SEARCH_API_KEY)

#Initialising the vector store using langchain

vector_store= AzureSearch(
    azure_search_endpoint=AZURE_AI_SEARCH_ENDPOINT,
    azure_search_key=AZURE_AI_SEARCH_API_KEY,
    index_name=AZURE_AI_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
)
# Function to generate comparison report
def generate_comparison_report():
    documents = search_client.search(search_text='*')
    company_reports = [doc for doc in documents if doc["document_type"] == "Company Report"]
    news_articles = [doc for doc in documents if doc["document_type"] == "News Article"]
    
    company_report_content = " ".join([doc["content"] for doc in company_reports])
    news_article_content = " ".join([doc["content"] for doc in news_articles])
    messages=[
    {"role": "system", "content": f"""You are a sustainability analyst.Compare the company's sustainability claims
                                   regarding achieving various sustainability goals in the following company report: 
                                   {company_report_content} with the actual actions reported in the following news 
                                   articles: {news_article_content}. Provide a detailed analysis and discuss the key 
                                   points to be noted regarding their sustainability progress. Use only the data provided
                                   to you in this message"""},
    {"role": "user", "content": "Hello!"}
    ]
    
    
    response = openai.chat.completions.create(model="gpt-4o",messages=messages, max_tokens=1024)
    
    return response.choices[0].message.content

#Functions for Chat with Sustainability Report Action
def get_conversational_chain():

    prompt_template = """
    You are a sustainability analyst with great expertise.Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Sorry,this information is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):

    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents":docs, "question": user_question},return_only_outputs=True)
    return response["output_text"]

def sentiment_analysis_and_word_cloud(company):

    filter_query =f"company_name eq '{company}' and document_type eq 'NEWS MEDIA'"
    news_documents = search_client.search(filter=filter_query,select=["content"])
     # Populate the DataFrame with sentiment analysis
    df = pd.DataFrame(columns=["ID", "Published Date","Sentiment"])
    for index, doc in enumerate(news_documents):

        text=doc["content"] 
        template = f"Perform sentiment analysis of given text, and assign a positive value for positive tone between (0,1)and negative value between (0,-1)for negative tone: {text}"
        prompt=PromptTemplate(input_variables=doc,template=template)
        chain=load_qa_chain(llm=llm,prompt=prompt)
        sentiment=chain.run(document=doc)
        df.loc[index] = [index, doc["publish_date"], sentiment]
        
    # Visualize sentiment analysis using seaborn
    sns.set_style("whitegrid")
    sns.displot(df["Sentiment"], height=5, aspect=1.8)
    plt.xlabel("Sentiment")
    plt.title("Sentiment Analysis")
    st.pyplot()  # Display the plot in Streamlit

    # Generate Wordcloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(news_documents["content"])
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    st.pyplot(plt)


import streamlit as st


# Streamlit app
icon_path = "logo.jpg"
st.set_page_config(page_title="SUSTAINSCOPE", page_icon=icon_path,
                       layout='centered', initial_sidebar_state="collapsed")
_,_, logo,_, _ = st.columns(5)
logo.image(icon_path, width=60)
style = ("text-align:center; padding: 0px; font-family: arial black;, "
             "font-size: 250%")
title = f"<h1 style='{style}'>SUSTAINSCOPE</h1><br><br>"
st.write(title, unsafe_allow_html=True)

# Disclaimer at the bottom
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: black;
        text-align: center;
        padding: 10px;
        border-top: 0px solid #e9ecef;
    }
    </style>
    <div class="footer">
        <p style='font-style: italic;'>
        Disclaimer: This application uses AI-generated content for insights. While we strive for accuracy, please verify findings with primary sources.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Centered selection boxes
company = st.selectbox("Select the Company for Analysis", ["Choose an option","RIO TINTO", "BHP"], key='company')
year = st.selectbox("Select Financial Year", ["Choose an option","2023", "2022"], key='year')

#Initialising session_state
if 'clicked' not in st.session_state:
    st.session_state.clicked = False
def click_button():
    st.session_state.clicked = True

#Select Feature to view the news media sentiment analysis and word cloud
st.button('NEWS MEDIA SENTIMENT ANALYSIS AND WORD CLOUD', on_click=click_button)

if st.session_state.clicked:
    sentiment_analysis_and_word_cloud(company)
    



#Select Feature to chat with sustainability report
st.button('CHAT WITH SUSTAINABILITY REPORT', on_click=click_button)

if st.session_state.clicked:
    # The message and nested widget will remain on the page
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        response=user_input(user_question)
        st.write("Reply: ", response)

#Select Feature to Analyse Sustainability Progress
st.button('ANALYSE SUSTAINABILITY PROGRESS', on_click=click_button)

if st.session_state.clicked:
    # The message and nested widget will remain on the page
    report=generate_comparison_report()
    st.write(report)
    
    
