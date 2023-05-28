import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import textwrap
import pytube
import whisper
import datetime
from pytube import YouTube
from transformers import GPT2TokenizerFast
from openai_helper import *


st.set_page_config(page_title="Predizify - LLM Assistant", page_icon=":tada:", layout="wide")

# Create a menu with different page options
st.sidebar.title("**Predizify - LLM Assistant**")

menu_model = ["davinci", "curie", "ada","bloom"]
choice_model = st.sidebar.selectbox("**Select Base Model**", menu_model)

menu_completions_model = ["text-davinci-003", "text-curie-001", "text-ada-001","curie-instruct-beta","davinci-instruct-beta","gpt-3.5-turbo-0301"]
choice_completions_model = st.sidebar.selectbox("**Select Completions Model**", menu_completions_model)

menu_service = ["PDF Assistant", "Video Assistant", "Earnings Call Assistant", "CSV Assistant"]
choice_service = st.sidebar.selectbox("**Select Assistant**", menu_service)
document_embeddings = None

# Show the appropriate page based on the user's menu choice
if choice_service == "PDF Assistant":
        with st.container():
            st.title("PDF Assistant by Geoffrey Porto")
            st.write("Upload a PDF and ask questions about the document")
            st.write(
            """
            **Steps:**
            1. Upload a PDF
            2. Compute document embeddings
            3. Ask questions & get summaries of the document

            """)

        st.sidebar.subheader("Parameters")
        temperature = st.sidebar.slider('Temperature controls how much randomness or "creativity" is in the output.', 0.0, 1.0, 0.0)
        max_length = st.sidebar.slider('The maximum token length to use in the response.', 0, 1000, 500)

        # create a file upload component and prompt the user to upload a PDF file
        file = st.file_uploader("Upload PDF file", type="pdf")


        if file is not None:
            # read the contents of the uploaded file using PyPDF2
            pdf_reader = PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            # split the text into paragraphs
            #paragraphs = text.split('\n')
            paragraphs = textwrap.wrap(text, width=500)

            # create a DataFrame with a single column 'content' containing each paragraph as a separate row
            data = [{'content': paragraph} for paragraph in paragraphs]
            df = pd.DataFrame(data)
            # add a 'title' column to the DataFrame
            df['title'] = "Upload"
            # add a 'heading' column to the DataFrame
            df['heading'] = range(1, len(df) + 1)
            # add a 'tokens' column to the DataFrame
            df['tokens'] = df['content'].str.len() / 4
            df = df[df['tokens'] >= 5]
            df = df.reindex(columns=['title', 'heading', 'content', 'tokens'])
            df.to_csv('sample_df.csv', index=False, escapechar='\\')
            st.success("File uploaded")
            # create a button to compute embeddings
            if st.button("Compute embeddings"):
                context_embeddings = compute_doc_embeddings(df)
                df_embeds = pd.DataFrame(context_embeddings).transpose()
                df_embeds.to_csv('sample_embeddings.csv', index=False)
                st.success("Embeddings completed")

            with st.container():
                st.write("---")
                st.subheader("Ask questions about the document")
                label = "Enter your question below"
                prompt = st.text_input(label, key="prompt_input", placeholder=None)
                temperature = temperature
                max_length = max_length
                choice_model = choice_model
                choice_completions_model = choice_completions_model
                if st.button("Generate answer"):
                    document_embeddings, new_df = load_embeddings('sample_embeddings.csv', 'sample_df.csv')
                    if choice_model == "bloom":
                        choice_completions_model = ""
            
                    prompt_response = answer_query_with_context(prompt, new_df, document_embeddings, False, temperature, max_length, choice_completions_model)
                    
                    st.success("Predizify response")
                    st.write(prompt_response)

elif choice_service == "Video Assistant":

    # Header Section
    with st.container():
        
        st.title("YouTube Video Assistant")
        st.write("Ask questions and get summaries about any YouTube video")
        st.write(
        """
        **Steps:**
        1. Enter YouTube video & retrieve transcript
        2. Compute transcript embeddings
        3. Ask questions & get summaries of the video
        """)
        st.sidebar.subheader("Parameters")
        temperature = st.sidebar.slider('Temperature controls how much randomness or "creativity" is in the output.', 0.0, 1.0, 0.0)
        max_length = st.sidebar.slider('The maximum token length to use in the response.', 0, 1000, 500)

        # Create a text input component and prompt the user to enter a YouTube video link
        st.write("**Step 1:**")
        url = st.text_input("Enter YouTube video link", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder="Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        if st.button("Get video transcript"):
            if url != "":
                # Download the YouTube video using pytube
                youtube_video = YouTube(url)
                streams = youtube_video.streams.filter(only_audio=True)
                stream = streams.first()
                stream.download(filename='video.mp4')

                # Load the Whisper model
                model = whisper.load_model("base")
                output = model.transcribe("video.mp4")
                #st.write(output)

                # split the text into paragraphs
                #paragraphs = output['text'].split('\n')
                paragraphs = textwrap.wrap(output['text'], width=500)

                # create a DataFrame with a single column 'content' containing each paragraph as a separate row
                data = [{'content': paragraph} for paragraph in paragraphs]
                df = pd.DataFrame(data)
                # add a 'title' column to the DataFrame
                df['title'] = "Upload"
                # add a 'heading' column to the DataFrame
                df['heading'] = range(1, len(df) + 1)
                # add a 'tokens' column to the DataFrame
                df['tokens'] = df['content'].str.len() / 4
                #df = df[df['tokens'] >= 5]
                df = df.reindex(columns=['title', 'heading', 'content', 'tokens'])
                df.to_csv('sample_video.csv', index=False, escapechar='\\')
                st.success("Transcript completed.")
                # display the DataFrame in the app
             #   st.write(df)
            # create a button to compute embeddings
        st.write("**Step 2:**")
        if st.button("Compute embeddings"):
            df = pd.read_csv('sample_video.csv')
            context_embeddings = compute_doc_embeddings(df)
            df_embeds = pd.DataFrame(context_embeddings).transpose()
            df_embeds.to_csv('sample_embeddings_video.csv', index=False)
            st.success("Embeddings completed.")

            #
        st.write("**Step 3:**")
    #    st.write("Ask questions about the video")
        label = "Ask questions or write summaries about the video"
        prompt = st.text_input(label, key="prompt_input", placeholder=None)
        temperature = temperature
        max_length = max_length
        if st.button("Generate answer"):              
                document_embeddings, new_df = load_embeddings('sample_embeddings_video.csv', 'sample_video.csv')
                prompt_response = answer_query_with_context(prompt, new_df, document_embeddings, max_length, temperature)
                st.success("MLQ response")
                st.write(prompt_response)

    # Add content for page 3 here
elif choice_service == "Earnings Call Assistant":
    # Header Section
    with st.container():
    #    st.subheader("Hi, welcome to my site")
        st.title("MLQ Earnings Call Assistant")
        st.write("Retrieve earnings call transcript and ask questions about the call")
        st.sidebar.subheader("Parameters")
        temperature = st.sidebar.slider('Temperature controls how much randomness or "creativity" is in the output.', 0.0, 1.0, 0.0)
        max_length = st.sidebar.slider('The maximum tokens to use in the response completion', 0, 1000, 500)
        st.write(
        """
        **Steps:**
    1. Retrieve earnings call trascript: input the ticker, quarter, & year
    2. Compute document embeddings
    3. Ask questions & get summaries of the document

        """)

        st.write("**Step 1:**")
        ticker = st.text_input("Enter Ticker", "TSLA")
        quarter = st.selectbox("Select Quarter", ["1", "2", "3", "4"])
        year = st.selectbox("Select Year", ["2023", "2022", "2021", "2020"])

        if st.button("Get earnings transcript"):
            df = earnings_summary(ticker, quarter, year)
            st.success("Earnings transcript retrieved.")

        st.write("**Step 2:**")
        if st.button("Compute embeddings"):
            df = pd.read_csv("prepared_ec.csv")
            context_embeddings = compute_doc_embeddings(df)
            df_embeds = pd.DataFrame(context_embeddings).transpose()
            df_embeds.to_csv('sample_embeddings_earnings.csv', index=False)
            st.success("Embeddings completed.")

        st.write("**Step 3:**")
        label = "Enter your question here"
        prompt = st.text_input(label, key="prompt_input", placeholder=None)
        temperature = temperature
        max_length = max_length
        if st.button("Generate answer"):
            prompt = prompt
            document_embeddings, new_df = load_embeddings('sample_embeddings_earnings.csv', 'prepared_ec.csv')
            prompt_response = answer_query_with_context(prompt, new_df, document_embeddings, max_length, temperature)
            st.success("MLQ response")
            st.write(prompt_response)
    st.write("Disclaimer: AI & LLMs can be sometimes be unpredictable, none of this is financial advice and always do your own resarch.")
