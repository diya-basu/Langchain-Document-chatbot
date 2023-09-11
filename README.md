# Langchain Document chatbot 
Chat-with-documents-using-Langchain-OpenAI-HuggingFace-frameworks
Using GPT 3.5 turbo and the free huggingface model-Dolly b v2 to query documents(pdf,txt,ppt,docx), images(jpg,png,jpeg) and csv files. The application is built on streamlit. 

# Working: 
1. Sign up on the OpenAI app and generate your own api key 
2. Create a new file called apikey.py in the same directory as the one in which the app.py will be stored. 
3. Open the apikey.py file and add the variable api_key=‘YOUR_API_KEY’ 
4. Install necessary dependencies: (Run this command in your terminal)
   ```
   pip install streamline langchain tiktoken pypdf2 openai faiss-cpu huggingface_hub
   ```

6. To run the final app, download and save app1.py and In terminal navigate to the directory app1.py is stored in. 
   Run the following command:
   ```
   streamlit run app1.py
   ```
  

# To use the Huggingface models: 
1. sign up on the Huggingface website and generate access token. 
2. Select the Huggingface model best suited to your needs and mention it under "repo id" in the given code files. 
