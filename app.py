import streamlit as st
from pathlib import Path
from langchain_community.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from llama_index.core import ServiceContext, VectorStoreIndex, download_loader,set_global_service_context
from llama_index.embeddings.langchain import LangchainEmbedding

# Define custom AI and Human prefixes
custom_ai_prefix = "Assistant"
custom_human_prefix = "User"

# Custom prompt template with new prefixes
custom_prompt_template = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest {custom_ai_prefix}. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Your goal is to provide answers relating to the financial performance of 
the company. Only answer to the asked question!<</SYS>>

Current conversation:
{{history}}
{custom_human_prefix}: {{input}}
{custom_ai_prefix}:"""

# Create a PromptTemplate object with the customized template
custom_prompt = PromptTemplate(input_variables=["history", "input"], template=custom_prompt_template)

@st.cache_resource(ttl=None, max_entries=None, show_spinner=True)
def llama_banker():
    # Example of using Amazon Bedrock with guardrails
    return Bedrock(
        model_id='meta.llama2-13b-chat-v1',  
        model_kwargs={
            'max_gen_len': 256,
            'top_p': 0.5,
            'temperature': 0.2
        },
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        verbose=False
    )

def get_conversation_memory():
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(ai_prefix=custom_ai_prefix, human_prefix=custom_human_prefix)
    return st.session_state.memory

def handle_conversation(input_text, document_context):
    # Combine the document context with the user's input text
    combined_input = f"{document_context}\n\nUser: {input_text}"
    
    memory = get_conversation_memory()
    llm = llama_banker()
    # Pass the combined input as the structured input to the ConversationChain
    structured_input = {'input': combined_input}
    
    conversation_chain = ConversationChain(llm=llm, memory=memory, prompt=custom_prompt, verbose=False, return_final_only=True)
    response = conversation_chain.invoke(structured_input, model_kwargs={'max_tokens': 120}, stop=[custom_human_prefix, custom_ai_prefix, '.'])
    
    ai_response = response.get('response')
    return ai_response

# Create embeddings instance explicitly with HuggingFaceEmbeddings
embed_model = LangchainEmbedding(SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))



service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llama_banker(),
    embed_model=embed_model  # Use the globally set HuggingFaceEmbeddings
)

# Set the global service context
set_global_service_context(service_context)

# Streamlit UI setup
st.title('🦙 Llama Banker')

# Load documents and create an index
PyMuPDFReader = download_loader("PyMuPDFReader")
loader = PyMuPDFReader()
documents = loader.load(file_path=Path('./data/annualreport.pdf'), metadata=True)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Get user input
user_prompt = st.text_input('Input your question here:')

if user_prompt:
    # Get the document context based on the user's query
    document_context = query_engine.query(user_prompt)
    
    # Invoke the conversation handler and get the AI's response
    ai_response = handle_conversation(user_prompt, document_context)
    
    # Display the AI's response in Streamlit
    st.write(ai_response)
