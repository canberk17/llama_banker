# Llama Banker

Llama Banker is a Streamlit application designed to leverage the power of the LangChain and LLAMA Index libraries for generating AI-driven insights into financial documents. Utilizing the `meta.llama2-13b-chat-v1` model, Llama Banker provides users with an interactive platform to ask questions about financial performance based on the content of an annual report PDF.

## Features

- **Interactive Q&A**: Ask specific questions regarding the financial performance of a company and receive AI-generated answers.
- **Document Contextual Understanding**: Integrates document reading capability to understand and respond based on the context provided in financial documents.
- **Custom AI and Human Prefixes**: Utilizes customized prefixes for AI and user inputs to distinguish between the two parties in the conversation.

## Setup

To set up and run Llama Banker on your local machine, follow these steps:

1. **Clone the Repository**
   Clone this repository to your local machine using git:



2. **Install Dependencies**
Navigate to the cloned repository's directory and install the required Python libraries:

```bash
pip install -r requirements.txt
```


This command will install Streamlit, LangChain, LLAMA Index, and other necessary libraries.

3. **Prepare the Data**
Place your annual report PDF in the `data` directory of your project. Ensure the file is named `annualreport.pdf`.

4. **Run the Application**
Start the Streamlit application by running:

```bash
streamlit run app.py

```

