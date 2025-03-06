# Indian Legal Chatbot

A Flask-based conversational assistant that specializes in Indian litigation law, helping users understand legal processes, procedures, and concepts.

## Features

- **AI-Powered Legal Assistance**: Using OpenAI GPT models to provide accurate legal information
- **Vector Search**: FAISS-powered similarity search for precise information retrieval
- **Context-Aware Responses**: Maintains conversation history to provide coherent multi-turn interactions
- **Caching**: Implements query caching to improve response times for common questions
- **Hierarchical Document Processing**: Advanced document processing for better understanding of legal text structures
- **Multi-Agent System**: Uses LangGraph to coordinate specialized agents for different aspects of the query pipeline

## System Architecture

The system uses a three-agent architecture:

1. **Supervisor Agent**: Coordinates the workflow, preprocesses queries, and generates final responses
2. **Query Agent**: Retrieves relevant information from the legal knowledge base
3. **Summarization Agent**: Processes and simplifies the legal information into accessible explanations

## Tech Stack

- **Backend**: Flask
- **AI Processing**: LangChain, OpenAI API
- **Vector Database**: FAISS
- **Workflow Management**: LangGraph
- **Frontend**: HTML, CSS, JavaScript (with Tailwind CSS)

## Installation

### Prerequisites

- Python 3.8+ (3.10 preferred)
- OpenAI API key
- Legal documents corpus (litigation.txt or similar)

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rithwik25/legal_agent.git
   cd legal-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DOCUMENT_PATH=litigation.txt
   VECTOR_STORE_PATH=litigation_faiss_index
   PORT=5000
   FLASK_DEBUG=True
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Access the web interface at `http://localhost:5000`


## Project Structure

```
legal-chatbot/
├── app.py                  # Main Flask application
├── chatbot.py                # LegalChatbot implementation
├── cache.py                # Query caching implementation
├── data_structure.py       # Data models and structures
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in repo)
├── litigation.txt          # Legal document corpus
├── litigation_faiss_index/ # Vector database (generated)
├── flask_session/          # Session storage (generated)
└── templates/
    └── index.html          # Frontend template
```

## Usage

1. Open the web interface in your browser
2. Type your legal question in the input field
3. Receive a detailed response based on Indian litigation law
4. Continue the conversation with follow-up questions

## Customization

### Using Different Legal Documents

1. Replace the `litigation.txt` file with your own legal corpus
2. Update the `DOCUMENT_PATH` environment variable
3. Delete the existing vector store directory to force reindexing

### Modifying the Model

To use a different OpenAI model or adjust parameters, modify the following in `paste.py`:

```python
self.llm = ChatOpenAI(
    model="your-preferred-model", 
    temperature=0.2,  # Adjust for more/less creative responses
    streaming=True,
    api_key=os.getenv('OPENAI_API_KEY')
)
```

## Limitations

- Responses are based on the provided legal document corpus
- The system does not provide legal advice, only information
- API costs will scale with usage (OpenAI API charges)

## License

[MIT License](LICENSE)

## Acknowledgements

- OpenAI for the GPT models
- LangChain and LangGraph for the AI application framework
- FAISS for vector search capabilities

## Contact

For questions or support, please contact r_vkhera@cs.iitr.ac.in
