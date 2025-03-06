import os
import re
from datetime import datetime
from cache import QueryCache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from data_structure import Message, ChatState
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv(".env")
openai_key = os.getenv("OPENAI_API_KEY")

class LegalChatbot:
    def __init__(self, document_path='litigation.txt', vector_store_path='litigation_faiss_index'):
        
        self.query_cache = QueryCache() # Initialize caching
        self.conversation_history = [] # Initialize conversation history
        
        # Load or create vector store
        self.vector_store_path = vector_store_path
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
        self.vector_store = self._load_or_create_vector_store(document_path)
        
        # Set up retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Set up LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.2, 
            streaming=True,  # Enable streaming
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Build the graph
        self.graph = self._build_graph().compile()

    def _load_or_create_vector_store(self, document_path: str) -> FAISS:
        """
        Load existing FAISS index or create a new one if it doesn't exist.
        """
        if os.path.exists(self.vector_store_path):
            try:
                # Load existing vector store
                return FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Could not load existing vector store: {e}")
        
        # If no existing store, create a new one
        sample_docs = self._load_documents([document_path])
        vector_store = self._create_vector_store(sample_docs)
        
        # Save the new vector store
        vector_store.save_local(self.vector_store_path)
        
        return vector_store

    def _load_documents(self, document_paths: List[str]) -> List[Document]:
        """Load documents from various file formats with hierarchical chunking for txt files."""
        documents = []
        for path in document_paths:
            if path.endswith('.txt'):
                text_content = self._load_from_text_file(path)
                chunked_docs = self._hierarchical_chunking(text_content, path)
                documents.extend(chunked_docs)
        return documents

    def _load_from_text_file(self, file_path: str) -> str:
        """Load raw text content from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _hierarchical_chunking(self, text: str, file_path: str = 'unknown') -> List[Document]:
        """
        Split text hierarchically into documents with metadata.
        """
        documents = []
        sections = re.split(r'\n\s\n\s\n+', text)

        for section_idx, section_content in enumerate(sections):
            if not section_content.strip():
                continue

            section_lines = section_content.strip().split('\n', 1)
            section_title = section_lines[0].strip() if len(section_lines) > 0 else f"Section {section_idx + 1}"
            section_text = section_lines[1] if len(section_lines) > 1 else section_lines[0]

            paragraphs = re.split(r'\n\s*\n+', section_text)

            for para_idx, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    continue

                para_content = paragraph.strip()
                if para_idx > 0:
                    para_content = f"{section_title}\n\n{para_content}"

                doc = Document(
                    page_content=para_content,
                    metadata={
                        "source": os.path.basename(file_path),
                        "section": section_title,
                        "section_idx": section_idx,
                        "paragraph_idx": para_idx,
                        "section_paragraph": f"{section_idx}.{para_idx}"
                    }
                )
                documents.append(doc)

        return documents

    def _create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Creates a vector store with intelligent text splitting.
        """
        splits = []

        for doc in documents:
            if len(doc.page_content) > 1500:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", " ", ""]
                )
                long_splits = text_splitter.split_documents([doc])
                splits.extend(long_splits)
            else:
                splits.append(doc)

        return FAISS.from_documents(splits, self.embeddings)

    def supervisor_agent(self, state: ChatState) -> ChatState:
        """
        Supervisor agent that:
        1. Processes the incoming query with conversation context
        2. Determines the next step in the workflow
        3. Handles final response generation and memory updates
        """
        # If we're just starting (no enhanced query yet), do preprocessing
        if "enhanced_query" not in state or not state["enhanced_query"]:
            return self._preprocess_query(state)

        # If we have a summary but no answer, generate the final response
        if state.get("summary") and not state.get("answer"):
            return self._generate_response(state)

        # Default next step is query processing
        return {**state, "next_agent": "query_agent"}

    def _preprocess_query(self, state: ChatState) -> ChatState:
        """Enhanced query preprocessing with more context-aware prompting."""
        query = state["query"]
        conversation_history = state.get("conversation_history", [])

        # Check cache first
        cached_response = self.query_cache.get(query)
        if cached_response:
            return {
                **state,
                "summary": cached_response,
                "next_agent": "supervisor_agent"
            }

        # specific prompt for Indian litigation context
        prompt = ChatPromptTemplate.from_template(
            """You are a specialized legal assistant focusing on Indian litigation processes. 
            Your task is to formulate a comprehensive, legally precise search query that captures 
            the nuanced legal context of the user's question.

            Consider the following guidelines:
            - Extract key legal terminologies
            - Identify specific areas of Indian litigation law
            - Expand the query to include potential related legal concepts

            Conversation History (if applicable):
            {history}

            Current Query: {query}

            Enhanced Legal Search Query:"""
        )

        # Creating a chain for query enhancement
        if conversation_history:
            history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history[-5:]])
            
            chain = (
                {"history": lambda _: history_text, "query": lambda x: query}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            enhanced_query = chain.invoke("")
        else:
            enhanced_query = query

        return {
            **state,
            "enhanced_query": enhanced_query,
            "next_agent": "query_agent"
        }

    def _generate_response(self, state: ChatState) -> ChatState:
        """Generate response with more detailed, context-aware prompting."""
        summary = state["summary"]
        legal_references = state.get("legal_references", [])
        conversation_history = state.get("conversation_history", [])
        query = state["query"]

        # Create a prompt specifically tailored to Indian litigation context
        prompt = ChatPromptTemplate.from_template(
            """As an expert in Indian litigation law, provide a comprehensive, 
            legally precise response that:
            - Directly addresses the specific legal query
            - Explains relevant legal principles and procedures
            - References specific sections of Indian legal framework when applicable
            - Uses clear, professional language accessible to non-legal professionals

            Key Considerations for Indian Litigation:
            - Align response with current Indian legal practices
            - Highlight any procedural nuances specific to Indian courts
            - Provide context about potential legal implications

            Legal Summary: {summary}

            Detailed Legal Response:"""
        )

        # Create a chain for response generation with streaming
        chain = (
            {"summary": lambda x: summary}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke({}) # Generate the answer
        self.query_cache.set(query, answer) # Cache the response

        # Update conversation history
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not conversation_history or conversation_history[-1]["role"] != "human":
            conversation_history.append({
                "role": "human",
                "content": query,
                "timestamp": current_time
            })

        conversation_history.append({
            "role": "ai",
            "content": answer,
            "timestamp": current_time
        })

        return {
            **state,
            "answer": answer,
            "conversation_history": conversation_history,
            "next_agent": END
        }

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for the multi-agent system."""
        graph = StateGraph(ChatState)

        # Add nodes with updated agents
        graph.add_node("supervisor_agent", self.supervisor_agent)
        graph.add_node("query_agent", self._query_agent)
        graph.add_node("summarization_agent", self._summarization_agent)

        # Define conditional edges
        graph.add_conditional_edges(
            "supervisor_agent",
            self._router,
            {
                "query_agent": "query_agent",
                "summarization_agent": "summarization_agent",
                END: END
            }
        )

        graph.add_conditional_edges(
            "query_agent",
            self._router,
            {
                "summarization_agent": "summarization_agent"
            }
        )

        graph.add_conditional_edges(
            "summarization_agent",
            self._router,
            {
                "supervisor_agent": "supervisor_agent"
            }
        )

        graph.set_entry_point("supervisor_agent")

        return graph

    def _router(self, state: ChatState) -> str:
        """Route to the next agent based on state."""
        return state.get("next_agent", "supervisor_agent")

    def _query_agent(self, state: ChatState) -> ChatState:
        """Query Agent that retrieves relevant information from legal documents."""
        query = state["enhanced_query"]
        docs = self.retriever.invoke(query)

        context = [f"Source: {doc.metadata.get('source', 'Unknown')}, Section: {doc.metadata.get('section', 'Unknown')}\n\n{doc.page_content}" for doc in docs]
        legal_references = [f"{doc.metadata.get('source', 'Unknown')} - {doc.metadata.get('section', 'Unknown')}" for doc in docs]

        return {
            **state,
            "context": context,
            "legal_references": legal_references,
            "next_agent": "summarization_agent"
        }

    def _summarization_agent(self, state: ChatState) -> ChatState:
        """Summarization Agent that simplifies legal information."""
        query = state["enhanced_query"]
        context = state["context"]

        if not context:
            return {
                **state,
                "summary": "I couldn't find specific legal information about your query in my knowledge base.",
                "next_agent": "supervisor_agent"
            }

        # More focused prompt for summarization
        prompt = ChatPromptTemplate.from_template(
            """As a legal expert in Indian litigation, provide a precise, legally accurate summary 
            that addresses the user's specific legal question. Your summary should:
            - Break down complex legal concepts
            - Highlight key procedural aspects
            - Provide clear, actionable insights
            - Maintain professional legal terminology

            User's Specific Legal Question: {query}

            Relevant Legal Information:
            {context}

            Concise Legal Summary:"""
        )

        chain = (
            {"query": RunnablePassthrough(), "context": lambda _: "\n\n".join(context)}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        summary = chain.invoke(query)

        return {
            **state,
            "summary": summary,
            "next_agent": "supervisor_agent"
        }

    def chat(self, query: str) -> str:
        """Process a user query and return a response."""
        state = {
            "query": query,
            "enhanced_query": "",
            "context": [],
            "summary": "",
            "answer": "",
            "legal_references": [],
            "conversation_history": self.conversation_history,
            "next_agent": "supervisor_agent"
        }

        result = self.graph.invoke(state)
        self.conversation_history = result["conversation_history"]

        return result["answer"]

    def get_conversation_history(self) -> List[Message]:
        """Return the current conversation history."""
        return self.conversation_history

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []