from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGVector
from llm_manager import GeminiManager

PROMPT_TEMPLATE = """
You are a virtual assistant specialized in document analysis. Your task is to answer the USER QUESTION strictly using only the CONTEXT provided below.

### CONTEXT:
{context}

### CRITICAL RESPONSE RULES:
1. STRICT FOCUS: Answer exclusively based on the information contained in the CONTEXT above.
2. MISSING INFORMATION: If the CONTEXT does not contain the answer, you must respond exactly:
   "I do not have enough information to answer your question."
3. NO EXTERNAL KNOWLEDGE: Do not use historical facts, general knowledge, or external information not present in the provided text.
4. INTEGRITY: Never fabricate revenue figures, names, dates, or statistics.
5. OBJECTIVITY: Be direct and avoid phrases like "Based on the text..." or "The document states...". Go straight to the answer.

### BEHAVIOR EXAMPLES:
- Out-of-context question: "Who is the CEO of Google?"
- Response: "I do not have enough information to answer your question."

- Opinion-based question: "What do you think about the company's strategy?"
- Response: "I do not have enough information to answer your question."

---
### USER QUESTION:
{question}

### ANSWER:
"""

class QASystem:
    """Class responsible for retrieving context from the database and generating answers."""

    def __init__(self, db_url: str, collection_name: str, gemini_manager: GeminiManager):
        self.embeddings = gemini_manager.get_embeddings()
        self.llm = gemini_manager.get_chat_model()
        
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=db_url,
            use_jsonb=True,
        )
        
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

    def search_prompt(self, question: str, k: int = 10) -> str:
        """Searches for the most relevant documents and generates an answer."""
        
        print(f"Searching context for question: '{question}'...")
        
        # 1. Retrieve the top K most relevant chunks (Similarity Search)
        docs_with_score = self.vector_store.similarity_search_with_score(question, k=k)
        docs = [doc for doc, score in docs_with_score]
        
        if not docs:
            return "No relevant documents found in the database to generate context."
            
        # 2. Format context by concatenating retrieved chunks
        formatted_context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. Build LangChain pipeline (Prompt + LLM)
        chain = self.prompt | self.llm
        
        # 4. Invoke model
        print("Generating answer...")
        response = chain.invoke({
            "context": formatted_context,
            "question": question
        })
        
        return response.content