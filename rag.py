import os
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


class RAGEngine:
    def __init__(self, pdf_path="data/Devraj_Structured_Profile.pdf"):
        self.pdf_path = pdf_path
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore_path = "vectorstore/faiss_index"

        # Auto-create folder if missing
        os.makedirs(self.vectorstore_path, exist_ok=True)

        # Automatically load or rebuild FAISS index
        self.vs = self.load_or_build_index()

    # --------------------------
    # Load FAISS or rebuild it
    # --------------------------
    def load_or_build_index(self):
        index_file = os.path.join(self.vectorstore_path, "index.faiss")

        if os.path.exists(index_file):
            try:
                print("🔵 Loading existing FAISS index...")
                return FAISS.load_local(
                    self.vectorstore_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print("⚠️ FAISS index corrupted. Rebuilding...")
                print("Error:", e)
                return self.create_vector_store()

        else:
            print("🟠 No FAISS index found. Creating a new one...")
            return self.create_vector_store()

    # --------------------------
    # Create vector store
    # --------------------------
    def create_vector_store(self):
        print("📄 Loading PDF:", self.pdf_path)
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()

        print("✂️ Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        print("🔢 Creating embeddings...")
        vs = FAISS.from_documents(chunks, self.embedding_model)

        print("💾 Saving FAISS index...")
        vs.save_local(self.vectorstore_path)

        print("✅ New FAISS index created successfully.")
        return vs

    # --------------------------
    # Search
    # --------------------------
    def search(self, query):
        print("🔍 Searching for:", query)
        results = self.vs.similarity_search(query, k=4)
        context = "\n".join([r.page_content for r in results])
        return context
