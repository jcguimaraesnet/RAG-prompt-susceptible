from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd # cuidado: se importar no topo, dá erro na indexação (gravação no disco)

load_dotenv()

# Hardcoded values for easy adjustment
CHUNK_SIZE = 1000 #only for db upload
TOKEN_CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Load dataset and convert to DataFrame for easier manipulation
dataset = load_dataset("jamescalam/ai-arxiv")
df = pd.DataFrame(dataset['train'])

#imprimir quantidade de documentos
print(f"Quantidade de documentos: {len(df['content'])}")

# Prepare document objects from the dataset for indexing
documents = [Document(text=content) for content in df['content']]

# Setup the embedding model
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Classic vector DB
# Initialize a text splitter with hardcoded values for chunking documents
parser = TokenTextSplitter(chunk_size=TOKEN_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
nodes = parser.get_nodes_from_documents(documents)

#imprimir quantidade de nós
print(f"Quantidade de nós: {len(nodes)}")
chroma_client = chromadb.PersistentClient('./chroma_db')
chroma_collection = chroma_client.get_or_create_collection("ai_arxiv_full_v1")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection, embed_model=embed_model)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True,
    use_async=False
)
print("Indexação finalizada!")
