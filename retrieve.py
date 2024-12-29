from log import get_logger
from retrieval_algorithms import BM25_preprocessed, BM25_raw, Cohere_embeddings, Cohere_embeddings_v3, Gemini_embeddings, Levenshtein_distance, Openai_embeddings, Openai_embeddings_v3

# Configure logging
logger = get_logger(__name__)

algorithms = {}

def exe_bm25_raw():
    bm25 = BM25_raw()
    bm25.build_index()
    bm25.retrieve(save_to_disk=True)
algorithms["bm25_raw"] = exe_bm25_raw

def exe_bm25_preprocessed():
    bm25 = BM25_preprocessed()
    bm25.build_index()
    bm25.retrieve(save_to_disk=True)
algorithms["bm25_preprocessed"] = exe_bm25_preprocessed

def exe_levenshtein_distance():
    levensh = Levenshtein_distance()
    levensh.retrieve(save_to_disk=True)
algorithms["levenshtein_distance"] = exe_levenshtein_distance

def exe_openai_embedding_model():
    embedding_model = Openai_embeddings()
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["openai_embeddings"] = exe_openai_embedding_model

def exe_openai_embedding_model_v3_small():
    embedding_model = Openai_embeddings_v3(model = "text-embedding-3-small")
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["openai_embeddings_text-embedding-3-small"] = exe_openai_embedding_model_v3_small

def exe_openai_embedding_model_v3_large():
    embedding_model = Openai_embeddings_v3(model = "text-embedding-3-large")
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["openai_embeddings_text-embedding-3-large"] = exe_openai_embedding_model_v3_large

def exe_cohere_embedding_model():
    embedding_model = Cohere_embeddings()
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["cohere_embeddings"] = exe_cohere_embedding_model

def exe_cohere_embedding_model_v3_search_document():
    embedding_model = Cohere_embeddings_v3()
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["cohere_embeddings_v3_search_document"] = exe_cohere_embedding_model_v3_search_document

def exe_cohere_embedding_model_v3_search_query():
    embedding_model = Cohere_embeddings_v3(input_type="search_query")
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["cohere_embeddings_v3_search_query"] = exe_cohere_embedding_model_v3_search_query

def exe_cohere_embedding_model_v3_classification():
    embedding_model = Cohere_embeddings_v3(input_type="classification")
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["cohere_embeddings_v3_classification"] = exe_cohere_embedding_model_v3_classification

def exe_cohere_embedding_model_v3_clustering():
    embedding_model = Cohere_embeddings_v3(input_type="clustering")
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["cohere_embeddings_v3_clustering"] = exe_cohere_embedding_model_v3_clustering

def exe_gemini_embeddings_semantic_similarity_model():
    embedding_model = Gemini_embeddings()
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["gemini_embeddings_semantic_similarity"] = exe_gemini_embeddings_semantic_similarity_model

def exe_gemini_embeddings_retrieval_query_model():
    embedding_model = Gemini_embeddings(task_type='retrieval_query')
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["gemini_embeddings_retrieval_query"] = exe_gemini_embeddings_retrieval_query_model

def exe_gemini_embeddings_retrieval_document_model():
    embedding_model = Gemini_embeddings(task_type='retrieval_document')
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["gemini_embeddings_retrieval_document"] = exe_gemini_embeddings_retrieval_document_model

def exe_gemini_embeddings_classification_model():
    embedding_model = Gemini_embeddings(task_type='classification')
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["gemini_embeddings_classification"] = exe_gemini_embeddings_classification_model

def exe_gemini_embeddings_clustering_model():
    embedding_model = Gemini_embeddings(task_type='clustering')
    embedding_model.build_index()
    embedding_model.retrieve(save_to_disk=True)
algorithms["gemini_embeddings_clustering"] = exe_gemini_embeddings_clustering_model


if __name__=='__main__':
    targets = [
        "bm25_raw",
        "bm25_preprocessed",
        "levenshtein_distance",
        "openai_embeddings",
        "openai_embeddings_text-embedding-3-small",
        "openai_embeddings_text-embedding-3-large",
        "cohere_embeddings",
        "cohere_embeddings_v3.0_search_document",
        "cohere_embeddings_v3.0_search_query",
        "cohere_embeddings_v3.0_classification",
        "cohere_embeddings_v3.0_clustering",
        "gemini_embeddings_semantic_similarity",
        "gemini_embeddings_retrieval_query",
        "gemini_embeddings_retrieval_document",
        "gemini_embeddings_classification",
        "gemini_embeddings_clustering",
    ]

    for t in targets:
        algorithms[t]()