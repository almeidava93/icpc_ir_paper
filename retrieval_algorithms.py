import uuid
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import pandas as pd
import spacy
from rank_bm25 import BM25Okapi
from unidecode import unidecode
from pathlib import Path
from typing import Union
from tqdm import tqdm
from datetime import datetime as dt
from dotenv import load_dotenv
import os
import cohere
import google.generativeai as genai
from polyleven import levenshtein

# Custom modules
from log import get_logger

# Configure logging
logger = get_logger(__name__)

# Load relevant API keys
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
cohere_api_key = os.environ.get("COHERE_API_KEY")
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Load relevant data
data_path = Path("data", "extended_thesaurus.csv")
THESAURUS = pd.read_csv(data_path, index_col=0)
THESAURUS_EXPRESSIONS_LIST = THESAURUS["expression"].to_list()

# List with all the queries to evaluate the algorithms
query_data_path = Path("data","queries_to_label.csv")
query_data_df = pd.read_csv(query_data_path, index_col=0)
QUERY_DATA_LIST = query_data_df["query"].to_list()

# Path to save the results of the retrieval algorithms
queries_results_path = Path("results","queries_results.csv")

# Create relevant directories if they do not exist yet
vector_database_path = Path('vector_database')
if not vector_database_path.exists():
    vector_database_path.mkdir(parents=True, exist_ok=True)

class BM25_raw():
    """
    This class handles all processes related to the BM25 algorithm \
    and to the information retrieval with it.
    """
    def __init__(self):
        self.index : BM25Okapi = None
        self.tokeniser : spacy.lang.pt.Portuguese = spacy.blank("pt")

    def build_index(self) -> None:
        """
        This function creates a BM25 index based on the data given \
        and assigns it to the class property: self.index
        """

        # load portuguese language tokeniser
        nlp = spacy.blank("pt")

        # prepare for tokenisation
        text_list = THESAURUS['expression'].astype(str).values
        tok_text=[] # for our tokenised corpus

        # tokenise using SpaCy:
        for doc in nlp.pipe(text_list, disable=["tagger", "parser","ner"]):
            tok = [t.text for t in doc]
            tok_text.append(tok)

        # build a BM25 index
        bm25 = BM25Okapi(tok_text)

        # assign to class property
        self.index = bm25

    def retrieve(self, 
                 input: Union[str, list[str]] = QUERY_DATA_LIST, 
                 top_k: int = 10, 
                 data : pd.DataFrame = THESAURUS, 
                 save_to_disk : bool = False, 
                 results_file_path : Path = queries_results_path
                 ) -> list[list[str]]:
        """
        This function handles the information retrieval process and returns a list \
        of ICPC codes based on the queries given.

        input : query for the retrieval
        top_k : number of top results to return
        data : this is the data used to build the BM25 index and will be used to retrieve the results
        save_to_disk: if True, saves results to disk in the csv file in results_file_path
        """
        if isinstance(input, str):
            input = [input]

        assert isinstance(input, list) and all(isinstance(item, str) for item in input), \
            "Input must be either of type str or list[str]"
        
        all_results = []
        all_results_retrieval_time = []
        
        logger.info("Retrieving with BM25_raw...")
        for query in tqdm(input):
            t0 = dt.now()
            tokenised_query = [t.text for t in self.tokeniser(query)]
            results = self.index.get_top_n(tokenised_query, data.code.values, n=top_k)
            all_results.append(results)
            t1 = dt.now()
            t_delta = t1 - t0
            all_results_retrieval_time.append(t_delta)
        
        if save_to_disk:
            logger.info(f"Saving results to disk at {results_file_path}")
            queries_results_df = pd.read_csv(results_file_path, index_col=0).dropna()
            queries_results_df["bm25_raw"] = ['|'.join(result) for result in all_results]
            queries_results_df["bm25_raw_time"] = all_results_retrieval_time
            queries_results_df.to_csv(results_file_path)
        
        logger.info("Done!")

        return all_results
    

class BM25_preprocessed(BM25_raw):
    """
    This is a subclass of BM25_raw. This class handles all processes \
    related to the BM25 algorithm and to the information retrieval with \
    it as the BM25_raw class does.

    It extends BM25_raw introducing some data preprocessing before \
    the retrieval is performed. The data preprocessing includes:
    - lower casing
    - special characters removal
    """

    def build_index(self,
                    remove_special_chars : bool = True, 
                    lowercase : bool = True,
                    ) -> None:
        """
        This function creates a BM25 index based on the data given \
        and assigns it to the class property: self.index
        """

        # load portuguese language tokeniser
        nlp = spacy.blank("pt")

        # prepare for tokenisation
        text_list = THESAURUS['expression'].astype(str).values
        tok_text=[] # for our tokenised corpus

        # tokenise using SpaCy:
        for doc in nlp.pipe(text_list, disable=["tagger", "parser","ner"]):
            tok = [t.text for t in doc]
            
            # Apply preprocessing
            if remove_special_chars:
                tok = [unidecode(t) for t in tok]

            if lowercase:
                tok = [t.lower() for t in tok]

            tok_text.append(tok)

        # build a BM25 index
        bm25 = BM25Okapi(tok_text)

        # assign to class property
        self.index = bm25

    def retrieve(self, 
                 input: Union[str, list[str]] = QUERY_DATA_LIST, 
                 top_k: int = 10, 
                 data : pd.DataFrame = THESAURUS, 
                 remove_special_chars : bool = True, 
                 lowercase : bool = True,
                 save_to_disk : bool = False, 
                 results_file_path : Path = queries_results_path) -> list[list[str]]:
        """
        This function handles the information retrieval process and returns a list \
        of ICPC codes based on the queries given.

        It applies the desired preprocessing through the arguments remove_special_chars \
        and lowercase.

        input : query for the retrieval
        top_k : number of top results to return
        data : this is the data used to build the BM25 index and will be used to retrieve the results
        remove_special_chars: if True, remove special characters from the input
        lowercase: if True, every input is lowercased
        save_to_disk: if True, saves results to disk in the csv file in results_file_path
        """
        if isinstance(input, str):
            input = [input]

        assert isinstance(input, list) and all(isinstance(item, str) for item in input), \
            "Input must be either of type str or list[str]"
        
        # Apply preprocessing
        if remove_special_chars:
            input = [unidecode(v) for v in input]

        if lowercase:
            input = [v.lower() for v in input]
        
        all_results = []
        all_results_retrieval_time = []
        
        logger.info("Retrieving with BM25_preprocessed...")
        for query in tqdm(input):
            t0 = dt.now()
            tokenised_query = [t.text for t in self.tokeniser(query)]
            results = self.index.get_top_n(tokenised_query, data.code.values, n=top_k)
            all_results.append(results)
            t1 = dt.now()
            t_delta = t1 - t0
            all_results_retrieval_time.append(t_delta)
        
        if save_to_disk:
            logger.info(f"Saving results to disk at {results_file_path}")
            queries_results_df = pd.read_csv(results_file_path, index_col=0).dropna()
            queries_results_df["bm25_preprocessed"] = ['|'.join(result) for result in all_results]
            queries_results_df["bm25_preprocessed_time"] = all_results_retrieval_time
            queries_results_df.to_csv(results_file_path)
        
        logger.info("Done!")

        return all_results


class Levenshtein_distance():
    def __init__(self):
        pass

    def build_index(self) -> None:
        pass

    def retrieve(self, 
                 input: Union[str, list[str]] = QUERY_DATA_LIST, 
                 top_k: int = 10, 
                 data : pd.DataFrame = THESAURUS, 
                 save_to_disk : bool = False, 
                 results_file_path : Path = queries_results_path) -> list[list[str]]:
        
        if isinstance(input, str):
            input = [input]

        assert isinstance(input, list) and all(isinstance(item, str) for item in input), \
            "Input must be either of type str or list[str]"
        
        all_results = []
        all_results_retrieval_time = []
        
        data = data[["expression", "code"]].to_dict('records')
        
        logger.info("Retrieving with Levenshtein distance...")
        for query in tqdm(input):
            t0 = dt.now()

            # calculate levenshtein distance of each query to each entry in the thesaurus
            results = []
            
            try:    
                for item in data:
                    distance = levenshtein(query, item["expression"])
                    results.append({"expression": item["expression"], "distance": distance, "code": item["code"]})
            
            except Exception as e:
                    logger.error(f"An error ocurred: {e}")
                    logger.error(f"Current item being processed: {item}")
                    logger.error(f"results variable state: {results}")
                    raise e
        	
            # sort results based on levenshtein distance and get the top_k results
            try:
                sorted_results = sorted(results, key=lambda a: a["distance"])[:top_k]
            except Exception as e:
                logger.error(f"An error ocurred while sorting levenshtein distance results: {e}")
                logger.error(f"Current item being processed: {results}")
                raise e

            # get only ICPC codes from results
            final_results = [r["code"] for r in sorted_results]

            all_results.append(final_results)
            t1 = dt.now()
            t_delta = t1 - t0
            all_results_retrieval_time.append(t_delta)
        
        if save_to_disk:
            logger.info(f"Saving results to disk at {results_file_path}")
            queries_results_df = pd.read_csv(results_file_path, index_col=0).dropna()
            queries_results_df["levenshtein_distance"] = ['|'.join(result) for result in all_results]
            queries_results_df["levenshtein_distance_time"] = all_results_retrieval_time
            queries_results_df.to_csv(results_file_path)
        
        logger.info("Done!")

        return all_results


class Openai_embeddings():
    """
    This class handles all processes related to the Openai embedding model \
    and to the information retrieval with it.
    """
    def __init__(self,
                 api_key : str = openai_api_key,
                 model : str = "text-embedding-ada-002",
                 ):
        self.model = model
        self.db_client = chromadb.PersistentClient(path="vector_database", settings=Settings(allow_reset=True))
        self.db_collection_name = "openai_embeddings"

        # Authentication at OpenAI API
        openai.api_key = api_key
        self.openai_client = OpenAI()

    def get_embeddings(self, document : str) -> list[float]:
        return self.openai_client.embeddings.create(input = [document], model=self.model).data[0].embedding
    
    def add_documents_to_db(self, documents : pd.DataFrame = THESAURUS) -> None:
        logger.info(f"Adding documents to collection: {self.db_collection_name}")
        # Get collection
        collection = self.db_client.get_collection(self.db_collection_name)
        # Get documents that already are in the vectorstore
        documents_in_vectorstore = collection.get()['documents']
        # Define pending documents
        pending_documents = [doc for doc in documents.to_dict('records') if doc["expression"] not in documents_in_vectorstore]
        # Add documents with embeddings
        for doc in tqdm(pending_documents,
                    desc=f"Processing documents",
                    unit="docs",
                    ):
            if doc["expression"] not in documents_in_vectorstore:
                collection.add(ids=str(uuid.uuid1()), 
                            documents=doc["expression"],
                            embeddings=self.get_embeddings(doc["expression"]),
                            metadatas={"code": doc["code"], "expression": doc["expression"]})
        logger.info(f"{len(pending_documents)} added to collection: {self.db_collection_name}")

    def setup_vector_database(self) -> None:
        # make sure that collection exists
        # define cosine similarity as the distance function through Hierarchical Navigable Small Worlds (HNSW)
        logger.info(f"Checking ChromaDB collection: {self.db_collection_name}")
        collection = self.db_client.get_or_create_collection(name=self.db_collection_name,
                                                             metadata={"hnsw:space": "cosine"}) 

        documents_in_vectorstore = collection.get()['documents']

        # check if the documents are exactly what we need
        if sorted(documents_in_vectorstore) != sorted(THESAURUS_EXPRESSIONS_LIST):
            logger.info(f"The data in the collection is not complete or incorrect. Applying corrections to {self.db_collection_name}")
            
            logger.info("Removing documents that should not be in the collection.")
            incorrect_documents = [item for item in documents_in_vectorstore if item not in THESAURUS_EXPRESSIONS_LIST]
            for item in tqdm(incorrect_documents):
                collection.delete(where={"expression": {"$eq": item}})
            logger.info(f"{len(incorrect_documents)} were removed from {self.db_collection_name} collection.")

            logger.info("Checking pending documents.")
            self.add_documents_to_db()

    def build_index(self) -> None:
        self.setup_vector_database()
        logger.info(f"Collection {self.db_collection_name} is ready for information retrieval.")

    def retrieve(self, 
                 input: Union[str, list[str]] = QUERY_DATA_LIST, 
                 top_k: int = 10, 
                 data : pd.DataFrame = THESAURUS, 
                 save_to_disk : bool = False, 
                 results_file_path : Path = queries_results_path) -> list[list[str]]:
        """
        This function handles the information retrieval process and returns a list \
        of ICPC codes based on the queries given.

        It applies the desired preprocessing through the arguments remove_special_chars \
        and lowercase.

        input : query for the retrieval
        top_k : number of top results to return
        data : this is the data used to build the BM25 index and will be used to retrieve the results
        save_to_disk: if True, saves results to disk in the csv file in results_file_path
        """

        if isinstance(input, str):
            input = [input]

        assert isinstance(input, list) and all(isinstance(item, str) for item in input), \
            "Input must be either of type str or list[str]"
        
        all_results = []
        all_results_retrieval_time = []

        # Get collection
        collection = self.db_client.get_collection(self.db_collection_name)

        logger.info(f"Retrieving with Collection {self.db_collection_name}...")
        for query in tqdm(input):
            t0 = dt.now()
        
            # Generate query embedding
            query_embedding = self.get_embeddings(query)

            # Query collection
            results = collection.query(query_embeddings=query_embedding, include=["documents", "metadatas"])
            metadatas = results["metadatas"][0]
            icpc_codes_list = [doc["code"] for doc in metadatas]            
            
            # Collect results and elapsed time
            all_results.append(icpc_codes_list)
            t1 = dt.now()
            t_delta = t1 - t0
            all_results_retrieval_time.append(t_delta)
        
        if save_to_disk:
            logger.info(f"Saving results to disk at {results_file_path}")
            queries_results_df = pd.read_csv(results_file_path, index_col=0).dropna()
            queries_results_df[f"{self.db_collection_name}"] = ['|'.join(result) for result in all_results]
            queries_results_df[f"{self.db_collection_name}_time"] = all_results_retrieval_time
            queries_results_df.to_csv(results_file_path)
        
        logger.info("Done!")

        return all_results
        

class Openai_embeddings_v3(Openai_embeddings):
    def __init__(self,
                 api_key : str = openai_api_key,
                 model : str = "text-embedding-3-small",
                 ):
        
        valid_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]

        if model not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}, but got '{model}'")

        self.model = model
        self.db_client = chromadb.PersistentClient(path="vector_database", settings=Settings(allow_reset=True))
        self.db_collection_name = "openai_embeddings_" + model

        # Authentication at OpenAI API
        openai.api_key = api_key
        self.openai_client = OpenAI()


    def get_embeddings(self, document: str):
        return self.openai_client.embeddings.create(input = [document], model=self.model).data[0].embedding


class Cohere_embeddings(Openai_embeddings):
    def __init__(self,
                 api_key : str = cohere_api_key,
                 model : str = "embed-multilingual-v2.0",
                 ):
        self.model = model
        self.db_client = chromadb.PersistentClient(path="vector_database", settings=Settings(allow_reset=True))
        self.db_collection_name = "cohere_embeddings"

        # Authentication at Cohere
        self.cohere_client = cohere.Client(api_key)


    def get_embeddings(self, document: str):
        return self.cohere_client.embed(texts=[document], model=self.model).embeddings


class Cohere_embeddings_v3(Openai_embeddings):
    def __init__(self,
                 api_key : str = cohere_api_key,
                 model : str = "embed-multilingual-v3.0",
                 input_type : str = "search_document",
                 ):
        self.model = model
        self.db_client = chromadb.PersistentClient(path="vector_database", settings=Settings(allow_reset=True))
        self.db_collection_name = "cohere_embeddings_v3.0_" + input_type
        self.input_type = input_type

        # Authentication at Cohere
        self.cohere_client = cohere.Client(api_key)


    def get_embeddings(self, document: str):
        return self.cohere_client.embed(texts=[document], model=self.model, input_type=self.input_type).embeddings


class Gemini_embeddings(Openai_embeddings):
    def __init__(self,
                 api_key : str = gemini_api_key,
                 model : str = "models/embedding-001",
                 task_type : str = "semantic_similarity"
                 ):
        self.model = model
        self.task_type = task_type
        self.db_client = chromadb.PersistentClient(path="vector_database", settings=Settings(allow_reset=True))
        self.db_collection_name = "gemini_embeddings_" + task_type

        # Authentication at Cohere
        genai.configure(api_key=api_key)


    def get_embeddings(self, document: str):
        if self.task_type == "retrieval_document":
            result = genai.embed_content(
                model=self.model,
                content=document,
                task_type=self.task_type,
                title="Embedding of single string")
            return result['embedding']
        
        else:
            result = genai.embed_content(
                model=self.model,
                content=document,
                task_type=self.task_type)
            return result['embedding']