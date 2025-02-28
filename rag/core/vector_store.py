from pinecone import Pinecone, ServerlessSpec

from typing import Literal
import time
import os 

PINECONE_API = os.getenv("PINECONE_API")
pc = Pinecone(api_key=PINECONE_API)


def build_vectordb(index_name: str, dims: int = 384, metric: Literal["cosine", "euclidean", "dotproduct"] = "cosine" ) -> None:
    """
    Creates a Pinecone Vector DB 
    """

    if index_name not in pc.list_indexes().names():   
        pc.create_index(
            index_name=index_name, 
            dimension=dims,
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)



class RetrievalChunks:
    def __init__(self, model):
        self.model = model

    def retreive_chunks(self, text, index):
        xq = self.model.encode([text])[0].tolist()

        matches = index.query(
            vector=xq,
            top_k=3,
            include_metadata=True
        )

        chunks = []

        for m in matches["matches"]:
            content = m["metadata"]["content"]
            title = m["metadata"]["title"]

            pre = m["metadata"]["prechunk_id"]
            post = m["metadata"]["postchunk_id"]


            print(index.fetch(ids=[pre, post]))

            fetch_response = index.fetch(ids=[pre, post])
            other_chunks = fetch_response.vectors
            prechunk = other_chunks[pre]["metadata"]["content"]
            postchunk = other_chunks[post]["metadata"]["content"]

            chunk = f"""# {title}

            {prechunk[-400:]}
            {content}
            {postchunk[:400]}"""
            
            chunks.append(chunk)
        return chunks


