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

    def retreive_chunks(self, text, index, doc_id):
        xq = self.model.encode([text])[0].tolist()

        matches = index.query(
            vector=xq,
            top_k=3,
            include_metadata=True, 
            filter={
                "doc_id": {"$eq": f"{doc_id}"}
            }
        )

        chunks = []
        for m in matches["matches"]:
            content = m["metadata"]["content"]
            title = m["metadata"]["title"]
            pre = m["metadata"].get("prechunk_id", "")
            post = m["metadata"].get("postchunk_id", "")
            
            prechunk_text = ""
            postchunk_text = ""
            
            # Fetch pre-chunk if it exists and is not empty
            if pre:
                try:
                    pre_fetch = index.fetch(ids=[pre])
                    if pre in pre_fetch.vectors:
                        prechunk = pre_fetch.vectors[pre]["metadata"]["content"]
                        prechunk_text = prechunk[-400:] if prechunk else ""
                except Exception as e:
                    print(f"Error fetching pre-chunk: {e}")
            
            # Fetch post-chunk if it exists and is not empty
            if post:
                try:
                    post_fetch = index.fetch(ids=[post])
                    if post in post_fetch.vectors:
                        postchunk = post_fetch.vectors[post]["metadata"]["content"]
                        postchunk_text = postchunk[:400] if postchunk else ""
                except Exception as e:
                    print(f"Error fetching post-chunk: {e}")
                
            chunk = f"""# {title}

            {prechunk_text}
            {content}
            {postchunk_text}"""
       
            chunks.append(chunk)
        return chunks


