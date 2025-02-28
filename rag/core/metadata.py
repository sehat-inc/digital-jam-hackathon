from chunking import SemanticChunker

from typing import List

chunker = SemanticChunker()

class BuildMetaData:
    def __init__(self, chunks: List, doc_id: int, doc_title: str, lease_type: str):
        self.chunks = chunks
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.lease_type = lease_type

    def build(self):
        metadata = []

        for i, content in enumerate(self.chunks):
            prechunk_id = "" if i == 0 else f"{self.doc_id}#{i-1}"
            postchunk_id = "" if i + 1 == len(self.chunks) else f"{self.doc_id}#{i+1}"

            metadata.append({
                "id" : f"{self.doc_id}#{i}",
                "title" : self.doc_title,
                "lease_type" : self.lease_type,
                "content" : content,
                "prechunk_id" : prechunk_id,
                "postchunk_id" : postchunk_id
            })

        return metadata