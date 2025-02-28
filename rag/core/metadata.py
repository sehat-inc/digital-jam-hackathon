from typing import List

class BuildMetaData:
    def __init__(self):
        pass

    def build(self, chunks: List, doc_id: int, doc_title: str, lease_type: str):
        metadata = []

        for i, content in enumerate(chunks):
            prechunk_id = "" if i == 0 else f"{doc_id}#{i-1}"
            postchunk_id = "" if i + 1 == len(chunks) else f"{doc_id}#{i+1}"

            metadata.append({
                "doc_id" : f"{doc_id}",
                "id" : f"{doc_id}#{i}",
                "title" : doc_title,
                "lease_type" : lease_type,
                "content" : content,
                "prechunk_id" : prechunk_id,
                "postchunk_id" : postchunk_id
            })

        return metadata