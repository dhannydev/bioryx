import faiss
import numpy as np


class FaceIndex:

    def __init__(self, dimension=512):

        self.dimension = dimension

        # usamos cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        self.labels = []


    def build(self, embeddings, labels):

        embeddings = embeddings.astype("float32")

        # normalizar embeddings
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.labels = labels


    def search(self, embedding, k=1):

        embedding = np.array([embedding]).astype("float32")

        # normalizar query
        faiss.normalize_L2(embedding)

        scores, indices = self.index.search(embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            results.append({
                "label": self.labels[idx],
                "score": float(score)
            })

        return results