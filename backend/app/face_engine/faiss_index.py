import faiss
import numpy as np
import os
import json


class FaceIndex:

    def __init__(self, dimension=512, index_path="face.index", labels_path="labels.json"):

        self.dimension = dimension
        self.index_path = index_path
        self.labels_path = labels_path

        self.index = faiss.IndexFlatIP(dimension)
        self.labels = []


    def build(self, embeddings, labels):

        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.labels = labels

        self.save()


    def save(self):

        faiss.write_index(self.index, self.index_path)

        with open(self.labels_path, "w") as f:
            json.dump(self.labels, f)


    def load(self):

        if not os.path.exists(self.index_path) or not os.path.exists(self.labels_path):
            return False

        self.index = faiss.read_index(self.index_path)

        with open(self.labels_path, "r") as f:
            self.labels = json.load(f)

        return True


    def search(self, embedding, k=1):

        if self.index.ntotal == 0:
            return []

        embedding = np.array([embedding]).astype("float32")
        faiss.normalize_L2(embedding)

        scores, indices = self.index.search(embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.labels):
                results.append({
                    "label": self.labels[idx],
                    "score": float(score)
                })

        return results