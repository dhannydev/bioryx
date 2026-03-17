import os
import numpy as np


def load_embeddings(database_path="embeddings"):

    embeddings = []
    labels = []

    for file in os.listdir(database_path):

        if file.endswith(".npy"):

            user_id = file.replace(".npy", "")

            user_embeddings = np.load(
                os.path.join(database_path, file)
            )

            for emb in user_embeddings:

                embeddings.append(emb)
                labels.append(user_id)

    return np.array(embeddings), labels