from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(df, columns):
    docs = df[columns].fillna("").agg(" ".join, axis=1).tolist()
    embeddings = model.encode(docs, convert_to_tensor=True)
    return {
        "df": df.reset_index(drop=True),
        "docs": docs,
        "embeddings": embeddings
    }

def retrieve(index, query, k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, index["embeddings"])[0]
    top_k = torch.topk(scores, k)
    return index["df"].iloc[top_k.indices.cpu()]
