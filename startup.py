from pymilvus import MilvusClient
import numpy as np 

collection_name1 = "demo_collection"

client = MilvusClient("milvus_demo.db")
client.create_collection(
    collection_name=collection_name1,
    dimension=384
)

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
vectors = [[ np.random.uniform(-1, 1) for _ in range(384) ] for _ in range(len(docs)) ]

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))

res = client.insert(collection_name=collection_name1, data=data)

# print(res)

# query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

res = client.search(
    collection_name=collection_name1,  # target collection
    data=[vectors[0]],  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)
