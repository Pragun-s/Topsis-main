import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import time
import torch

# Define pre-trained models
models = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "stsb-roberta-base",
    "distilbert-base-nli-stsb-mean-tokens"
]

# Define sample sentence pairs for similarity evaluation
sentences = [
    ("The cat sits outside.", "A feline is outdoors."),
    ("He is playing football.", "The boy enjoys a soccer match."),
    ("The weather is nice today.", "It is a beautiful day outside."),
]

# Criteria: Cosine Similarity (higher is better), Inference Time, Memory Usage, Model Size
criteria = ["Similarity Score", "Inference Time", "Memory Usage", "Model Size"]

# Data storage
performance_data = []

for model_name in models:
    model = SentenceTransformer(model_name)
    start_time = time.time()
    memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    model_size = sum(p.numel() for p in model.parameters())
    
    similarities = []
    for sent1, sent2 in sentences:
        emb1 = model.encode(sent1, convert_to_tensor=True)
        emb2 = model.encode(sent2, convert_to_tensor=True)
        cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
        similarities.append(cosine_sim)
    
    avg_similarity = np.mean(similarities)
    inference_time = time.time() - start_time
    
    performance_data.append([avg_similarity, inference_time, memory_usage, model_size])

# Convert to DataFrame
df = pd.DataFrame(performance_data, columns=criteria, index=models)

# Normalize Data for TOPSIS
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)

# Define weights and impact
weights = [0.4, 0.2, 0.2, 0.2]  # Adjust based on importance
impact = [1, -1, -1, -1]  # 1 for benefit criteria, -1 for cost criteria

# Calculate weighted normalized matrix
weighted_matrix = df_normalized * weights

# Identify ideal best and worst solutions
ideal_best = np.max(weighted_matrix, axis=0) * impact
ideal_worst = np.min(weighted_matrix, axis=0) * impact

# Compute distances to ideal best and worst
dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)

# Compute TOPSIS scores
topsis_scores = dist_worst / (dist_best + dist_worst)

# Add scores to DataFrame
df["TOPSIS Score"] = topsis_scores

df_sorted = df.sort_values(by="TOPSIS Score", ascending=False)
print(df_sorted)

# Save results
df_sorted.to_csv("topsis_results.csv", index=True)
