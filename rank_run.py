import json
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
from transformers import logging
from tqdm import tqdm

logging.set_verbosity_error()

MODEL_NAME = "CodeBERT"
WEIGHTS_PATH = "/rank_train/pytorch_model.bin"
MAX_LENGTH = 510
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CodeRankingModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.codebert = base_model
        self.dropout = nn.Dropout(0.1)
        self.scoring_layer = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        scores = self.scoring_layer(cls_embedding)
        return scores.squeeze(-1)


def rank_candidates(prompt, candidates, model, tokenizer, device, max_length=MAX_LENGTH):
    encoding_list = []
    for candidate in candidates:
        encoding = tokenizer(
            text=prompt,
            text_pair=candidate,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encoding_list.append(encoding)

    input_ids = torch.stack([e["input_ids"].squeeze(0) for e in encoding_list]).to(device)
    attention_mask = torch.stack([e["attention_mask"].squeeze(0) for e in encoding_list]).to(device)

    model.eval()
    with torch.no_grad():
        scores = model(input_ids, attention_mask)

    scores = scores.cpu().numpy()
    ranked_candidates = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked_candidates


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    base_model = RobertaModel.from_pretrained(MODEL_NAME)
    model = CodeRankingModel(base_model).to(DEVICE)

    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()


    ranked_results = rank_candidates(sample_prompt, sample_candidates, model, tokenizer, DEVICE)
    for rank, (idx, score) in enumerate(ranked_results):
        print(f"Rank {rank+1}: Candidate {idx}, Score {score:.4f}")

    INPUT_FILE = "/triplet_data.json"
    OUTPUT_FILE = "/ranking_results.json"

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in:
        raw_data = json.load(f_in)

    ranking_output = []

    for item in tqdm(raw_data, desc="Batch Ranking"):
        item_id = item.get("id", None)
        prompt = item["prompt"]
        candidates = [c["code"] for c in item["candidates"]]

        ranked = rank_candidates(prompt, candidates, model, tokenizer, DEVICE)

        ranked_list = [
            {
                "candidate_idx": idx,
                "score": float(score),
                "code": candidates[idx]
            }
            for idx, score in ranked
        ]

        ranking_output.append({
            "id": item_id,
            "prompt": prompt,
            "ranked_candidates": ranked_list
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        json.dump(ranking_output, f_out, ensure_ascii=False, indent=2)

    print(f"Ranking completed. Results saved to {OUTPUT_FILE}")
