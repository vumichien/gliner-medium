from gliner import GLiNER
import json
import torch

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_small-v1", max_length=512)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device: {device}")
model = model.to(device)

# evaluate the model on a test dataset
with open("data/meal_data_gliner.json", "r", encoding="utf-8") as f:
    data = json.load(f)

unique_entities = []
for d in data:
    for n in d["ner"]:
        unique_entities.append((str(n[2]).lower()))
unique_entities = list(set(unique_entities))
out, f1 = model.evaluate(data, entity_types=unique_entities, batch_size=8)
print(out)


