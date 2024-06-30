import json
import random
import typer
import os
import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.data_processing import WordsSplitter, GLiNERDataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"

train_path = "data/meal_data_gliner.json"

with open(train_path, "r", encoding="utf-8") as f:
    data = json.load(f)

typer.secho(f"Data size: {len(data)}", fg=typer.colors.GREEN)
random.shuffle(data)
typer.secho("Data shuffled successfully!", fg=typer.colors.GREEN)

train_data = data[:int(len(data) * 0.9)]
test_data = data[int(len(data) * 0.9):]
typer.secho("Dataset is splitted.", fg=typer.colors.GREEN)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
typer.secho(f"Device: {device}", fg=typer.colors.GREEN)

model = GLiNER.from_pretrained("urchade/gliner_small-v1")

train_dataset = GLiNERDataset(train_data, model.config, data_processor=model.data_processor)
test_dataset = GLiNERDataset(test_data, model.config, data_processor=model.data_processor)

data_collator = DataCollatorWithPadding(model.config)

torch.set_float32_matmul_precision('high')
model.to(device)
model.compile_for_training()

training_args = TrainingArguments(
    output_dir="models",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="linear",  # cosine
    warmup_ratio=0.1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_steps=1000,
    save_total_limit=10,
    dataloader_num_workers=8,
    use_cpu=False,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=model.data_processor.transformer_tokenizer,
    data_collator=data_collator,
)
trainer.train()
