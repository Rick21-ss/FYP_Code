import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import open_clip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve, classification_report
)
from transformers import BioGptTokenizer, BioGptForCausalLM
from tqdm import tqdm

device = torch.device("cpu")
print("Using device:", device)

# 自定义 Dataset
class StanfordCXRDataset(Dataset):
    def __init__(self, dataset, tokenizer, transform=None):
        self.dataset = []
        iterable_dataset = list(dataset)
        for item in iterable_dataset:
            if isinstance(item, tuple) and isinstance(item[1], dict):
                item = item[1]
            if isinstance(item, dict):
                if 'findings' in item and ('image' in item or 'images' in item) and 'impression' in item:
                    self.dataset.append(item)
        if len(self.dataset) == 0:
            print("No valid samples found.")
        else:
            print("Sample keys from dataset[0]:", list(self.dataset[0].keys()))
        print(f"Loaded StanfordCXRDataset with {len(self.dataset)} valid samples.")
        self.tokenizer = tokenizer
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item.get('image') or item.get('images')
        if isinstance(image, list):
            image = image[0]
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), color="white")
        elif not isinstance(image, Image.Image):
            image = Image.new("RGB", (224, 224), color="white")
        image = self.transform(image)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        elif image.size(0) != 3:
            image = image[:3, :, :]

        text = item.get('findings', 'no findings')
        tokens = self.tokenizer(text)
        if hasattr(tokens, "input_ids"):
            tokens = tokens.input_ids
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
            tokens = tokens[0]
        if isinstance(tokens, list) and all(isinstance(t, int) for t in tokens):
            tokens = tokens[:77] + [0] * max(0, 77 - len(tokens))
        else:
            tokens = [0] * 77

        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        findings = item.get("findings", "").lower()
        label_keywords = {
            0: "pneumonia",
            1: "cardiomegaly",
            2: "edema",
            3: "effusion",
            4: "atelectasis",
            5: "infiltrate",
            6: "mass",
            7: "nodule",
            8: "consolidation",
            9: "fibrosis",
            10: "pleural thickening",
            11: "hernia",
            12: "emphysema",
            13: "fracture"
        }
        labels = [0] * 14
        for idx, keyword in label_keywords.items():
            if keyword in findings:
                labels[idx] = 1

        impression = item.get('impression', '')

        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "impression": impression
        }

    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    batch = [item for item in batch if item is not None and isinstance(item, dict)]
    if len(batch) == 0:
        return None
    impression_list = [item['impression'] for item in batch]
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'impression': impression_list
    }

# 模型
class MedCLIPWithMLPClassifier(nn.Module):
    def __init__(self, base_model, embedding_dim=512, hidden_dim=512, num_labels=14):
        super().__init__()
        self.base_model = base_model
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, image, text):
        image_features = self.base_model.encode_image(image)
        text_features = self.base_model.encode_text(text)
        features = (image_features + text_features) / 2
        x = self.fc1(features)
        x = self.relu(x)
        return self.fc2(x)

# 加载模型和 tokenizer
model_name = "ViT-B-16"
clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)
model = MedCLIPWithMLPClassifier(clip_model).to(device)

# 加载BioGPT模型和分词器用于报告生成
bio_gpt_tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
bio_gpt_model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT").to(device)

# 加载数据
raw = load_dataset("StanfordAIMI/interpret-cxr-test-public")["test"]
train_data, val_data = train_test_split(list(raw), test_size=0.2, random_state=42)

train_dataset = StanfordCXRDataset(train_data, tokenizer)
val_dataset = StanfordCXRDataset(val_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 训练
epochs = 1
losses = []
for epoch in range(epochs):
    model.train()
    # 使用 tqdm 包装训练数据加载器以显示进度条
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
    for batch in progress_bar:
        if batch is None:
            continue
        image = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(image, input_ids)
        if outputs.shape != labels.shape:
            print(f"Shape mismatch: outputs {outputs.shape}, labels {labels.shape}")
            continue
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # 更新进度条的显示信息，显示当前损失
        progress_bar.set_postfix({'loss': loss.item()})
    print(f"Epoch {epoch + 1} done")

# 保存模型
torch.save(model.state_dict(), "medclip_classifier.pt")
print("Model saved to medclip_classifier.pt")

# 损失图
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig("training_loss.png")
print("Saved training loss curve.")

# 验证
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        if batch is None:
            continue
        image = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].cpu().numpy()
        outputs = model(image, input_ids)
        preds = torch.sigmoid(outputs).cpu().numpy()
        if preds.shape != labels.shape:
            continue
        all_preds.append(preds)
        all_labels.append(labels)

if len(all_preds) == 0:
    print("No predictions to evaluate.")
else:
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    pred_binary = all_preds >= 0.5
    acc = accuracy_score(all_labels.flatten(), pred_binary.flatten())
    f1 = f1_score(all_labels.flatten(), pred_binary.flatten(), average="macro")
    try:
        auc = roc_auc_score(all_labels, all_preds, average="macro")
    except:
        auc = -1
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print("Classification report:")
    print(classification_report(all_labels, pred_binary, zero_division=0))

    # 绘制 PR 曲线
    plt.figure(figsize=(12, 8))
    for i in range(14):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_preds[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.savefig("pr_curves.png")
    print("Saved Precision-Recall curve.")

# 使用BioGPT生成报告
model.eval()
with torch.no_grad():
    for batch in val_loader:
        if batch is None:
            continue
        image = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        impressions = batch['impression']
        for i in range(len(image)):
            single_image = image[i].unsqueeze(0)
            if input_ids.shape[0] == 1:
                single_input_ids = input_ids
            else:
                single_input_ids = input_ids[i].unsqueeze(0)
            image_features = clip_model.encode_image(single_image)
            text_features = clip_model.encode_text(single_input_ids)
            features = (image_features + text_features) / 2

            # 构建输入提示文本，包含impression部分
            prompt = f"根据医学图像、病症信息和印象：{impressions[i]}，生成以下放射学报告："
            input_ids_bio = bio_gpt_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            output = bio_gpt_model.generate(
                input_ids=input_ids_bio,
                max_length=300,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            generated_report = bio_gpt_tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Generated Report: {generated_report}")