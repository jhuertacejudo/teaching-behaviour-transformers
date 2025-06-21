import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys

print("Python usado:", sys.executable)

# ------------------- SEMILLA GLOBAL -------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# ------------------------------------------------------

# Configurar dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

# Cargar modelo y tokenizador de RoBERTa en español
model_name = "BSC-TeMU/roberta-base-bne"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Congelar capas de RoBERTa
num_layers_frozen = 2  
for i in range(num_layers_frozen):
    for param in model.roberta.encoder.layer[i].parameters():
        param.requires_grad = False

model.to(device)

# Cargar datos
datos = pd.read_excel("JHC_clases.xlsx")

# Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    datos["Frase"].astype(str),
    datos["Consensuada"],
    test_size=0.2,
    stratify=datos["Consensuada"],
    random_state=42
)

# Recuentos
for grupo, etiquetas in zip(["TOTAL", "TRAIN", "TEST"], [datos["Consensuada"], train_labels, test_labels]):
    unique, counts = np.unique(etiquetas, return_counts=True)
    for i in range(len(unique)):
        print(f'Consensuada {grupo} {unique[i]}: Samples {counts[i]}')

# Tokenización
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=208)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=208)

# Tensores
train_labels_tensor = torch.tensor(train_labels.tolist()).to(device)
test_labels_tensor = torch.tensor(test_labels.tolist()).to(device)

train_dataset = TensorDataset(
    torch.tensor(train_encodings["input_ids"]).to(device),
    torch.tensor(train_encodings["attention_mask"]).to(device),
    train_labels_tensor
)

test_dataset = TensorDataset(
    torch.tensor(test_encodings["input_ids"]).to(device),
    torch.tensor(test_encodings["attention_mask"]).to(device),
    test_labels_tensor
)

# DataLoader con semilla fija
g = torch.Generator()
g.manual_seed(42)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, generator=g)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# Optimizador y función de pérdida
optimizer = AdamW(model.parameters(), lr=2.945810400413748e-05, weight_decay=0.0793754589408825)
criterion = torch.nn.CrossEntropyLoss()

# Entrenamiento
model.train()
for epoch in range(14):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_predictions += len(labels)

    accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

# Evaluación
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
class_names = [
    "Estilo sin identificar",
    "Apoyo a la autonomía",
    "Estructura",
    "Control",
    "Caos"
]

# Informe
print(classification_report(all_labels, all_preds, target_names=class_names))

# Visualización
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión RoBERTa")
plt.tight_layout()
plt.show()
