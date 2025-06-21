import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys

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

print("Python usado:", sys.executable)

# Configurar dispositivo para GPU si está disponible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Usando el dispositivo: {device}')

# Cargar el modelo y el tokenizador pre-entrenado de ELECTRA
model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=5)
tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator", do_lower_case=False)

for name, param in model.electra.encoder.layer[:3].named_parameters():
    param.requires_grad = False

model.to(device)

# Cargar datos
datos_editados = pd.read_excel("JHC_clases.xlsx")
print(len(datos_editados))

train_texts, test_texts, train_labels, test_labels = train_test_split(
    datos_editados['Frase'].astype(str),
    datos_editados['Consensuada'],
    test_size=0.2,
    random_state=42,
    stratify=datos_editados['Consensuada']
)

# Recuentos por clase
y = datos_editados['Consensuada']
for grupo, etiquetas in zip(["TOTAL", "TRAIN", "TEST"], [y, train_labels, test_labels]):
    unique, counts = np.unique(etiquetas, return_counts=True)
    for i in range(len(unique)):
        print(f'Consensuada {grupo} {unique[i]}: Samples {counts[i]}')

# Tokenización y creación de datasets
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=277)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=277)

train_labels = torch.tensor(train_labels.tolist()).to(device)
test_labels = torch.tensor(test_labels.tolist()).to(device)

train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']).to(device),
    torch.tensor(train_encodings['attention_mask']).to(device),
    train_labels
)

test_dataset = TensorDataset(
    torch.tensor(test_encodings['input_ids']).to(device),
    torch.tensor(test_encodings['attention_mask']).to(device),
    test_labels
)

# DataLoaders con generador para reproducibilidad
g = torch.Generator()
g.manual_seed(42)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, generator=g)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# Optimización
optimizer = AdamW(model.parameters(), lr=1.9758563822509995e-05, weight_decay=0.14199816723373726)
criterion = torch.nn.CrossEntropyLoss()

# Entrenamiento
model.train()
for epoch in range(22):
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

        predictions = torch.argmax(outputs.logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
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
        _, preds = torch.max(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Matriz de confusión
class_names = [
    'Estilo sin identificar',
    'Apoyo a la autonomía',
    'Estructura',
    'Control',
    'Caos'
]

cm = confusion_matrix(all_labels, all_preds)
print(classification_report(all_labels, all_preds, target_names=class_names))

# Visualización
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión ELECTRA')
plt.tight_layout()
plt.show()
