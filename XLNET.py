import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import XLNetForSequenceClassification, XLNetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from torch.optim import AdamW
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

# Configurar dispositivo para GPU si está disponible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Usando el dispositivo: {device}')

# Cargar modelo y tokenizador de XLNet entrenado en español
model_name = "xlnet-base-cased"
model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=5)
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model.to(device)

# Leer datos
datos_editados = pd.read_excel("JHC_clases.xlsx")

# Dividir en train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    datos_editados['Frase'].astype(str),
    datos_editados['Consensuada'],
    test_size=0.2,
    random_state=42,
    stratify=datos_editados['Consensuada']
)

# Recuentos
for grupo, etiquetas in zip(["TOTAL", "TRAIN", "TEST"], [datos_editados['Consensuada'], train_labels, test_labels]):
    unique, counts = np.unique(etiquetas, return_counts=True)
    for i in range(len(unique)):
        print(f'Consensuada {grupo} {unique[i]}: Samples {counts[i]}')

# Tokenización
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=497)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=497)

# Conversión a tensores
train_labels_tensor = torch.tensor(train_labels.tolist()).to(device)
test_labels_tensor = torch.tensor(test_labels.tolist()).to(device)

train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']).to(device),
    torch.tensor(train_encodings['attention_mask']).to(device),
    train_labels_tensor
)

test_dataset = TensorDataset(
    torch.tensor(test_encodings['input_ids']).to(device),
    torch.tensor(test_encodings['attention_mask']).to(device),
    test_labels_tensor
)

# Crear generador con semilla fija para DataLoader
g = torch.Generator()
g.manual_seed(42)

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Optimizador y pérdida
optimizer = AdamW(model.parameters(), lr=5.45554789517329e-06, weight_decay=0.15306187719644712)
criterion = torch.nn.CrossEntropyLoss()

# Entrenamiento
model.train()
for epoch in range(23):
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
    'Estilo sin identificar',
    'Apoyo a la autonomía',
    'Estructura',
    'Control',
    'Caos'
]

# Informe
print(classification_report(all_labels, all_preds, target_names=class_names))

# Visualización
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión XLNet')
plt.tight_layout()
plt.show()
