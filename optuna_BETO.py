import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import random

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar datos
datos_editados = pd.read_excel("JHC_clases.xlsx")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    datos_editados['Frase'].astype(str),
    datos_editados['Consensuada'],
    test_size=0.2,
    random_state=42,
    stratify=datos_editados['Consensuada']
)

# Cargar tokenizer una sola vez
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", do_lower_case=False)

def objective(trial):
    # Hiperparámetros a optimizar
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 50)
    num_layers_frozen = trial.suggest_int("num_layers_frozen", 0, 11)
    max_len = trial.suggest_int("max_len", 16, 512)

    # Tokenización con max_len dinámico
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=max_len)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=max_len)

    train_labels_tensor = torch.tensor(train_labels.tolist())
    val_labels_tensor = torch.tensor(val_labels.tolist())

    train_dataset = TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        train_labels_tensor
    )

    val_dataset = TensorDataset(
        torch.tensor(val_encodings['input_ids']),
        torch.tensor(val_encodings['attention_mask']),
        val_labels_tensor
    )

    # Cargar modelo y congelar capas
    model = BertForSequenceClassification.from_pretrained(
        "dccuchile/bert-base-spanish-wwm-cased", num_labels=5
    ).to(device)

    for i in range(num_layers_frozen):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Entrenamiento
    model.train()
    for epoch in range(num_train_epochs):
        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    # Evaluación
    model.eval()
    preds, labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predictions = torch.max(outputs.logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = accuracy_score(labels_list, preds)
    return accuracy

# Ejecutar la optimización
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  

# Resultados
print("Mejores hiperparámetros encontrados:")
print(study.best_params)
print(f"Mejor accuracy: {study.best_value:.4f}")
