import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

DATASET_PATH = 'data.csv'
SEED = 42

def load_data(path):
    print("--- 1. CARICAMENTO DATI ---")
    df = pd.read_csv(path, sep=';') 
    
    print("Dimensioni dataset:", df.shape)
    print("\nPrime righe del dataset:")
    print(df.head())
    
    return df

def preprocess_data(df):
    print("\n--- 2. PREPROCESSING (Manuale) ---")
    
    # Separazione Feature e Target
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Encoding del Target usando i Dizionari (Intro Python slide 17)
    # Sostituiamo le stringhe con numeri manualmente
    target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    print(f"Mappatura classi applicata: {target_mapping}")
    
    # Usiamo il metodo map che sfrutta la logica dei dizionari
    y_encoded = y.map(target_mapping)
    
    return X, y_encoded, list(target_mapping.keys())

def train_and_evaluate(X, y, class_names):
    print("\n--- 3. ADDESTRAMENTO E VALUTAZIONE ---")
    
    # Split Train/Test come visto nelle slide Naive Bayes
    # test_size=0.2 corrisponde al dividere i dati in 80% training e 20% test
    # stratify=y mantiene le proporzioni delle classi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    # Modelli trattati nelle lezioni
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=SEED, max_depth=10),
        "Gaussian NB": GaussianNB(), # GaussianNB gestisce i valori negativi del dataset (es. Inflation rate)
        "Random Forest": RandomForestClassifier(random_state=SEED, n_estimators=100)
    }
    
    # Dizionario per salvare il modello migliore (Random Forest)
    best_model_obj = None
    
    for name, model in models.items():
        print(f"\n>>> Training {name}...")
        start_time = time.time()
        
        # Training (metodo .fit visto nelle slide)
        model.fit(X_train, y_train)
        
        # Salviamo il modello Random Forest per l'analisi successiva
        if name == "Random Forest":
            best_model_obj = model
        
        # Predizione (metodo .predict visto nelle slide)
        y_pred = model.predict(X_test)
        
        # Calcolo metriche singole come da slide
        # Usiamo 'weighted' per gestire le 3 classi (simile alla logica spiegata per F1)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {acc:.2%}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Matrice di Confusione
        cm = confusion_matrix(y_test, y_pred)
        print("Matrice di Confusione:")
        print(cm)
        
        # Plot Matrice di Confusione usando solo Matplotlib
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {name}')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Aggiunta manuale dei numeri nelle celle
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('Valore Reale')
        plt.xlabel('Valore Predetto')
        plt.tight_layout()
        plt.savefig(f'cm_{name.replace(" ", "_")}.png')
        print(f"Grafico salvato: cm_{name.replace(' ', '_')}.png")
        plt.close()

    return best_model_obj, X.columns

def feature_importance_simple(model, feature_names):
    print("\n--- 4. FEATURE IMPORTANCE ---")
    # L'importanza delle feature è un attributo dell'albero addestrato
    importances = model.feature_importances_
    
    # Ordinamento usando argsort (NumPy)
    indices = np.argsort(importances)[::-1]
    
    print("Top 5 Feature:")
    for f in range(5):
        idx = indices[f]
        print(f"{f+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
    # Plot semplice con Matplotlib (pyplot.bar)
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(5), importances[indices[:5]], align="center")
    plt.xticks(range(5), feature_names[indices[:5]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Grafico salvato: feature_importance.png")

# --- MAIN ---
if __name__ == "__main__":
    # Caricamento
    df = load_data(DATASET_PATH)
    
    # Distribuzione classi (usando groupby e count come da slide Pandas)
    print("\nDistribuzione Classi:")
    print(df.groupby('Target').size())
    
    # Preprocessing
    X, y, class_names = preprocess_data(df)
    
    # Training e Valutazione
    best_model, feat_names = train_and_evaluate(X, y, class_names)
    
    # Feature Importance
    if best_model is not None:
        feature_importance_simple(best_model, feat_names)