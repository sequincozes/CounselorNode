import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def calculate_metrics_for_node(node_number):
    # Path to the CSV file
    csv_path = f'../logs/no_{node_number}_classificacoes.csv'
    
    if not os.path.exists(csv_path):
        print(f"Arquivo {csv_path} não encontrado.")
        return
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Extract ground truth and predictions
    y_true = df['ground_truth']
    y_pred = df['final_decision']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Print results
    print(f"Métricas para o nó {node_number}:")
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred))
    print("\nMatriz de Confusão:")
    print(conf_matrix)
    print("-" * 50)
    
    # Save to a file
    os.makedirs('../metrics', exist_ok=True)
    with open(f'../metrics/no_{node_number}_metrics.txt', 'w') as f:
        f.write(f"Métricas para o nó {node_number}:\n")
        f.write(f"Acurácia: {accuracy:.4f}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_true, y_pred))
        f.write("\nMatriz de Confusão:\n")
        f.write(str(conf_matrix))
    
    # Plot confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusão - Nó {node_number}')
    plt.savefig(f'../metrics/no_{node_number}_confusion_matrix.png')
    plt.close()  # Close to avoid display issues

def main():
    for node in [1, 2, 3]:
        calculate_metrics_for_node(node)

if __name__ == "__main__":
    main()