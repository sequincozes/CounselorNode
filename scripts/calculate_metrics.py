import pandas as pd
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def calculate_metrics(node_number):
    # Path to the CSV file
    csv_path = f'../logs/no_{node_number}_decisoes.csv'
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Extract ground truth and predictions
    y_true = df['ground_truth']
    y_pred = df['decisao']
    
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
    
    # Optionally, save to a file
    with open(f'../metrics/no_{node_number}_metrics.txt', 'w') as f:
        f.write(f"Métricas para o nó {node_number}:\n")
        f.write(f"Acurácia: {accuracy:.4f}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_true, y_pred))
        f.write("\nMatriz de Confusão:\n")
        f.write(str(conf_matrix))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python calculate_metrics.py <node_number>")
        sys.exit(1)
    
    node_number = sys.argv[1]
    calculate_metrics(node_number)