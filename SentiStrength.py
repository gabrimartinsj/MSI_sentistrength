from sentistrength import PySentiStr
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Função para ler dados usando pandas
def read_text_from_csv(file_path):
    return pd.read_csv(file_path, header=None, names=['text'])

# Função para analisar o texto
def analyze_text(text):
    text = str(text).strip()
    print(f"Processando texto: {text}")
    try:
        result_scale = sentistrength.getSentiment(text, score='scale')[0]
        result_dual = sentistrength.getSentiment(text, score='dual')[0]
        result_trinary = sentistrength.getSentiment(text, score='trinary')[0]
        print(f"Texto processado: {text}")
    except Exception as e:
        print(f"Erro ao processar o texto: {text}\nErro: {e}")
        result_scale = 0
        result_dual = (0, 0)
        result_trinary = (0, 0, 0)
    return text, result_scale, result_dual, result_trinary

SENTISTRENGTH_DATA_PATH = r'C:\Users\gabriel.juarez_a3dat\Documents\Projetos\Pessoal\MSI2\SentiStrength\SentiStrength_Data'
SENTISTRENGTH_JAR_PATH = r'C:\Users\gabriel.juarez_a3dat\Documents\Projetos\Pessoal\MSI2\SentiStrength\SentiStrength.jar'

# Ler os dados de um arquivo CSV
file_path = "youtube_data_messages.csv"
df = read_text_from_csv(file_path)

sentistrength = PySentiStr()
sentistrength.setSentiStrengthPath(SENTISTRENGTH_JAR_PATH)
sentistrength.setSentiStrengthLanguageFolderPath(SENTISTRENGTH_DATA_PATH)

# Processamento paralelo
print("Iniciando processamento paralelo...")
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(analyze_text, df['text']))

# Converter os resultados em DataFrame
print("Convertendo resultados em DataFrame...")
df_results = pd.DataFrame(results, columns=['text', 'result_scale', 'result_dual', 'result_trinary'])

print("Calculando estatísticas...")

# Estatísticas básicas
mean_scale = df_results["result_scale"].mean()
positive_counts = (df_results["result_trinary"].apply(lambda x: x[2]) == 1).sum()
negative_counts = (df_results["result_trinary"].apply(lambda x: x[2]) == -1).sum()
neutral_counts = (df_results["result_trinary"].apply(lambda x: x[2]) == 0).sum()

# Exibir os resultados agregados
print(f"Média da Escala: {mean_scale}")
print(f"Textos Positivos: {positive_counts}")
print(f"Textos Negativos: {negative_counts}")
print(f"Textos Neutros: {neutral_counts}")

# Estatísticas detalhadas dos resultados duais
mean_positive_strength = df_results["result_dual"].apply(lambda x: x[0]).mean()
mean_negative_strength = df_results["result_dual"].apply(lambda x: x[1]).mean()

print(f"Média da Força Positiva: {mean_positive_strength}")
print(f"Média da Força Negativa: {mean_negative_strength}")

# Identificar textos com maior e menor pontuação de escala
top_5_max_scale = df_results.nlargest(5, 'result_scale')
top_5_min_scale = df_results.nsmallest(5, 'result_scale')

print("Textos com maior pontuação de escala:")
print(top_5_max_scale[['text', 'result_scale']])

print("Textos com menor pontuação de escala:")
print(top_5_min_scale[['text', 'result_scale']])

# Salvar os textos em um arquivo de texto
with open("top_texts.txt", "w", encoding="utf-8") as f:
    f.write("Textos com maior pontuação de escala:\n")
    for idx, row in top_5_max_scale.iterrows():
        f.write(f"{row['result_scale']}: {row['text']}\n")

    f.write("\nTextos com menor pontuação de escala:\n")
    for idx, row in top_5_min_scale.iterrows():
        f.write(f"{row['result_scale']}: {row['text']}\n")

# Plotar gráficos
plt.figure(figsize=(10, 5))

# Histograma das Escalas
plt.subplot(1, 2, 1)
plt.hist(df_results['result_scale'], bins=range(-5, 6), edgecolor='black')
plt.title('Distribuição da Escala de Sentimentos')
plt.xlabel('Escala de Sentimento')
plt.ylabel('Frequência')

# Contagem de Sentimentos Positivos, Negativos e Neutros
plt.subplot(1, 2, 2)
sentiment_counts = [positive_counts, negative_counts, neutral_counts]
sentiment_labels = ['Positivos', 'Negativos', 'Neutros']
plt.bar(sentiment_labels, sentiment_counts, color=['green', 'red', 'blue'])
plt.title('Contagem de Sentimentos')
plt.xlabel('Sentimento')
plt.ylabel('Frequência')

plt.tight_layout()
plt.show()
