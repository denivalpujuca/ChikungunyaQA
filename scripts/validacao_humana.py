import pandas as pd
import scipy.stats
from sklearn.metrics import cohen_kappa_score
import os

# Nome dos arquivos
ARQUIVO_AMOSTRA = "chikungunya_human_validation_sample.csv"
ARQUIVO_ANOTADO = "validacao_humana_anotada.csv"

def gerar_template_anotacao():
    """Lê a amostra gerada pelo app.py e prepara uma coluna vazia para o humano preencher"""
    if not os.path.exists(ARQUIVO_AMOSTRA):
        print(f"ERROR: Arquivo '{ARQUIVO_AMOSTRA}' não encontrado.")
        print("Dica: Gere o dataset e salve a amostra humana.")
        return

    df = pd.read_csv(ARQUIVO_AMOSTRA)
    
    # Se a coluna 'Nota Humana' já existir, não sobrescreve
    if "Nota Humana" not in df.columns:
        df["Nota Humana"] = ""  # Coluna vazia para o humano preencher
        
    df.to_csv(ARQUIVO_ANOTADO, index=False)
    print("="*60)
    print(f"SUCCESS: Template de Validacao Criado: {ARQUIVO_ANOTADO}")
    print("="*60)
    print("O que voce deve fazer agora:")
    print("1. Abra o arquivo 'validacao_humana_anotada.csv' no Excel.")
    print("2. Leia a 'Pergunta' e a 'Resposta'.")
    print("3. Na coluna 'Nota Humana', coloque uma nota de 0 a 100 (ex: 80, 100, 20).")
    print("   -> (Não precisa olhar a coluna do G-Eval para não enviesar sua nota).")
    print("4. Salve o CSV e rode este script novamente!")
    print("="*60)

def calcular_spearman():
    """Lê as anotações preenchidas e calcula a Correlação de Spearman"""
    if not os.path.exists(ARQUIVO_ANOTADO):
        print("Arquivo anotado não encontrado. Gerando template inicial...")
        gerar_template_anotacao()
        return

    df = pd.read_csv(ARQUIVO_ANOTADO)
    
    # Limpa linhas vazias na coluna Humana
    df_validos = df.dropna(subset=["Nota Humana"])
    
    if len(df_validos) == 0:
        print("⚠️ Nenhuma nota humana foi preenchida ainda.")
        print("Abra o 'validacao_humana_anotada.csv' e preencha a coluna 'Nota Humana' com valores numéricos.")
        return
        
    # Extração das notas: 
    # Converte string (ex: "80%") para número (80.0) se necessário
    try:
        # Pega a nota G-Eval, remove o % se houver, converte para float
        g_eval_raw = df_validos["G-Eval Score"].astype(str).str.replace("%", "").astype(float)
        
        # Pega a nota humana, garantindo que seja lida como float
        humana_raw = df_validos["Nota Humana"].astype(str).str.replace("%", "").astype(float)
    except Exception as e:
        print(f"ERROR ao processar as notas. Certifique-se de usar apenas números na coluna Humana. Detalhe: {e}")
        return

    print("="*60)
    print(" 🔬 EXPERIMENTO: ALINHAMENTO HUMANO-MÁQUINA")
    print(f" Amostras analisadas: {len(df_validos)}")
    print("="*60)
    
    # Cálculo Spearman
    spearman_corr, p_value = scipy.stats.spearmanr(g_eval_raw, humana_raw)
    
    # Cálculo Kappa (Categorização Binária: Aprovado se >= 80)
    limiar = 80
    g_eval_bin = (g_eval_raw >= limiar).astype(int)
    humana_bin = (humana_raw >= limiar).astype(int)
    kappa = cohen_kappa_score(g_eval_bin, humana_bin)

    print(f"Coeficiente de Spearman (rho): {spearman_corr:.4f}")
    print(f"P-Value (Spearman): {p_value:.4f}")
    print(f"Coeficiente Kappa de Cohen: {kappa:.4f}")
    print("-"*60)
    
    # Interpretação Acadêmica Automática
    print("INTERPRETACAO SPEARMAN:")
    if pd.isna(spearman_corr):
        print("WARN: Correlacao nao pode ser calculada (Falta de variancia). Todas as notas foram identicas.")
    elif spearman_corr > 0.7:
        print("SUCCESS: FORTE CORRELACAO: O G-Eval tem alto alinhamento com os humanos!")
    elif spearman_corr > 0.4:
        print("WARN: CORRELACAO MODERADA: O G-Eval concorda razoavelmente.")
    else:
        print("ERROR: CORRELACAO FRACA: O G-Eval nao concorda com os humanos.")
        
    print("\nINTERPRETACAO KAPPA (Acordo sobre Aprovação/Rejeição):")
    if kappa > 0.8:
        print("SUCCESS: ACORDO QUASE PERFEITO!")
    elif kappa > 0.6:
        print("SUCCESS: ACORDO SUBSTANCIAL.")
    elif kappa > 0.4:
        print("WARN: ACORDO MODERADO.")
    else:
        print("ERROR: ACORDO FRACO ou BAIXO.")

    print("-"*60)
    if p_value < 0.05:
        print("SUCCESS: Significância Estatística: O resultado é estatisticamente válido (p < 0.05).")
    else:
        print("WARN: Sem Significancia: O p-value e alto (> 0.05). Voce precisa anotar mais exemplos para provar o valor matematicamente.")
    print("="*60)

if __name__ == "__main__":
    import sys
    # Simples roteador: se rodar o script sem a coluna preenchida, gera o template. Se já tiver preenchido, calcula.
    if not os.path.exists(ARQUIVO_ANOTADO):
        gerar_template_anotacao()
    else:
        df_check = pd.read_csv(ARQUIVO_ANOTADO)
        if "Nota Humana" in df_check.columns and not df_check["Nota Humana"].isnull().all():
            calcular_spearman()
        else:
            print("INFO: A coluna 'Nota Humana' ainda esta vazia no 'validacao_humana_anotada.csv'.")
            print("Por favor, preencha as notas e rode novamente.")
