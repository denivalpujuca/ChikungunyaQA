import json
import os
import argparse
from collections import Counter

def load_qas(file_path):
    if not file_path or not os.path.exists(file_path):
        return []
    qas = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        qas.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list): qas = data
                elif isinstance(data, dict):
                    # Tenta chaves comuns ou retorna o próprio dict em lista
                    qas = data.get("approved", []) or data.get("repaired", []) or data.get("discarded", []) or [data]
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
    return qas

def run_analysis(approved_path, repaired_path, discarded_path):
    approved = load_qas(approved_path)
    repaired = load_qas(repaired_path)
    discarded = load_qas(discarded_path)
    
    # Adiciona tags para controle interno
    for qa in approved: qa["_status"] = "approved"
    for qa in repaired: qa["_status"] = "repaired"
    for qa in discarded: qa["_status"] = "discarded"
    
    all_qas = approved + repaired + discarded
    total_questions = len(all_qas)
    
    if total_questions == 0:
        print("Nenhuma QA encontrada nos arquivos fornecidos.")
        return

    print("="*60)
    print("📊 ANÁLISE CONSOLIDADA DO DATASET (PIPELINE COMPLETO)")
    print("="*60)
    print(f"Total de QAs processadas: {total_questions}")
    print(f"Aprovação Direta: {len(approved)} ({len(approved)/total_questions:.1%})")
    print(f"Resgatadas (Reparadas): {len(repaired)} ({len(repaired)/total_questions:.1%})")
    print(f"Descartadas: {len(discarded)} ({len(discarded)/total_questions:.1%})")
    
    eficacia_resgate = (len(repaired) / (len(repaired) + len(discarded)) * 100) if (len(repaired) + len(discarded)) > 0 else 0
    print(f"Eficácia do Mecanismo de Resgate: {eficacia_resgate:.1f}%")
    print("-" * 60)

    # 1. Performance por Persona
    persona_stats = {}
    for qa in all_qas:
        p = qa.get("persona", "Desconhecida")
        if p not in persona_stats:
            persona_stats[p] = {"total": 0, "approved": 0, "repaired": 0, "discarded": 0}
        
        persona_stats[p]["total"] += 1
        status = qa.get("_status")
        persona_stats[p][status] += 1

    print("\n📈 PERFORMANCE POR PERSONA:")
    for p, s in persona_stats.items():
        taxa_sucesso = (s["approved"] + s["repaired"]) / s["total"]
        print(f"- {p:10}: {s['total']:3} QAs | Sucesso: {taxa_sucesso:5.1%} (A: {s['approved']:3}, R: {s['repaired']:3}, D: {s['discarded']:3})")

    # 2. Análise de Scores Médios (G-Eval) - Apenas das aprovadas/reparadas
    valid_qas = approved + repaired
    scores = [qa.get("judge", {}).get("score_medio", 0) for qa in valid_qas if "judge" in qa]
    if scores:
        print(f"\n⭐ Score Médio Final (Dataset Gold): {sum(scores)/len(scores):.2f} / 5.0")

    # 3. Análise de Motivos de Descarte
    reasons = [qa.get("judge", {}).get("reason", "") for qa in discarded if "judge" in qa]
    if reasons:
        print("\n🔍 TAXONOMIA DE ERROS (MOTIVOS DE DESCARTE):")
        common_errors = []
        for r in reasons:
            r_low = r.lower()
            if "alucinação" in r_low or "não está no texto" in r_low or "não contém" in r_low: common_errors.append("Alucinação/Extrapolação")
            elif "persona" in r_low or "voz" in r_low: common_errors.append("Inconsistência de Persona")
            elif "incompleta" in r_low or "omissão" in r_low or "exaustividade" in r_low: common_errors.append("Omissão de Informação")
            else: common_errors.append("Outros / Qualidade Geral")
        
        for error, count in Counter(common_errors).most_common():
            print(f"- {error:25}: {count} ocorrências ({count/len(discarded):.1%})")

    print("\n" + "="*60)

if __name__ == "__main__":
    _DEFAULT_APPROVED  = os.path.join("output", "gold_dataset_approved.jsonl")
    _DEFAULT_REPAIRED  = os.path.join("output", "gold_dataset_repaired.jsonl")
    _DEFAULT_DISCARDED = os.path.join("output", "gold_dataset_discarded.jsonl")

    parser = argparse.ArgumentParser(
        description="Analisador de Datasets RAG - le de output/ por padrao."
    )
    parser.add_argument("--approved",  default=_DEFAULT_APPROVED,  help="QAs aprovadas")
    parser.add_argument("--repaired",  default=_DEFAULT_REPAIRED,  help="QAs reparadas")
    parser.add_argument("--discarded", default=_DEFAULT_DISCARDED, help="QAs descartadas")
    parser.add_argument("file", nargs="?", help="Um unico arquivo JSON/JSONL")

    args = parser.parse_args()
    if args.file:
        run_analysis(args.file, None, None)
    else:
        run_analysis(args.approved, args.repaired, args.discarded)
