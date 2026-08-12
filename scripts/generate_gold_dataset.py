import json
import os
import time
from pipeline.data_ingestion import load_documents, split_documents
from pipeline.qa_generator import QAGenerator

# ── Configuração ────────────────────────────────────────────────────────────
QUESTIONS_PER_PERSONA_PER_PASS = 2   # QAs tentadas por persona em cada passagem
MAX_PASSES_PER_CHUNK           = 6   # Limite de segurança: máx 6 passagens por chunk
OUTPUT_DIR                     = "output"
SATURATED_FILE                 = os.path.join(OUTPUT_DIR, "saturated_chunks.json")


# ── Persistência de saturação ────────────────────────────────────────────────
def load_saturated(path=SATURATED_FILE) -> set:
    """Carrega o conjunto de chunk_indices já totalmente esgotados."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def mark_saturated(i: int, path=SATURATED_FILE):
    """Adiciona chunk_index ao registro de chunks esgotados."""
    indices = load_saturated(path)
    indices.add(i)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(indices), f)


# ── Salvar resultados de uma passagem ────────────────────────────────────────
def save_results(results: dict, chunk_index: int, pass_num: int):
    for cat in ["approved", "repaired", "discarded"]:
        for qa in results[cat]:
            qa["chunk_index"] = chunk_index
            qa["pass_num"]    = pass_num   # rastreabilidade de qual passagem gerou

    with open(os.path.join(OUTPUT_DIR, "gold_dataset_approved.jsonl"), "a", encoding="utf-8") as f:
        for qa in results["approved"]:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    with open(os.path.join(OUTPUT_DIR, "gold_dataset_repaired.jsonl"), "a", encoding="utf-8") as f:
        for qa in results["repaired"]:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    with open(os.path.join(OUTPUT_DIR, "gold_dataset_discarded.jsonl"), "a", encoding="utf-8") as f:
        for qa in results["discarded"]:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")


# ── Processamento exaustivo de um único chunk ─────────────────────────────────
def exhaust_chunk(qa_gen: QAGenerator, chunk, chunk_index: int) -> int:
    """
    Roda passagens sucessivas no chunk até que nenhuma QA nova seja aprovada
    (saturação) ou MAX_PASSES seja atingido.
    Retorna o total de QAs aprovadas (diretas + reparadas).
    """
    total_approved = 0

    for pass_num in range(1, MAX_PASSES_PER_CHUNK + 1):
        print(f"  [passagem {pass_num}/{MAX_PASSES_PER_CHUNK}]", end=" ", flush=True)

        results = qa_gen.run_pipeline(
            [chunk],
            questions_per_chunk=QUESTIONS_PER_PERSONA_PER_PASS,
            limit=1,
            exhaustive=True,
        )

        approved_this_pass = len(results["approved"]) + len(results["repaired"])
        save_results(results, chunk_index, pass_num)
        total_approved += approved_this_pass

        print(f"{approved_this_pass} aprovadas | {len(results['discarded'])} descartadas")

        # ── Critério de saturação: nenhuma QA útil nesta passagem ──
        if approved_this_pass == 0:
            print(f"  [saturação] Chunk {chunk_index} esgotado após {pass_num} passagem(ns).")
            break

    return total_approved


# ── Loop Principal ────────────────────────────────────────────────────────────
def main():
    print("--- Iniciando Geração Exaustiva por Saturação ---")
    print(f"    {QUESTIONS_PER_PERSONA_PER_PASS} QAs/persona/passagem | Máx {MAX_PASSES_PER_CHUNK} passagens por chunk\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    docs   = load_documents("./docs")
    chunks = split_documents(docs)

    qa_gen = QAGenerator()

    # Chunks já totalmente esgotados em execuções anteriores
    saturated = load_saturated()
    print(f"Total de chunks   : {len(chunks)}")
    print(f"Já saturados      : {len(saturated)}")
    print(f"A processar       : {len(chunks) - len(saturated)}\n")

    t0 = time.time()
    chunks_processados = 0
    total_qa_sessao    = 0

    for i, chunk in enumerate(chunks):
        if i in saturated:
            continue

        print(f"\n[Chunk {i}/{len(chunks) - 1}] ({len(chunk.page_content)} chars)")

        try:
            aprovadas = exhaust_chunk(qa_gen, chunk, i)
            total_qa_sessao += aprovadas
            chunks_processados += 1
            mark_saturated(i)   # só marca como esgotado após concluir todas as passagens

        except Exception as e:
            print(f"\n[ERROR] Chunk {i}: {e}")
            print("Pausando. Verifique créditos/conexão e reinicie o script.")
            break

    t1 = time.time()
    print(f"\n{'='*50}")
    print(f"  Sessão finalizada")
    print(f"  Chunks processados : {chunks_processados}")
    print(f"  QAs aprovadas      : {total_qa_sessao}")
    print(f"  Tempo              : {t1-t0:.0f}s ({(t1-t0)/60:.1f} min)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
