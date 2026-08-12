"""
qa_generator.py
───────────────
Arquitetura híbrida:
  • Gerador  → GPT-4o-mini            (chat_openai — geração de QAs e reparo)
  • Juiz     → Claude Sonnet 4.5      (chat_judge  — g-Eval 4 dimensões)

Todos os prompts vivem em prompts.py — fonte única de verdade.
"""

import json
import os
import random
from typing import Dict, List, Optional

from langchain_openai import OpenAIEmbeddings

from .llm import chat_openai, chat_judge, GENERATOR_MODEL, JUDGE_MODEL
from .prompts import (
    GENERATOR_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    QA_CHUNK_SCREENING_PROMPT,
    QA_GENERATION_PROMPT,
    QA_JUDGE_GEVAL_PROMPT,
    QA_JUDGE_BATCH_GEVAL_PROMPT,
    QA_REPAIR_PROMPT,
)


# ── Configuração ──────────────────────────────────────────────────────────────
# Agora herdados centralmente do llm.py
JUDGE_APPROVAL_THRESHOLD = 90.0   # score médio mínimo (0-100) para aprovação (gold standard)

# ── Persistência de tópicos recentes ─────────────────────────────────────────
_TOPICS_FILE = os.path.join("output", "recent_topics.json")
_MAX_TOPICS  = 200  # quantos tópicos manter em disco



# ═════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════
class QAGenerator:
    def __init__(
        self,
        generator_model: str = GENERATOR_MODEL,
        judge_model: str = JUDGE_MODEL,
        judge_threshold: float = JUDGE_APPROVAL_THRESHOLD,
    ):
        self.generator_model = generator_model
        self.judge_model     = judge_model
        self.judge_threshold = judge_threshold
        self.recent_topics: List[str] = self._load_topics()  # Persistido entre sessões
        
        # Inicializa embeddings para desduplicação semântica
        self.embeddings = OpenAIEmbeddings()
        if self.recent_topics:
            print(f"  [init] Pré-computando embeddings de {len(self.recent_topics)} tópicos recentes...")
            self.recent_topics_embeddings = self.embeddings.embed_documents(self.recent_topics)
        else:
            self.recent_topics_embeddings = []

    # ── Persistência de tópicos ───────────────────────────────────────────────
    @staticmethod
    def _load_topics() -> List[str]:
        """Carrega o histórico de tópicos do disco (sobrevive entre sessões)."""
        try:
            if os.path.exists(_TOPICS_FILE):
                with open(_TOPICS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_topics(self) -> None:
        """Persiste os tópicos recentes no disco."""
        try:
            os.makedirs("output", exist_ok=True)
            with open(_TOPICS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.recent_topics[-_MAX_TOPICS:], f, ensure_ascii=False)
        except Exception as e:
            print(f"  [tópicos] Erro ao salvar: {e}")

    # ── Triagem de chunk com escopo por persona ───────────────────────────────
    def _screen_chunk(self, context: str) -> tuple[bool, str, dict]:
        """
        Verifica se o chunk tem conteúdo factual suficiente e quais personas
        podem gerar QAs respondíveis integralmente pelo texto.

        Retorna (aprovado: bool, motivo: str, personas_validas: dict).
        personas_validas = {"Paciente": bool, "Cuidador": bool, "Medica": bool}
        """
        prompt = QA_CHUNK_SCREENING_PROMPT.format(context=context[:2000])
        try:
            response = chat_openai(
                prompt,
                model=self.generator_model,
                temperature=0,
                system_prompt="Responda APENAS com JSON válido, sem markdown.",
            )
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            data = json.loads(text.strip())
            aprovado = bool(data.get("aprovado", False))
            motivo   = data.get("motivo", "")
            personas = data.get("personas_validas", {
                "Paciente": True, "Cuidador": True, "Medica": True
            })
            return aprovado, motivo, personas
        except Exception as e:
            print(f"[QAGenerator] Erro na triagem: {e}")
            # fallback seguro: aprova com todas as personas
            return True, "Triagem falhou — prosseguindo por segurança.", {
                "Paciente": True, "Cuidador": True, "Medica": True
            }

    # ── Geração (GPT-4o-mini, 1 chamada por chunk) ────────────────────────────
    def generate(
        self,
        chunks: List,
        num_questions: int = 3,
        chunk_index: Optional[int] = None,
        extra_instructions: str = "",
    ) -> List[Dict]:
        """
        Gera QAs para um único chunk.
        num_questions = total desejado, dividido entre as personas habilitadas.
        Chunks sem conteúdo QA-gerável são descartados antes da geração.
        Personas bloqueadas pela triagem não são enviadas ao gerador.
        """
        if not chunks:
            return []

        idx     = chunk_index if chunk_index is not None else 0
        context = chunks[idx].page_content

        # ── Triagem: rejeita chunks ruins e bloqueia personas por escopo ──────
        aprovado, motivo, personas_validas = self._screen_chunk(context)
        if not aprovado:
            print(f"  [triagem] Chunk rejeitado — {motivo}")
            return []

        personas_ativas = [p for p, ok in personas_validas.items() if ok]
        bloqueadas      = [p for p, ok in personas_validas.items() if not ok]
        if bloqueadas:
            print(f"  [triagem] Personas bloqueadas para este chunk: {bloqueadas}")
        if not personas_ativas:
            print(f"  [triagem] Nenhuma persona válida — chunk ignorado.")
            return []

        n_personas   = len(personas_ativas)
        per_persona  = max(1, num_questions // max(n_personas, 1))
        total        = per_persona * n_personas

        # Injeta no prompt apenas as personas habilitadas
        personas_str = " | ".join(
            f"{'①' if p == 'Paciente' else '②' if p == 'Cuidador' else '③'} {p.upper()}"
            for p in personas_ativas
        )
        prompt = QA_GENERATION_PROMPT.format(
            num_questions_per_persona=per_persona,
            total_questions=total,
            context=context[:6000],
        )
        # Substitui a seção de personas no prompt para listar apenas as ativas
        personas_block = self._build_personas_block(personas_ativas)
        prompt = prompt.replace(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nESTÁGIO 2 — GERAÇÃO DE QAs",
            f"Personas habilitadas para ESTE chunk: {personas_str}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nESTÁGIO 2 — GERAÇÃO DE QAs"
        )
        if extra_instructions:
            prompt += f"\n\n{extra_instructions}"

        try:
            response = chat_openai(
                prompt,
                model=self.generator_model,
                temperature=0.6,  # Aumentado para maior diversidade linguística
                system_prompt=GENERATOR_SYSTEM_PROMPT,
            )
            qas = self._parse_response(response)
            # Filtra QAs de personas bloqueadas que vazaram
            qas = [qa for qa in qas if qa.get("persona") in personas_ativas]
            qas = self._validate_source_chunks(qas, context)
            return self._deduplicate_intra(qas)
        except Exception as e:
            print(f"[QAGenerator] Erro na geração: {e}")
            return []

    @staticmethod
    def _build_personas_block(personas_ativas: List[str]) -> str:
        """Constrói descrição textual das personas ativas — para log interno."""
        labels = {"Paciente": "pessoa leiga", "Cuidador": "familiar/cuidador",
                  "Medica": "profissional de saúde"}
        return ", ".join(f"{p} ({labels.get(p, '')})" for p in personas_ativas)

    # ── Geração em lote ───────────────────────────────────────────────────────
    def generate_batch(
        self,
        chunks: List,
        questions_per_chunk: int = 1,
        limit: int = 50,
        exhaustive: bool = False,
    ) -> List[Dict]:
        """Gera QAs para múltiplos chunks com deduplicação cross-chunk."""
        all_qa: List[Dict]  = []
        seen_questions: set = set()

        indices = list(range(len(chunks)))
        if not exhaustive:
            random.shuffle(indices)
            indices = indices[:limit]

        mode = "EXAUSTIVO" if exhaustive else "Lote"
        print(f"Generating QAs in mode {mode} ({len(indices)} chunks)...")
        print(f"  [memória] Tópicos conhecidos: {len(self.recent_topics)}")

        for i in indices:
            # Injeta tópicos recentes no prompt para evitar repetição
            history_str = "\n".join([f"- {t}" for t in self.recent_topics[-20:]])
            current_prompt_suffix = f"\n\nTEMAS JÁ COBERTOS (EVITE REPETIR EXATAMENTE ESTES): \n{history_str}" if self.recent_topics else ""
            
            qas = self.generate(
                chunks=[chunks[i]],
                num_questions=questions_per_chunk * 3,
                chunk_index=0,
                extra_instructions=current_prompt_suffix
            )
            for qa in qas:
                q_text = qa.get("question", "").strip()
                
                # ── Dedup Semântico Cross-Chunk (Global) ──
                is_dup = False
                q_embedding = self.embeddings.embed_query(q_text)
                for known_emb in self.recent_topics_embeddings:
                    if self._cosine_similarity(q_embedding, known_emb) > 0.92:
                        is_dup = True
                        break
                
                if is_dup or q_text.lower() in seen_questions:
                    print(f"  [dedup-cross] Ignorada (Semântica): {q_text[:80]}")
                    continue
                
                seen_questions.add(q_text.lower())
                
                # Adiciona ao histórico de tópicos (pergunta completa para melhor comparação fuzzy)
                self.recent_topics.append(q_text)
                self.recent_topics_embeddings.append(q_embedding)

                qa["chunk_index"] = i
                if hasattr(chunks[i], "metadata"):
                    qa["source_file"] = chunks[i].metadata.get("source", "unknown")
                all_qa.append(qa)

        self._save_topics()  # Persiste no disco ao fim do lote
        print(f"OK: {len(all_qa)} QAs únicas geradas.")
        return all_qa

    # ── Julgamento g-Eval (Claude Sonnet 4.5) ────────────────────────────────
    def judge(self, qa: Dict, context: str) -> Dict:
        """
        Avalia uma QA com Claude Sonnet 4.5.
        Devolve o dict original enriquecido com o campo "judge".
        """
        prompt = QA_JUDGE_GEVAL_PROMPT.format(
            context=context[:6000],
            persona=qa.get("persona", "Médica"),
            question=qa.get("question", ""),
            answer=qa.get("answer", ""),
            threshold=self.judge_threshold,
        )
        try:
            response = chat_judge(
                prompt,
                temperature=0,            # determinístico para reprodutibilidade
                system_prompt=JUDGE_SYSTEM_PROMPT,
            )

            evaluation = self._parse_response(response)
            if isinstance(evaluation, list):
                evaluation = evaluation[0] if evaluation else {}
            # Remove campo 'persona' vazio do output do juiz (F-16)
            evaluation.pop("persona", None)
            return {**qa, "judge": evaluation}
        except Exception as e:
            print(f"[QAGenerator] Erro no julgamento: {e}")
            return {**qa, "judge": {"error": str(e)}}

    def judge_batch(self, qas: List[Dict], context: str) -> List[Dict]:
        """
        Otimização: Avalia um lote de QAs para o mesmo contexto em uma única chamada.
        """
        if not qas: return []
        
        # Constrói o bloco de QAs para o prompt
        qas_block = ""
        for i, qa in enumerate(qas):
            qas_block += f"QA #{i+1}\nPersona: {qa.get('persona')}\nPergunta: {qa.get('question')}\nResposta: {qa.get('answer')}\n---\n"
            
        prompt = QA_JUDGE_BATCH_GEVAL_PROMPT.format(
            context=context[:6000],
            qas_block=qas_block,
            threshold=self.judge_threshold
        )
        
        try:
            response = chat_judge(
                prompt,
                temperature=0,
                system_prompt=JUDGE_SYSTEM_PROMPT,
            )
            evaluations = self._parse_response(response)
            
            # Garante que temos uma avaliação para cada QA (mesmo que com erro)
            results = []
            for i, qa in enumerate(qas):
                # Tenta casar a avaliação pela ordem
                eval_data = evaluations[i] if i < len(evaluations) else {"error": "Sem feedback para esta QA"}
                results.append({**qa, "judge": eval_data})
            return results
            
        except Exception as e:
            print(f"[QAGenerator] Erro no batch judge: {e}")
            # Fallback para julgamento individual se o lote falhar
            return [self.judge(qa, context) for qa in qas]

    # ── Classificação de erro ─────────────────────────────────────────────────
    def classify_error(self, reason: str) -> str:
        """
        Classifica o feedback do juiz em três categorias:
          HARD_FAILURE  → alucinação grave ou tema completamente ausente
                          → descartar sem tentar reparar
          REPAIRABLE    → omissão, dado parcial, ressalva não incluída,
                          voz errada ou lista incompleta
                          → enviar ao reparo com instruções específicas
          OTHER         → falha ambígua → tenta reparar por precaução
        """
        r = reason.lower()

        hard_signals = [
            "alucinação grave", "erro médico sério",
            "não trata de", "completamente ausente",
            "conhecimento externo", "não está presente no texto",
            "não contém essa informação", "extrapolação",
            "não fundamentado no contexto",
        ]
        if any(w in r for w in hard_signals):
            return "HARD_FAILURE"

        repairable_signals = [
            # omissões de conteúdo existente no texto
            "incompleta", "omissão", "faltou", "omitida", "não mencionou",
            "não captura completamente", "não foi mencionada",
            # dados parciais / listas referenciadas
            "exaustividade", "lista", "quadro", "todos os itens",
            "porcentagem", "dados parciais", "detalha",
            # ressalvas do texto não incluídas
            "ressalva", "potencialmente fatal", "conduta específica",
            "formas atípicas", "formas graves",
            # voz / vocabulário
            "voz", "vocabulário", "fidelidade", "inferência",
            # score entre 40-80% (indica problema corrigível)
            "parcialmente correta", "parcialmente",
        ]
        if any(w in r for w in repairable_signals):
            return "REPAIRABLE"

        return "OTHER"

    # ── Reparo (GPT-4o-mini guiado pelo feedback do juiz Claude) ─────────────
    def repair(self, qa: Dict, context: str, judge_reason: str) -> Dict:
        """
        Repara a resposta entregando o feedback do Claude como instrução
        explícita ao GPT-4o-mini.
        Para listas incompletas, extrai itens faltantes do feedback do juiz
        como checklist explícito antes do reparo.
        """
        # Detecta se é falha de lista incompleta para usar checklist
        is_list_failure = any(w in judge_reason.lower() for w in [
            "omitiu", "não mencionou", "faltou mencionar", "lista", "itens",
            "omissão de itens", "sunflower", "basil", "incompleta"
        ])

        if is_list_failure:
            # Extrai itens faltantes do feedback e passa como bloco destacado
            missing_block = (
                f"\n⚠️ ITENS IDENTIFICADOS COMO FALTANTES PELO JUIZ:\n"
                f"{judge_reason}\n\n"
                f"Verifique CADA item acima no contexto e inclua na resposta "
                f"TODOS que estiverem presentes no texto."
            )
        else:
            missing_block = ""

        prompt = QA_REPAIR_PROMPT.format(
            context=context[:6000],
            persona=qa.get("persona", "Medica"),
            question=qa["question"],
            answer=qa["answer"],
            reason=judge_reason + missing_block,
        )
        try:
            response = chat_openai(
                prompt,
                model=self.generator_model,
                temperature=0,
                system_prompt=GENERATOR_SYSTEM_PROMPT,
            )
            repaired_data = self._parse_response(response)
            if isinstance(repaired_data, list):
                repaired_data = repaired_data[0] if repaired_data else {}

            new_qa             = qa.copy()
            new_qa["answer"]   = repaired_data.get("answer", qa["answer"])
            new_qa["repaired"] = True
            return new_qa
        except Exception as e:
            print(f"[QAGenerator] Erro no reparo: {e}")
            return qa

    # ── Pipeline end-to-end ───────────────────────────────────────────────────
    def run_pipeline(
        self,
        chunks: List,
        questions_per_chunk: int = 1,
        limit: int = 50,
        exhaustive: bool = False,
    ) -> Dict[str, List[Dict]]:
        """
        Pipeline completo:
          1. Gera QAs em lote          (GPT-4o-mini)
          2. Julga cada QA             (Claude Sonnet 4.5 — g-Eval 4D)
             → contexto = chunk_content completo (não apenas source_chunk)
          3. REPAIRABLE → repara       (GPT-4o-mini) → re-julga (Claude)
          4. HARD_FAILURE → descarta

        Retorna:
          {
            "approved":  [...],   # aprovadas diretamente pelo juiz
            "repaired":  [...],   # reparadas e aprovadas na re-avaliação
            "discarded": [...],   # descartadas (falha grave ou reparo insuficiente)
          }
        """
        raw_qas = self.generate_batch(chunks, questions_per_chunk, limit, exhaustive)
        print(f"\nJudging {len(raw_qas)} QAs with {self.judge_model} (Batch Mode Enabled)...")

        # Agrupa QAs por chunk para otimizar contexto
        from collections import defaultdict
        qas_by_chunk = defaultdict(list)
        for qa in raw_qas:
            qas_by_chunk[qa.get("chunk_index", -1)].append(qa)

        chunk_map: Dict[int, str] = {
            i: chunks[i].page_content for i in range(len(chunks))
        }

        approved, repaired, discarded = [], [], []

        for chunk_idx, qas_in_chunk in qas_by_chunk.items():
            full_context = chunk_map.get(chunk_idx, qas_in_chunk[0].get("source_chunk", ""))
            
            # ── Julgamento em Lote (Redução de custo: 1 contexto para N QAs) ──
            judged_qas = self.judge_batch(qas_in_chunk, full_context)

            for judged in judged_qas:
                eval_result = judged.get("judge", {})
                is_approved = eval_result.get("aprovado", False)
                
                # Check actual score mathematically to avoid LLM hallucination
                score_medio = eval_result.get("score_medio", 0)
                if score_medio < self.judge_threshold:
                    is_approved = False
                    
                reason      = eval_result.get("reason", "")

                if is_approved:
                    approved.append(judged)
                else:
                    error_class = self.classify_error(reason)
                    if error_class == "HARD_FAILURE":
                        discarded.append(judged)
                        print(f"  [FAIL] DESCARTADA  [{judged.get('persona')}] {judged.get('question', '')[:60]}")
                    else:
                        # Reparo ainda é individual para maior precisão
                        fixed     = self.repair(judged, full_context, reason)
                        re_judged = self.judge(fixed, full_context)

                        if re_judged.get("judge", {}).get("aprovado", False):
                            repaired.append(re_judged)
                            print(f"  [OK] REPARADA    [{judged.get('persona')}] {judged.get('question', '')[:60]}")
                        else:
                            discarded.append(re_judged)
                            print(f"  [FAIL] DESCARTADA após reparo [{judged.get('persona')}] {judged.get('question', '')[:60]}")

        total_ok = len(approved) + len(repaired)
        print(
            f"\nResultado: {total_ok} aprovadas "
            f"({len(approved)} diretas + {len(repaired)} reparadas) | "
            f"{len(discarded)} descartadas"
        )
        return {"approved": approved, "repaired": repaired, "discarded": discarded}

    # ── Helpers internos ──────────────────────────────────────────────────────
    def _parse_response(self, response: str) -> List[Dict]:
        if not response:
            return []
        try:
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            data = json.loads(text.strip())
            items = data if isinstance(data, list) else [data]
            # Normaliza variações de persona (ex: "Cuidadores" → "Cuidador")
            _persona_map = {
                "cuidadores": "Cuidador", "cuidador": "Cuidador",
                "paciente": "Paciente", "pacientes": "Paciente",
                "medica": "Medica", "médica": "Medica",
                "médico": "Medica", "medico": "Medica",
            }
            for item in items:
                raw = item.get("persona", "")
                item["persona"] = _persona_map.get(raw.lower(), raw)
            return items
        except Exception as e:
            print(f"[QAGenerator] Parse error: {e}")
            return []

    def _validate_source_chunks(self, qas: List[Dict], context: str) -> List[Dict]:
        """
        Descarta QAs cujo source_chunk está vazio, genérico ou não aparece
        de forma reconhecível no texto original.
        Isso é a última barreira contra alucinação antes do juiz.
        """
        valid = []
        context_lower = context.lower()

        for qa in qas:
            sc = qa.get("source_chunk", "").strip()

            # Rejeita source_chunk vazio ou placeholder
            if not sc or sc in ("...", "—", "-", ""):
                print(f"  [source-check] QA sem source_chunk descartada: {qa.get('question', '')[:60]}")
                continue

            # Verifica se ao menos 60% das palavras do source_chunk aparecem no contexto
            words_sc      = set(sc.lower().split())
            words_context = set(context_lower.split())
            if not words_sc:
                continue
            overlap = len(words_sc & words_context) / len(words_sc)
            if overlap < 0.60:
                print(
                    f"  [source-check] source_chunk sem âncora no texto "
                    f"(overlap={overlap:.0%}) — descartada: {qa.get('question', '')[:60]}"
                )
                continue

            valid.append(qa)

        return valid

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calcula a similaridade de cosseno entre dois vetores."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = sum(a * a for a in v1) ** 0.5
        norm_v2 = sum(b * b for b in v2) ** 0.5
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def _deduplicate_intra(self, qas: List[Dict]) -> List[Dict]:
        """Remove duplicatas semânticas dentro do mesmo batch de chunk."""
        result = []
        result_embeddings = []
        for qa in qas:
            is_dup = False
            q_text = qa.get("question", "").strip()
            q_embedding = self.embeddings.embed_query(q_text)
            for r_emb in result_embeddings:
                if self._cosine_similarity(q_embedding, r_emb) > 0.90:
                    is_dup = True
                    break
            if not is_dup:
                result.append(qa)
                result_embeddings.append(q_embedding)
            else:
                print(f"  [dedup-intra] Pergunta redundante removida: {q_text[:50]}...")
        return result


# ═════════════════════════════════════════════════════════════════════════════
#  INSTÂNCIA GLOBAL + WRAPPERS — compatibilidade total com app.py existente
# ═════════════════════════════════════════════════════════════════════════════
qa_generator = QAGenerator()


def generate_qa(chunks: List, num_questions: int = 3) -> List[Dict]:
    return qa_generator.generate(chunks, num_questions=num_questions)


def generate_qa_batch(
    chunks: List,
    questions_per_chunk: int = 1,
    limit: int = 20,
    exhaustive: bool = False,
) -> List[Dict]:
    return qa_generator.generate_batch(
        chunks, questions_per_chunk, limit=limit, exhaustive=exhaustive
    )


def judge_qa(qa: Dict, context: str) -> Dict:
    """Wrapper isolado para o juiz Claude — pode ser importado pelo seu judge.py."""
    return qa_generator.judge(qa, context)


def repair_qa(qa: Dict, context: str, reason: str) -> Dict:
    return qa_generator.repair(qa, context, reason)


def classify_qa_error(reason: str) -> str:
    return qa_generator.classify_error(reason)


def run_full_pipeline(
    chunks: List,
    questions_per_chunk: int = 1,
    limit: int = 50,
    exhaustive: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Atalho para o pipeline completo: gerar → julgar → reparar/descartar.
    Retorna {"approved": [...], "repaired": [...], "discarded": [...]}.
    """
    return qa_generator.run_pipeline(
        chunks, questions_per_chunk, limit=limit, exhaustive=exhaustive
    )