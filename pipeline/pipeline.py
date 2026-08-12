from typing import List
from types import SimpleNamespace
import os
import shutil
import json
import time
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .retrieval import HybridRetriever
from .evaluator import ResponseEvaluator
from .llm import chat_openai as chat
from .prompts import RAG_SYSTEM_PROMPT  # importado aqui só para validação de sintaxe

class RAGPipeline:
    def __init__(self, chunks: List):
        # A inicialização do banco de dados e embeddings agora é centralizada no HybridRetriever
        # para evitar conflitos de escrita e redundância.
        self.retriever = HybridRetriever(chunks)
        self.evaluator = ResponseEvaluator()

    def run(self, query: str, persona: str = "Médica", stream: bool = False):
        # Reload forçado para garantir prompt atualizado mesmo com @st.cache_resource
        import importlib
        import pipeline.prompts as _pm
        importlib.reload(_pm)
        RAG_PROMPT = _pm.RAG_PROMPT
        PERSONA_INSTRUCTIONS = _pm.PERSONA_INSTRUCTIONS
        RAG_SYSTEM_PROMPT = _pm.RAG_SYSTEM_PROMPT
        print(f"--- DEBUG: Usando RAG_PROMPT (Tam: {len(RAG_PROMPT)})")

        docs, retrieval_metrics = self.retriever.retrieve(query)

        # GUARDRAILS
        max_score = max([d.metadata.get("relevance_score", 0.0) for d in docs]) if docs else 0.0
        has_safety_docs = any(d.metadata.get("safety_boost") for d in docs)
        critical_terms = ["aspirina", "aas", "acetilsalicil", "salicil", "disque-notifica", "dipirona", "paracetamol"]
        query_has_critical = any(t in query.lower() for t in critical_terms)
        is_low_confidence = max_score < 0.25 and not has_safety_docs and not query_has_critical

        if not docs or is_low_confidence:
            answer = "Sinto muito, mas não encontrei informações suficientemente relevantes nos documentos técnicos para responder a essa pergunta com segurança."
            if stream:
                def _gen(): yield answer
                return {"answer": _gen(), "docs": docs, "is_refusal": True, "retrieval_metrics": retrieval_metrics}
            
            return {
                "answer": answer,
                "docs": docs,
                "retrieval_metrics": retrieval_metrics,
                "response_metrics": SimpleNamespace(geval_score=1.0, faithfulness=1.0, geval_reason="Recusa correta.")
            }

        # Termos farmacológicos: sempre incluir docs que contenham medicamentos mencionados na query
        drug_terms = ["dipirona", "paracetamol", "tramadol", "codeína", "ibuprofeno", "prednisona", "corticoide"]
        query_drugs = [t for t in drug_terms if t in query.lower()]
        
        relevant_docs = [
            d for d in docs
            if d.metadata.get("is_relevant", False)
            or d.metadata.get("safety_boost", False)
            or d.metadata.get("gold_boost", False)
            # Força inclusão de docs com medicamentos da query (evita exclusão por score baixo)
            or any(drug in d.page_content.lower() for drug in query_drugs)
        ]
        relevant_docs.sort(key=lambda d: (d.metadata.get("gold_boost", False), d.metadata.get("safety_boost", False), d.metadata.get("relevance_score", 0)), reverse=True)
        relevant_docs = relevant_docs[:8]

        
        context_parts = []
        for d in relevant_docs:
            source = d.metadata.get("source", "Documento")
            h = " > ".join([v for v in [d.metadata.get(f"Header {i}", "") for i in range(1,5)] if v])
            context_parts.append(f"### FONTE: {source} | SEÇÃO: {h or 'Conteúdo Geral'}\n{d.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        p_instr = PERSONA_INSTRUCTIONS.get(persona, "")
        prompt = RAG_PROMPT.format(
            query=query, 
            context=context, 
            persona=persona, 
            persona_instructions=p_instr
        )
        
        if stream:
            response_stream = chat(prompt, temperature=0, stream=True, system_prompt=RAG_SYSTEM_PROMPT)
            def _stream_gen():
                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
            def _evaluate_and_log(alt_answer):
                ev = self.evaluator.evaluate(query, alt_answer, docs)
                log_path = os.path.join(os.getcwd(), "output", "chat_debug.log")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query, "persona": persona, "answer": alt_answer,
                        "geval_score": ev.get("geval_score"), "geval_reason": ev.get("geval_reason"),
                        "context_used": [d.page_content[:200] for d in docs]
                    }, ensure_ascii=False) + "\n")
                return ev

            return {
                "answer": _stream_gen(),
                "docs": docs,
                "retrieval_metrics": retrieval_metrics,
                "evaluate": _evaluate_and_log
            }

        answer = chat(prompt, temperature=0, system_prompt=RAG_SYSTEM_PROMPT)
        raw_eval = self.evaluator.evaluate(query, answer, docs)
        
        log_path = os.path.join(os.getcwd(), "output", "chat_debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": query, "persona": persona, "answer": answer,
                "geval_score": raw_eval.get("geval_score"), "geval_reason": raw_eval.get("geval_reason"),
                "context_used": [d.page_content[:200] for d in docs]
            }, ensure_ascii=False) + "\n")

        return {
            "answer": answer,
            "docs": docs,
            "retrieval_metrics": retrieval_metrics,
            "response_metrics": SimpleNamespace(
                context_relevance=float(max_score),
                faithfulness=raw_eval.get("faithfulness", 0.0),
                geval_score=raw_eval.get("geval_score", 0.0),
                geval_reason=raw_eval.get("geval_reason", "Não disponível")
            )
        }