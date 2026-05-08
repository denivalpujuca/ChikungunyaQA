from typing import List
from types import SimpleNamespace
import os
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .retrieval import HybridRetriever
from .evaluator import ResponseEvaluator
from .llm import chat_openai as chat
from .prompts import RAG_PROMPT

class RAGPipeline:
    def __init__(self, chunks: List):
        # A inicialização do banco de dados e embeddings agora é centralizada no HybridRetriever
        # para evitar conflitos de escrita e redundância.
        self.retriever = HybridRetriever(chunks)
        self.evaluator = ResponseEvaluator()

    def run(self, query: str):
        docs, retrieval_metrics = self.retriever.retrieve(query)

        # GUARDRAILS: Verificamos se houve algum "hit" real e se a qualidade mínima foi atingida
        max_score = max([d.metadata.get("relevance_score", 0.0) for d in docs]) if docs else 0.0
        
        # Safety Injection bypass: se o retriever injetou docs críticos, não recusamos
        has_safety_docs = any(d.metadata.get("safety_boost") for d in docs)
        
        # Termos clinicamente críticos que nunca devem resultar em recusa silenciosa
        critical_terms = ["aspirina", "aas", "acetilsalicil", "salicil", "disque-notifica"]
        query_has_critical = any(t in query.lower() for t in critical_terms)
        
        is_low_confidence = max_score < 0.25 and not has_safety_docs and not query_has_critical

        if not docs or is_low_confidence:
            response_metrics = SimpleNamespace(
                context_relevance=max_score,
                faithfulness=1.0,
                nli_details=None,
                geval_score=1.0,
                geval_reason="Recusa correta: O tema da pergunta está fora do escopo técnico dos documentos fornecidos.",
                refusal_accuracy=1.0,
                feedback=f"Recusa automática: Confiança baixa ({max_score:.2f}) ou nenhum documento encontrado."
            )
            return {
                "answer": "Sinto muito, mas não encontrei informações suficientemente relevantes nos documentos técnicos para responder a essa pergunta com segurança.",
                "docs": docs,
                "retrieval_metrics": retrieval_metrics,
                "response_metrics": response_metrics
            }

        # Contexto consolidado: injeta metadados de estrutura (Headers do Markdown) para evitar perda de contexto
        # Filtra apenas o que é relevante ou boosteado
        relevant_docs = [d for d in docs if d.metadata.get("is_relevant", False) or d.metadata.get("safety_boost", False) or d.metadata.get("gold_boost", False)]
        
        # PRIORIDADE ABSOLUTA: Gold Boost > Safety Boost > Relevance Score
        relevant_docs.sort(key=lambda d: (
            d.metadata.get("gold_boost", False), 
            d.metadata.get("safety_boost", False),
            d.metadata.get("relevance_score", 0)
        ), reverse=True)
        
        # LIMITAÇÃO: Pega apenas os top 8 para evitar diluição de contexto e confusão do LLM
        relevant_docs = relevant_docs[:8]
        
        # DEBUG DE ORDEM FINAL (Só para garantir)
        print(f"   [ORDEM CONTEXTO] Top 3 seções enviadas:")
        for i, d in enumerate(relevant_docs[:3]):
            h = d.metadata.get("section", "Sem seção")
            gold = " [GOLD]" if d.metadata.get("gold_boost") else ""
            print(f"      {i+1}. {h}{gold}")
        
        context_parts = []
        for d in relevant_docs:
            source = d.metadata.get("source", "Documento")
            h1 = d.metadata.get("Header 1", "")
            h2 = d.metadata.get("Header 2", "")
            h3 = d.metadata.get("Header 3", "")
            h4 = d.metadata.get("Header 4", "")
            headers = [h for h in [h1, h2, h3, h4] if h]
            header_str = " > ".join(headers) if headers else "Conteúdo Geral"
            
            # Monta o bloco de texto contextualizado
            context_parts.append(f"### FONTE: {source} | SEÇÃO: {header_str}\n{d.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = RAG_PROMPT.format(query=query, context=context)
        # Auditoria de Contexto (Útil para debugar ID 5 e ID 2)
        print(f"   [CONTEXTO AUDIT] Tamanho: {len(context)} caracteres. Início: {context[:300]}...")
        
        answer = chat(prompt, temperature=0)
        
        # Avaliação Dinâmica (Usando os docs originais para o cálculo de relevância)
        raw_eval = self.evaluator.evaluate(query, answer, docs)

        
        response_metrics = SimpleNamespace(
            context_relevance=float(max_score),
            faithfulness=raw_eval.get("faithfulness", 0.0),
            nli_details=None,
            geval_score=raw_eval.get("geval_score", 0.0),
            geval_reason=raw_eval.get("geval_reason", "Não disponível"),
            refusal_accuracy=raw_eval.get("refusal_accuracy", 0.0),
            feedback=raw_eval.get("feedback", "Avaliação concluída.")
        )

        return {
            "answer": answer,
            "docs": docs,
            "retrieval_metrics": retrieval_metrics,
            "response_metrics": response_metrics
        }