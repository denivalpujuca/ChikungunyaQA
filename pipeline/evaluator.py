"""
evaluator.py
────────────
Avaliação de respostas RAG usando:
  • Similaridade de Cosseno (OpenAI) - Relevância de Contexto
  • G-Eval Judge (Claude/GPT-4o via llm.py) - Auditoria Clínica
"""

import json
import time
import numpy as np
from typing import Dict
from langchain_openai import OpenAIEmbeddings

# Importamos o juiz centralizado
from .llm import chat_judge, chat_openai
from .prompts import G_EVAL_JUDGE_PROMPT, JUDGE_SYSTEM_PROMPT

class ResponseEvaluator:
    def __init__(self):
        # Unificando para os mesmos embeddings do retrieval.py
        self.embeddings = OpenAIEmbeddings()

    def _get_embedding(self, text: str):
        return self.embeddings.embed_query(text)

    def _cosine_similarity(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0.0

    def evaluate(self, query: str, answer: str, contexts: list) -> Dict:
        """
        Avaliação Híbrida:
        1. Similaridade Semântica (Cosseno)
        2. Auditoria Clínica (G-Eval)
        """
        start_time = time.time()
        
        # 1. Métrica de Relevância (Similaridade de Cosseno)
        query_vec = self._get_embedding(query)
        full_context = "\n".join([c.page_content for c in contexts])
        context_vec = self._get_embedding(full_context)
        
        context_relevance = float(self._cosine_similarity(query_vec, context_vec))
        
        # 2. G-Eval (Auditoria via LLM Juiz)
        eval_prompt = G_EVAL_JUDGE_PROMPT.format(
            query=query,
            answer=answer,
            context=full_context
        )
        
        try:
            # Usa o juiz centralizado (Claude por padrão, GPT como fallback)
            judge_response = chat_judge(
                eval_prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                temperature=0.0
            )
            
            # Limpeza básica caso o modelo retorne blocos de markdown
            clean_response = judge_response.strip().replace("```json", "").replace("```", "")
            geval_data = json.loads(clean_response)
        except Exception as e:
            print(f"--- [ERRO EVALUATOR] Falha no G-Eval: {e}")
            geval_data = {"score": 1, "faithfulness_score": 1, "reason": f"Erro na auditoria: {str(e)}"}

        latency = time.time() - start_time

        return {
            "context_relevance": context_relevance,
            "geval_score": geval_data.get("score", 0),
            "faithfulness": geval_data.get("faithfulness_score", 0),
            "geval_reason": geval_data.get("reason", "N/A"),
            "latency": latency
        }
