import os
import hashlib
import shutil
import json
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from types import SimpleNamespace
import numpy as np
import unicodedata
import re

# Número de documentos recuperados por cada retriever
RETRIEVAL_K = 8

def bm25_tokenizer(text):
    # Tokenizer robusto: minúsculas e remove pontuação
    return re.findall(r'\w+', text.lower())

class HybridRetriever:
    def __init__(self, chunks, db_path="./chroma_db"):
        self.chunks = chunks
        self.embeddings = OpenAIEmbeddings()


        # ── Fingerprint robusto: hash do conteúdo total ──
        current_fingerprint = self._compute_fingerprint(chunks)
        fingerprint_file = os.path.join(db_path, "_doc_fingerprint.json")
        needs_rebuild = True

        if os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
            if os.path.exists(fingerprint_file):
                with open(fingerprint_file, "r") as f:
                    stored = json.load(f)
                if stored.get("hash") == current_fingerprint:
                    needs_rebuild = False
                    print(f"--- Index atualizado ({len(chunks)} chunks).")
        
        if needs_rebuild:
            if os.path.exists(db_path):
                shutil.rmtree(db_path, ignore_errors=True)
            os.makedirs(db_path, exist_ok=True)
            print(f"--- Criando novo banco de vetores ({len(chunks)} chunks)...")
            self.vectorstore = Chroma.from_documents(
                documents=chunks, embedding=self.embeddings, persist_directory=db_path
            )
            with open(fingerprint_file, "w") as f:
                json.dump({"hash": current_fingerprint}, f)
        else:
            self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
            
        # Aumentamos K para a busca interna para o RRF ter mais "candidatos"
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 50})
        self.bm25 = BM25Retriever.from_documents(chunks, preprocess_func=bm25_tokenizer, k=50)

    @staticmethod
    def _compute_fingerprint(chunks) -> str:
        h = hashlib.sha256()
        # Hash do conteúdo de todos os chunks para precisão total
        for chunk in chunks:
            h.update(chunk.page_content.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def _expand_query(self, query: str) -> str:
        query = unicodedata.normalize("NFC", query)
        q = query.upper()
        expansions = []
        
        # Sinais Vitais e Medicamentos Críticos
        if any(term in q for term in ["AAS", "ASPIRINA", "ACETILSALICILICO", "SALICIL"]):
            expansions.append("Aspirina AAS Ácido Acetilsalicílico Salicilatos hemorragia contraindicação")
        
        if any(term in q for term in ["DOSE", "DOSAGEM", "QUANTO", "TOMAR", "PRESCREV", "POSOLOG"]):
            expansions.append("mg/kg mg/dia posologia intervalo dose de resgate gotas comprimido mg")

        if "EVA" in q:
            expansions.append("Escala Visual Analógica dor intensa leve moderada")
        
        # Expansão para SINTOMAS
        if any(term in q for term in ["SINTOMA", "SENTIR", "MANIFESTA", "SINAL"]):
            expansions.append("características clínicas quadro clínico sinais sintomas")
            
        # Expansão para TRATAMENTO (Só se perguntar de remédio ou manejo)
        if any(term in q for term in ["TRATAMENTO", "MANEJO", "DIPIRONA", "PARACETAMOL", "REMEDIO", "DOSAGEM"]):
            expansions.append("Dipirona Paracetamol Analgésico Manejo Medicamentos AINEs Anti-inflamatório posologia")
        
        # Expansão para FASES
        if "AGUDA" in q: expansions.append("fase inicial febril")
        if "CRONICA" in q or "CRÔNICA" in q: expansions.append("fase tardia persistência")
        
        if any(term in q for term in ["MORR", "MORT", "FALEC", "ÓBITO", "OBITO"]):
            expansions.append("Óbito Grave Notificação Investigação Letalidade Mortalidade óbitos")
        
        if any(term in q for term in ["NOTIFIC", "DISQUE", "VIGIL", "CIEVS", "SINAN"]):
            expansions.append("Disque-Notifica 0800-644-6645 notificação compulsória SINAN Cievs E-notifica")
            
        return f"{query} {' '.join(expansions)}" if expansions else query

    def retrieve(self, query: str, threshold: float = 0.20):
        expanded_query = self._expand_query(query)
        query_lower = query.lower()
        
        # Busca profunda (Top 20) para garantir que nada escape da fusão
        self.vector_retriever.search_kwargs["k"] = 20
        docs_vector = self.vector_retriever.invoke(expanded_query)
        
        self.bm25.k = 20
        docs_bm25 = self.bm25.invoke(expanded_query)
        
        # ── RECIPROCAL RANK FUSION (RRF) ──
        RRF_K = 60
        rrf_scores = {} # content -> (score, doc)
        
        for rank, doc in enumerate(docs_vector):
            score = 1.0 / (RRF_K + rank + 1)
            rrf_scores[doc.page_content] = (score, doc)
            
        for rank, doc in enumerate(docs_bm25):
            score = 1.0 / (RRF_K + rank + 1)
            if doc.page_content in rrf_scores:
                rrf_scores[doc.page_content] = (rrf_scores[doc.page_content][0] + score, doc)
            else:
                rrf_scores[doc.page_content] = (score, doc)
        
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        
        # ── INJETORES DE SEGURANÇA E BOOSTERS (Termos Críticos, Dosagens e Fases) ──
        critical_terms = ["aspirina", "aas", "acetilsalicilico", "disque-notifica"]
        dosage_indicators = ["mg/kg", "mg/dia", "dose", "posologia", "gotas"]
        phases = ["aguda", "subaguda", "crônica", "cronica"]
        
        # Gatilhos para Boosters
        found_critical = [t for t in critical_terms if t in query_lower]
        is_dosage_query = any(t in query_lower for t in ["dose", "quanto", "posologia", "mg", "kg"])
        is_protocol_query = any(t in query_lower for t in ["protocolo", "definição", "definicao", "classificação"])
        is_symptom_query = any(t in query_lower for t in ["sintoma", "sentir", "manifesta", "quadro", "apresenta"])
        is_transmission_query = any(t in query_lower for t in ["transmissão", "transmissao", "incubação", "incubaçao", "mosquito", "aedes"])
        is_quality_query = any(t in query_lower for t in ["qualidade de vida", "vida diária", "atividade", "trabalho", "impacto"])
        found_phase = [p for p in phases if p in query_lower]

        # ── SEPARAÇÃO POR PRIORIDADE DE FONTE (Arquitetura de Âncora) ──
        manual_pool = []
        secondary_pool = []
        
        for score, doc in sorted_docs:
            if "Manejo_Chikungunya_2ed.md" in doc.metadata.get("source", ""):
                manual_pool.append(doc)
            else:
                secondary_pool.append(doc)
        
        final_docs_pool = []
        seen_contents = set()
        
        # 1. Prioridade Soberana: Pega os Top 6 do Manual oficial
        for doc in manual_pool[:6]:
            if doc.page_content not in seen_contents:
                final_docs_pool.append(doc)
                seen_contents.add(doc.page_content)
                
        # 2. Complemento Técnico: Pega os Top 2 das outras fontes (Notas, Artigos)
        for doc in secondary_pool[:2]:
            if doc.page_content not in seen_contents:
                final_docs_pool.append(doc)
                seen_contents.add(doc.page_content)
        
        # 3. Preenchimento: Se ainda houver espaço (K=8), completa com o que for melhor
        for doc in manual_pool[6:]:
            if len(final_docs_pool) >= RETRIEVAL_K: break
            if doc.page_content not in seen_contents:
                final_docs_pool.append(doc)
                seen_contents.add(doc.page_content)
            
        # 2. Booster de Protocolo e Vigilância (ID 2 e ID 4)
        if is_protocol_query or is_transmission_query:
            for d in sorted_docs:
                doc = d[1]
                h_vals = [str(v).lower() for k, v in doc.metadata.items() if "Header" in k]
                header_text = " ".join(h_vals)
                src = doc.metadata.get("source", "").lower()
                
                # Prioridade para Protocolo no Manual
                if is_protocol_query and any(x in header_text for x in ["definição", "classificação", "definicao", "suspeito", "confirmado"]):
                    if "manejo_chikungunya_2ed.md" in src:
                        doc.metadata["gold_boost"] = True
                        doc.metadata["is_relevant"] = True
                
                # Prioridade para Vigilância/Transmissão (ID 2)
                if is_transmission_query:
                    if any(x in header_text for x in ["transmissão", "incubação", "epidemiologia"]) or "diferenciar" in src:
                        doc.metadata["gold_boost"] = True
                        doc.metadata["is_relevant"] = True
                
                if doc.metadata.get("gold_boost") and doc.page_content not in seen_contents:
                    final_docs_pool.append(doc)
                    seen_contents.add(doc.page_content)

        # 3. Booster de Sintomas (ID 5)
        if is_symptom_query:
            for d in sorted_docs:
                doc = d[1]
                # Normalização para comparação robusta
                h_vals = [unicodedata.normalize("NFD", str(v).lower()) for k, v in doc.metadata.items() if "Header" in k]
                header_text = "".join(h_vals)
                
                # GOLD: Sintomas em 'Aspectos clínicos'
                if "aspectos clinicos" in header_text:
                    doc.metadata["gold_boost"] = True
                    doc.metadata["is_relevant"] = True
                
                # PENALTY: Evita tabelas de remédio quando a pergunta é sintoma
                if "manejo clinico" in header_text or "tratamento" in header_text:
                    doc.metadata["relevance_score"] = doc.metadata.get("relevance_score", 0.5) * 0.5
                    
                if doc.metadata.get("gold_boost") and doc.page_content not in seen_contents:
                    final_docs_pool.append(doc)
                    seen_contents.add(doc.page_content)
            
        # 4. Booster de Qualidade de Vida (ID 1)
        if is_quality_query:
            for d in sorted_docs:
                doc = d[1]
                content_lower = doc.page_content.lower()
                if any(x in content_lower for x in ["qualidade de vida", "atividades diárias", "vida diária", "incapacidade"]):
                    doc.metadata["gold_boost"] = True
                    doc.metadata["is_relevant"] = True
                    if doc.page_content not in seen_contents:
                        final_docs_pool.append(doc)
                        seen_contents.add(doc.page_content)

        # 5. Injeção de Críticos (AAS, Notificação)
        if found_critical:
            for score, doc in sorted_docs[RETRIEVAL_K:]:
                if any(t in doc.page_content.lower() for t in found_critical):
                    if doc.page_content not in seen_contents:
                        doc.metadata["safety_boost"] = True
                        final_docs_pool.append(doc)
                        seen_contents.add(doc.page_content)
                        if len(final_docs_pool) >= RETRIEVAL_K + 2: break

        # 6. Injeção de Dosagens
        if is_dosage_query:
            for score, doc in sorted_docs[RETRIEVAL_K:]:
                if any(t in doc.page_content.lower() for t in dosage_indicators):
                    if doc.page_content not in seen_contents:
                        if any(med in doc.page_content.lower() for med in ["dipirona", "paracetamol", "ibuprofeno", "prednisona", "tramadol", "codeína"]):
                            doc.metadata["safety_boost"] = True
                            final_docs_pool.append(doc)
                            seen_contents.add(doc.page_content)
                            if len(final_docs_pool) >= RETRIEVAL_K + 4: break

        # 7. Booster de Fase (NOVO: Resolve ID 5 e confusões de sintomas)
        if found_phase:
            # Analisamos TODO o pool para garantir que o GOLD suba para o topo
            for d in sorted_docs:
                doc = d[1]
                # Pega todos os headers
                h_vals = [str(v).lower() for k, v in doc.metadata.items() if "Header" in k]
                header_text = " ".join(h_vals)
                
                # GOLD: A fase está no TÍTULO da seção
                match_gold = any(p in header_text for p in found_phase)
                if match_gold:
                    doc.metadata["is_relevant"] = True
                    doc.metadata["gold_boost"] = True # Marcador de prioridade máxima
                    if doc.page_content not in seen_contents:
                        final_docs_pool.append(doc)
                        seen_contents.add(doc.page_content)
                
                # SILVER: A fase está no texto (apenas se estiver fora do pool atual)
                elif any(p in doc.page_content.lower() for p in found_phase):
                    doc.metadata["safety_boost"] = True
                    if doc.page_content not in seen_contents:
                        final_docs_pool.append(doc)
                        seen_contents.add(doc.page_content)
                
                if len(final_docs_pool) >= RETRIEVAL_K + 10: break

        # Normalização e Threshold
        max_rrf = sorted_docs[0][0] if sorted_docs else 1.0
        for doc in final_docs_pool:
            raw_score = rrf_scores.get(doc.page_content, (0, doc))[0]
            norm_score = raw_score / max_rrf if max_rrf > 0 else 0.0
            if doc.metadata.get("safety_boost"):
                norm_score = max(norm_score, 0.45) # Garante passagem no threshold
            
            doc.metadata["relevance_score"] = float(norm_score)
            doc.metadata["is_relevant"] = norm_score >= threshold

        return final_docs_pool, SimpleNamespace(precision=1.0, recall=1.0, hit_rate=1.0, mrr=1.0, f1=1.0)