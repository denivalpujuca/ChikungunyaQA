import os
import unicodedata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def load_documents(data_dir: str) -> list[Document]:
    """
    Carrega apenas arquivos .md da pasta `data_dir`.
    Subpastas (ex: docs/en/) são ignoradas — use-as para isolar documentos
    que não devem compor o dataset (ex: artigos em inglês).
    """
    docs = []

    for file in sorted(os.listdir(data_dir)):
        # Ignora subpastas
        path = os.path.join(data_dir, file)
        if os.path.isdir(path):
            continue

        if not file.endswith(".md"):
            print(f"[ingestão] Ignorado (não é .md): {file}")
            continue

        print(f"[ingestão] Lendo: {file}")
        content = ""
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content:
            content = unicodedata.normalize("NFC", content)
            docs.append(Document(page_content=content, metadata={"source": path}))
        else:
            print(f"[ingestão] AVISO: não foi possível ler {file}")

    print(f"[ingestão] {len(docs)} arquivo(s) carregado(s) de '{data_dir}'")
    return docs



#  CLASSIFICAÇÃO MELHORADA
def classify_section(text: str):
    t = text.lower()

    # prioridade alta: fase
    if any(k in t for k in [
        "fase aguda",
        "fase subaguda",
        "fase crônica",
        "fase cronica"
    ]):
        return "fase"

    # tratamento
    if any(k in t for k in [
        "tratamento",
        "dose",
        "dosagem",
        "dipirona",
        "paracetamol",
        "analgésico",
        "anti-inflamatório",
        "medicamento",
        "aspirina",
        "corticoide",
        "corticosteroide",
        "metotrexato",
        "hidroxicloroquina",
        "prednisona",
        "ácido acetilsalicílico"
    ]):
        return "tratamento"

    # notificação e vigilância epidemiológica
    if any(k in t for k in [
        "notificação",
        "notificacao",
        "vigilância",
        "vigilancia",
        "sinan",
        "disque-notifica",
        "cievs",
        "e-notifica",
        "compulsória",
        "compulsoria"
    ]):
        return "notificacao"

    # gestante, neonatal, periparto
    if any(k in t for k in [
        "gestante",
        "grávida",
        "gravida",
        "gravidez",
        "neonatal",
        "recém-nascido",
        "puérpera",
        "puerpera",
        "lactante",
        "periparto",
        "transmissão vertical"
    ]):
        return "gestante"

    # reabilitação e fisioterapia
    if any(k in t for k in [
        "fisioterapia",
        "reabilitação",
        "reabilitacao",
        "exercício",
        "exercicio",
        "alongamento",
        "fortalecimento muscular"
    ]):
        return "reabilitacao"

    # sintomas
    if any(k in t for k in [
        "sintoma",
        "febre",
        "dor",
        "artralgia",
        "mialgia",
        "fadiga"
    ]):
        return "sintomas"

    # diagnóstico
    if any(k in t for k in [
        "diagnóstico",
        "exame",
        "sorologia",
        "rt-pcr",
        "laboratorial"
    ]):
        return "diagnostico"

    return "geral"



def split_documents(docs):
    # Quebra Semântica por Tópicos do Manual
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Fallback Recursivo (Caso uma seção do markdown seja gigantesca)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,       # Aumentado para não cortar listas clínicas
        chunk_overlap=800,     # Aumentado para manter contexto entre seções
        separators=["\n\n", "\n", ".", " "]  
    )

    chunks = []

    for doc in docs:
        source_file = doc.metadata.get("source", "unknown")
        
        # Se for markdown, aplicamos a segmentação inteligente primeiro
        if source_file.endswith(".md"):
            md_splits = markdown_splitter.split_text(doc.page_content)
            final_splits = recursive_splitter.split_documents(md_splits)
            
            for split in final_splits:
                # Preserva a hierarquia do manual médico
                header_name = split.metadata.get("Header 2", split.metadata.get("Header 1", "Geral"))
                if "Header 3" in split.metadata:
                    header_name += f" - {split.metadata['Header 3']}"
                
                internal_section = classify_section(split.page_content)
                
                split.metadata["source"] = source_file
                split.metadata["topic"] = "chikungunya"
                split.metadata["section"] = f"{header_name} | {internal_section}"
                
                chunks.append(split)
        else:
            # Fallback: documento .md sem segmentação por headers (ex: arquivo muito curto)
            parts = recursive_splitter.split_text(doc.page_content)
            for part in parts:
                section = classify_section(part)
                chunks.append(
                    Document(
                        page_content=part,
                        metadata={
                            "source": source_file,
                            "topic": "chikungunya",
                            "section": section
                        }
                    )
                )

    print(f"Dividido em {len(chunks)} chunks com metadata semântica estruturada.")
    return chunks