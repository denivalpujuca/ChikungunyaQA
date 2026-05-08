"""
llm.py
──────
Centralização de modelos e motores de inferência.
Inclui retry automático com backoff exponencial para resiliência a erros de API.
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── CONFIGURAÇÃO CENTRAL DE MODELOS ──────────────────────────────────────────
# Mude aqui para alternar entre Claude e GPT como juiz
GENERATOR_MODEL = "gpt-4o-mini"

# Opções para JUDGE_MODEL:
# 1. "claude-sonnet-4-5"        (Ouro — padrão, mesmo preço do 3.5, geração mais recente)
# 2. "claude-3-5-sonnet-latest" (Prata — fallback se 4.5 não estiver disponível)
# 3. "gpt-4o"                   (Bronze — fallback se créditos Anthropic acabarem)
JUDGE_MODEL = "claude-sonnet-4-5"

# ── Retry com Backoff Exponencial ─────────────────────────────────────────────
def _retry(func, max_retries: int = 3, base_delay: float = 5.0):
    """
    Tenta executar func() até max_retries vezes com espera exponencial.
    Erros 429 (rate limit) e 5xx são retentados. Outros erros sobem imediatamente.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(k in err_str for k in [
                "429", "rate limit", "overloaded", "timeout",
                "503", "502", "500", "connection"
            ])
            if not is_retryable or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  [LLM] Tentativa {attempt + 1}/{max_retries} falhou: {e}. "
                  f"Aguardando {delay:.0f}s...")
            time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
#  MOTOR OPENAI (GPT)
# ─────────────────────────────────────────────────────────────────────────────
class LLMEngine:
    _instances: dict = {}

    def __new__(cls, model: str = GENERATOR_MODEL):
        if model not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[model] = instance
        return cls._instances[model]

    def __init__(self, model: str = GENERATOR_MODEL):
        if self._initialized:
            return
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self._initialized = True

    def generate(self, prompt: str, temperature: float = 0.1, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content

        return _retry(_call)


def chat_openai(prompt: str, model: str = GENERATOR_MODEL, temperature: float = 0.1, system_prompt: str | None = None) -> str:
    engine = LLMEngine(model=model)
    return engine.generate(prompt, temperature=temperature, system_prompt=system_prompt)


# ─────────────────────────────────────────────────────────────────────────────
#  MOTOR ANTHROPIC (CLAUDE)
# ─────────────────────────────────────────────────────────────────────────────
def chat_anthropic(prompt: str, model: str = JUDGE_MODEL, temperature: float = 0.0, system_prompt: str | None = None) -> str:
    def _call():
        llm = ChatAnthropic(model=model, temperature=temperature)
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = llm.invoke(messages)
        return response.content

    return _retry(_call)


# ─────────────────────────────────────────────────────────────────────────────
#  WRAPPER INTELIGENTE PARA O JUIZ
# ─────────────────────────────────────────────────────────────────────────────
def chat_judge(prompt: str, temperature: float = 0.0, system_prompt: str | None = None) -> str:
    """
    Usa o modelo definido em JUDGE_MODEL.
    Se for um modelo da Anthropic (Claude), usa chat_anthropic.
    Se for da OpenAI (GPT-4o), usa chat_openai.
    """
    if "claude" in JUDGE_MODEL.lower():
        return chat_anthropic(prompt, model=JUDGE_MODEL, temperature=temperature, system_prompt=system_prompt)
    else:
        return chat_openai(prompt, model=JUDGE_MODEL, temperature=temperature, system_prompt=system_prompt)


# Alias para compatibilidade legada
chat_claude = chat_judge
