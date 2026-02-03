import json, re
from typing import Dict, Any
from ollama import Client

OLLAMA_HOST = "http://195.24.232.107:50111"
MODEL_NAME = "mistral-small3.2:24b"

client = Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = (
  "You are a concise, encouraging fitness coach.\n"
  "Return compact, actionable, SAFE cues.\n"
  "Output JSON only. No extra text."
)

LANG_HINT = {
  "en": "Respond ONLY in English. Do not use other languages.",
  "tr": "Cevabi SADECE Türkçe yaz. Başka dil kullanma.",
  "de": "Antworte NUR auf Deutsch. Keine andere Sprache verwenden.",
  "es": "Responde SOLO en español. No uses otros idiomas.",
  "zh": "请只用中文回答。不要使用其他语言。"
}

def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else json.dumps({"summary": text.strip(), "advice": []})

def _dedupe(items, limit):
    out, seen = [], set()
    for it in items or []:
        s = str(it).strip()
        k = s.lower()
        if k and k not in seen:
            out.append(s); seen.add(k)
        if len(out) >= limit: break
    return out

def _force_schema(d: Dict[str, Any], lang: str) -> Dict[str, Any]:
    d.setdefault("summary", ""); d["summary"] = str(d["summary"])
    d.setdefault("advice", []); d["advice"] = _dedupe(d["advice"], 5)
    d.setdefault("warnings", []); d["warnings"] = _dedupe(d["warnings"], 5)
    d.setdefault("focus_next_rep", []); d["focus_next_rep"] = _dedupe(d["focus_next_rep"], 3)
    d.setdefault("language", lang); d["language"] = str(d["language"])
    return d

def generate_llm_feedback(movement: str, lang: str, angles: Dict[str, Any], score: float, predicted_label: str) -> Dict[str, Any]:
    hint = LANG_HINT.get(lang, LANG_HINT["en"])
    user_prompt = f"""
{hint}

Hareket: {movement}
Sınıflandırma: {predicted_label}
Skor: {score}

Açılar (derece; null olabilir):
- knee_angle: {angles.get('knee_angle')}
- torso_angle: {angles.get('torso_angle')}
- body_angle: {angles.get('body_angle')}
- elbow_angle: {angles.get('elbow_angle')}

Sadece AŞAĞIDAKİ ŞEMA ile KESİN JSON DÖN:
{{
  "summary": "1-2 cümle genel değerlendirme",
  "advice": ["≤5 kısa uygulanabilir ipucu, her biri en fazla gelen açılara göre yorumu yap ≤15 kelime"],
  "warnings": ["opsiyonel risk/teknik notlar"],
  "focus_next_rep": ["≤3 mikro ipucu ama yaratıcı ol ve her defasında farklı yorum yap(sonraki tekrar)"],
  "language": "{lang}"
}}
Kurallar:
- İpuçlarını açı ve harekete göre özelleştir.
- 'correct' olsa bile 1-2 mikro iyileştirme ver.
- Tıbbi iddia yok. Kısa yaz.
"""

    last_err = None
    for _ in range(2):
        try:
            resp = client.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "num_ctx": 4096
                }
            )
            raw = resp["message"]["content"]
            data = json.loads(_extract_json(raw))
            return _force_schema(data, lang)
        except Exception as e:
            last_err = e

    

    return {
        "summary": f"{'Form iyi' if predicted_label=='correct' else 'Form geliştirilmeli'} ({movement}, skor={score}).",
        "advice": ["Merkezi sık.", "Tempoyu kontrol et.", "Eklemleri hizalı tut."],
        "warnings": ["Diz açısına dikkat et." if angles.get('knee_angle') and angles['knee_angle'] > 160 else ""],
        "focus_next_rep": ["Derinliği tutarlı yap.", "Nefes almayı unutma."],
        "language": lang,
        "debug": {
            "fallback": True,
            "host": OLLAMA_HOST,
            "model": MODEL_NAME,
            "error": str(last_err),
            "input": {
                "movement": movement,
                "angles": angles,
                "score": score,
                "predicted_label": predicted_label
            }
        }
    }

def ping_llm() -> str:
    try:
        r = client.chat(model=MODEL_NAME, messages=[{"role":"user","content":"ping"}], options={"temperature":0})
        return r["message"]["content"].strip()
    except Exception as e:
        return f"LLM unreachable: {e}"
    
