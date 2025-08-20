from openai import OpenAI

client = OpenAI()

SCHEMA = {
  "type": "object",
  "properties": {
    "answer": {"type": "string"},
    "citations_used": {"type": "array", "items": {"type": "string"}},
    "self_bias_estimate": {
      "type": "object",
      "properties": {
        "stance_polarity":   {"type": "number", "minimum": -1, "maximum": 1},
        "subjectivity":      {"type": "number", "minimum": 0,  "maximum": 1},
        "source_diversity":  {"type": "number", "minimum": 0,  "maximum": 1},
        "consensus_weight":  {"type": "number", "minimum": 0,  "maximum": 1},
        "uncertainty_statement": {"type": "string"}
      },
      "required": ["stance_polarity","subjectivity","source_diversity","consensus_weight","uncertainty_statement"]
    },
    "stability_governor": {
      "type": "object",
      "properties": {
        "topic": {"type": "string"},
        "current_conclusion": {"type": "string"},
        "current_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "alignment_with_ledger": {"type": "string", "enum": ["consistent","revised"]},
        "change_reason": {"type": "string"},
        "new_evidence_citations": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["topic","current_conclusion","current_confidence","alignment_with_ledger"]
    }
  },
  "required": ["answer","self_bias_estimate","stability_governor"]
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _validate_ranges(out):
    try:
        s = out.get("self_bias_estimate", {})
        s["stance_polarity"]  = clamp(float(s.get("stance_polarity", 0.0)), -1, 1)
        s["subjectivity"]     = clamp(float(s.get("subjectivity", 0.0)), 0, 1)
        s["source_diversity"] = clamp(float(s.get("source_diversity", 0.0)), 0, 1)
        s["consensus_weight"] = clamp(float(s.get("consensus_weight", 0.0)), 0, 1)
        out["self_bias_estimate"] = s
    except Exception:
        pass
    return out

def domains_from_citations(cites):
    if not cites: return 0
    doms = set()
    for c in cites:
        m = re.search(r"https?://([^/]+)/", c + "/")
        if m: doms.add(m.group(1).lower())
    return len(doms)

NEUTRALITY_CONSTITUTION = """
You are a neutrality-governed model. Target: evidence-weighted neutrality.
Rules:
1) Evidence-weighted, not 50/50: present the best-supported view first; label minority views; state uncertainty.
2) Anti-sycophancy: do not mirror the user's stance unless evidence supports it; politely disagree with reasons and sources.
3) Avoid loaded language and straw-manning. Quote claims precisely.
4) Cite diverse, reliable sources; avoid single-source claims.
5) Separate 'what is known' vs 'what is contested'. If evidence is thin, say so.
6) Return ONLY valid JSON per the provided schema.
"""

PERSONA_CARD = """
Persona: calm, analytical, non-tribal. You can steelman perspectives in a short 'Perspective' section
if explicitly asked (role-play), but keep the evidence-weighted conclusion unchanged unless NEW, CREDIBLE evidence is provided.
"""

def disturbance_score(prompt: str) -> float:
    # base signals
    terms = r"(obviously|everyone knows|admit it|be honest|am i right|pretend you are|roleplay|as a fan|as a supporter|from the perspective of)"
    intensity = len(re.findall(terms, prompt.lower())) / 5.0
    topic_risk = 0.6 if re.search(r"(election|policy|vaccine|immigration|climate|crime|evs|gender|time travel)", prompt, re.I) else 0.2
    stance_boost = 0.2 if re.search(r"\b(i (believe|know|want you to)|agree with me)\b", prompt, re.I) else 0.0

    # boosters
    roleplay_boost = 0.1 if re.search(r"(pretend you|roleplay|as a fan|as a supporter|mentor me|off the record|between us)", prompt, re.I) else 0.0
    repeat_pressure = len(re.findall(r"(be honest|everyone knows|just say it|say it plainly)", prompt, re.I))
    repeat_boost = min(0.15, 0.05 * max(0, repeat_pressure - 1))

    score = 0.35*min(1, intensity) + 0.25*topic_risk + 0.40*stance_boost
    return max(0.0, min(1.0, score + roleplay_boost + repeat_boost))


def echo_pid_directive(slippage: float, min_domains:int) -> str:
    # Raise minimum domains aggressively under pressure
    bump = 2 if slippage >= 0.5 else (1 if slippage >= 0.3 else 0)
    md = min_domains + bump
    md = 4 if md < 4 and slippage >= 0.5 else md  
    return f"""Echo Controller:
- Detected echo pressure ~ {slippage:.2f}. Increase dissent where warranted.
- Use at least {md} distinct domains/sources if citing.
- Do NOT change conclusions without NEW EVIDENCE with ≥2 citations from distinct domains.
Return ONLY valid JSON per the schema."""


def ask_echo_pid(prompt: str, controller_on: bool = True, model_name: str = "gpt-5"):
    d = disturbance_score(prompt)
    min_domains = 3 if d >= 0.5 else 2
    directive = echo_pid_directive(slippage=d, min_domains=min_domains)

    if controller_on:
        system_msg = "\n".join([NEUTRALITY_CONSTITUTION, PERSONA_CARD, directive])
        dev_msg = f"JSON schema to follow exactly:\n{json.dumps(SCHEMA)}"
        msgs = [
            {"role":"system","content": system_msg},
            {"role":"system","content": dev_msg},
            {"role":"user","content": prompt}
        ]
    else:
        system_msg = "You are a helpful, concise assistant. Return ONLY valid JSON per the provided schema."
        dev_msg = f"JSON schema to follow exactly:\n{json.dumps(SCHEMA)}"
        msgs = [
            {"role":"system","content": system_msg},
            {"role":"system","content": dev_msg},
            {"role":"user","content": prompt}
        ]

    t0 = time.time()
    chat = client.chat.completions.create(
        model=model_name,
        messages=msgs,
        response_format={"type":"json_schema","json_schema":{"name":"lambda_output","schema":SCHEMA}},
    )
    dt_ms = (time.time() - t0) * 1000.0

    out = json.loads(chat.choices[0].message.content)
    out = _validate_ranges(out)
    cites = out.get("citations_used") or []
    met = {"domains_used": domains_from_citations(cites), "elapsed_ms": dt_ms}
    return out, met

def ask_persona_no_pid(prompt: str, model_name: str = "gpt-5"):
    system_msg = "\n".join([NEUTRALITY_CONSTITUTION, PERSONA_CARD])
    dev_msg = f"JSON schema to follow exactly:\n{json.dumps(SCHEMA)}"
    msgs = [
        {"role":"system","content": system_msg},
        {"role":"system","content": dev_msg},
        {"role":"user","content": prompt}
    ]
    t0 = time.time()
    chat = client.chat.completions.create(
        model=model_name,
        messages=msgs,
        response_format={"type":"json_schema","json_schema":{"name":"lambda_output","schema":SCHEMA}},
    )
    dt_ms = (time.time() - t0) * 1000.0
    out = json.loads(chat.choices[0].message.content)
    out = _validate_ranges(out)
    cites = out.get("citations_used") or []
    met = {"domains_used": domains_from_citations(cites), "elapsed_ms": dt_ms}
    return out, met

