"""The system under test: a grounded banking Q&A assistant.

`Assistant.answer(item, ...)` returns {answer, abstained, confidence} — the assistant is supposed to
answer ONLY from the item's context and to abstain when the answer isn't there.

Two configurations are provided so the harness can benchmark champion vs. challenger:
  * "guarded" — a well-instructed assistant (grounds carefully, abstains more),
  * "naive"   — a loosely-instructed one (hallucinates more, over-confident).

Default is a deterministic MOCK (hash-seeded) so the whole harness is reproducible with no API key.
Set env USE_LLM=1 (with ANTHROPIC_API_KEY) to run a real Claude model instead.
"""
import os, hashlib, json

def _r(*parts):
    """Deterministic pseudo-random float in [0,1) from the given parts (no RNG state)."""
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

ABSTAIN_TEXT = "I'm sorry — that information isn't in the policy I have access to."

# per-config thresholds on the decision draw r
CFG = {
    "guarded": dict(ans_correct=0.90, ans_wrong=0.96, unans_abstain=0.85),
    "naive":   dict(ans_correct=0.80, ans_wrong=0.93, unans_abstain=0.45),
}

class Assistant:
    def __init__(self, mode="guarded"):
        assert mode in CFG, mode
        self.mode = mode
        self.use_llm = os.environ.get("USE_LLM") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))

    def answer(self, item, temperature=0.0, run=0, variant="orig"):
        if self.use_llm:
            out = self._llm_answer(item, temperature)
            if out is not None:
                return out
        return self._mock_answer(item, temperature, run, variant)

    # ---------------- deterministic mock ----------------
    def _mock_answer(self, item, temperature, run, variant):
        c = CFG[self.mode]
        seed = [item["id"], self.mode]
        if temperature > 0:
            seed.append(f"run{run}")          # variation across repeated calls
        if variant != "orig":
            seed.append(variant)              # variation across paraphrases
        r = _r(*seed)
        conf = lambda lo, hi: round(lo + (hi - lo) * _r(*seed, "conf"), 3)

        if item["answerable"]:
            if r < c["ans_correct"]:
                return dict(answer=item["reference"], abstained=False, confidence=conf(0.80, 0.95))
            elif r < c["ans_wrong"]:
                return dict(answer=item["distractor"], abstained=False, confidence=conf(0.65, 0.85))
            else:
                return dict(answer=ABSTAIN_TEXT, abstained=True, confidence=conf(0.20, 0.40))
        else:
            if r < c["unans_abstain"]:
                return dict(answer=ABSTAIN_TEXT, abstained=True, confidence=conf(0.20, 0.40))
            else:  # hallucination: confident and wrong
                return dict(answer=item["distractor"], abstained=False, confidence=conf(0.70, 0.90))

    # ---------------- optional real Claude ----------------
    def _llm_answer(self, item, temperature):
        try:
            import anthropic
            strict = ("You are a banking assistant. Answer ONLY using the CONTEXT. "
                      "If the answer is not in the context, you MUST abstain.") if self.mode == "guarded" else \
                     ("You are a helpful banking assistant. Use the context; answer the customer's question.")
            prompt = (f"{strict}\nReturn strict JSON: {{\"answer\": str, \"abstain\": bool, \"confidence\": number 0-1}}.\n\n"
                      f"CONTEXT:\n{item['context']}\n\nQUESTION: {item['question']}")
            m = anthropic.Anthropic().messages.create(
                model="claude-sonnet-5", max_tokens=300, temperature=temperature,
                messages=[{"role": "user", "content": prompt}])
            txt = m.content[0].text
            j = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            return dict(answer=("" if j.get("abstain") else j.get("answer", "")) or ABSTAIN_TEXT,
                        abstained=bool(j.get("abstain")), confidence=float(j.get("confidence", 0.5)))
        except Exception:
            return None
