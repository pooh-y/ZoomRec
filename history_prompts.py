"""Original observed-history summary prompt and parser."""
import json, re

def prompt_for(event, cards):
    hist = event["history"][-25:]
    lines = [f"{i+1}. {cards[x]['title']} | {', '.join(cards[x]['categories'][-3:])}" for i, x in enumerate(hist)]
    return ("You are a deterministic recommendation user model. Do not predict a hidden target. "
            "Summarize only the observed history below. Return one JSON object with arrays "
            "stable and transient containing at most 8 short lowercase preference tokens. "
            "stable means recurring themes; transient means recent themes. No prose.\n" + "\n".join(lines))

def parse_state(raw, event, cards):
    fallback = []
    for item in event["history"]:
        fallback += cards[item]["categories"] + cards[item]["tokens"][:6]
    fallback = list(dict.fromkeys([x.lower() for x in fallback]))[:8]
    try:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        obj = json.loads(match.group(0)) if match else {}
        stable = [str(x).lower().strip() for x in obj.get("stable", []) if str(x).strip()]
        transient = [str(x).lower().strip() for x in obj.get("transient", []) if str(x).strip()]
        return {"stable": stable[:8] or fallback[:4], "transient": transient[:8] or fallback[-4:], "raw": raw}
    except Exception:
        return {"stable": fallback[:4], "transient": fallback[-4:], "raw": raw}
