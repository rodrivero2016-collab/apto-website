"""
Apto Bid Agent - Flask blueprint for the existing Railway app.

Drop this file next to app.py in the Railway repo, put apto_profile.json in the
same folder, then add two lines to app.py:

    from bid_agent import bid_agent_bp
    app.register_blueprint(bid_agent_bp)

Requires: ANTHROPIC_API_KEY in Railway environment variables.
Endpoint: POST /bid-agent

The Apto profile stays server side. It contains the EIN, taxpayer number, and
rate card, none of which belong in a public GitHub Pages repo.
"""

import json
import os
import re
import time

from flask import Blueprint, jsonify, make_response, request

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None

bid_agent_bp = Blueprint("bid_agent", __name__)

# Bump on every change. /bid-agent/health reports it, so you can confirm which
# version Railway is actually running instead of guessing.
CODE_VERSION = "1.2-tool-json"

# Forcing tool use makes the model return its answer as a tool input, which the API
# validates as JSON before we ever see it. That removes the whole class of
# "could not parse the model response" failures caused by fences, preamble prose,
# or a stray brace. The loose schema is deliberate: the exact shape is dictated by
# the system prompt, and a rigid schema here would reject valid partial answers.
SUBMIT_TOOL = {
    "name": "submit_analysis",
    "description": (
        "Submit the completed analysis. The input must be the single JSON object "
        "whose exact structure is specified in the system prompt."
    ),
    "input_schema": {"type": "object", "additionalProperties": True},
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_HEAVY = os.environ.get("BID_MODEL_HEAVY", "claude-sonnet-5")
MODEL_LIGHT = os.environ.get("BID_MODEL_LIGHT", "claude-haiku-4-5-20251001")

# Roughly 4 characters per token. 320k characters is about 80k tokens, which
# leaves comfortable room for the system prompt and the response.
MAX_DOC_CHARS = int(os.environ.get("BID_MAX_DOC_CHARS", "320000"))

PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apto_profile.json")

ALLOWED_ORIGINS = [
    "https://aptostrategic.com",
    "https://www.aptostrategic.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

    # max_tokens is a ceiling, not a charge. Raised well above expected output so a
    # long compliance matrix on a big PDF cannot be cut off mid-structure.
TASKS = {
    "fit": {"model": MODEL_HEAVY, "max_tokens": 12000},
    "compliance": {"model": MODEL_HEAVY, "max_tokens": 32000},
    "proposal": {"model": MODEL_HEAVY, "max_tokens": 32000},
    "qa": {"model": MODEL_LIGHT, "max_tokens": 8000},
}

FACTOR_KEYS = [
    "capability_fit",
    "set_aside_advantage",
    "past_performance_match",
    "competitive_landscape",
    "effort_vs_value",
    "timeline_feasibility",
    "financial_contractual_risk",
]

_profile_cache = {"data": None, "mtime": 0}


def load_profile():
    """Read apto_profile.json, re-reading only when the file changes on disk."""
    try:
        mtime = os.path.getmtime(PROFILE_PATH)
        if _profile_cache["data"] is None or mtime != _profile_cache["mtime"]:
            with open(PROFILE_PATH, "r", encoding="utf-8") as fh:
                _profile_cache["data"] = json.load(fh)
            _profile_cache["mtime"] = mtime
        return _profile_cache["data"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not load apto_profile.json from %s: %s" % (PROFILE_PATH, exc)
        )


# ---------------------------------------------------------------------------
# Document handling
# ---------------------------------------------------------------------------

PRIORITY_PATTERNS = [
    r"(?i)\bsection\s+[LM]\b",
    r"(?i)\bevaluation\s+(?:criteria|factors)\b",
    r"(?i)\binstructions\s+to\s+(?:offerors|bidders|respondents|proposers)\b",
    r"(?i)\bsubmission\s+(?:requirements|instructions)\b",
    r"(?i)\bscope\s+of\s+(?:work|services)\b",
    r"(?i)\bstatement\s+of\s+work\b",
    r"(?i)\bperformance\s+work\s+statement\b",
    r"(?i)\bminimum\s+qualifications\b",
    r"(?i)\bmandatory\s+requirements\b",
    r"(?i)\bhistorically\s+underutilized\b",
    r"(?i)\bdue\s+(?:date|no\s+later\s+than)\b",
    r"(?i)\bpage\s+limit\b",
    r"(?i)\bpricing\s+(?:schedule|sheet|form)\b",
    r"(?i)\bperiod\s+of\s+performance\b",
    r"(?i)\bset[\-\s]?aside\b",
    r"(?i)\bNAICS\b",
    r"(?i)\binsurance\s+requirements\b",
    r"(?i)\bbond\b",
]


def condense(text, limit=MAX_DOC_CHARS):
    """
    Keep the whole document when it fits. When it does not, keep the front
    matter, the tail, and every window around a high-signal heading, so the
    parts that decide compliance and fit survive the trim.
    """
    text = text or ""
    if len(text) <= limit:
        return text, False

    head_len = int(limit * 0.30)
    tail_len = int(limit * 0.12)
    mid_budget = limit - head_len - tail_len

    head = text[:head_len]
    tail = text[-tail_len:]
    middle = text[head_len:-tail_len]

    spans = []
    for pat in PRIORITY_PATTERNS:
        for m in re.finditer(pat, middle):
            spans.append((max(0, m.start() - 900), min(len(middle), m.end() + 5200)))

    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 400:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    chunks, used = [], 0
    for start, end in merged:
        piece = middle[start:end]
        if used + len(piece) > mid_budget:
            piece = piece[: max(0, mid_budget - used)]
        if piece:
            chunks.append(piece)
            used += len(piece)
        if used >= mid_budget:
            break

    body = "\n\n[...document trimmed between excerpts...]\n\n".join(chunks)
    out = (
        head
        + "\n\n[...document trimmed. High-signal excerpts follow...]\n\n"
        + body
        + "\n\n[...document trimmed. Final pages follow...]\n\n"
        + tail
    )
    return out, True


def profile_brief(profile, task):
    """Build the profile block. Rates and identifiers only where needed."""
    keep = {
        "identity": {
            k: v
            for k, v in profile.get("identity", {}).items()
            if k
            in (
                "legal_name",
                "principal",
                "principal_title",
                "address",
                "email",
                "website",
                "entity_type",
                "employees",
                "structure_note",
            )
        },
        "socioeconomic_status": profile.get("socioeconomic_status", {}),
        "codes": profile.get("codes", {}),
        "capabilities": profile.get("capabilities", []),
        "disqualifiers": profile.get("disqualifiers", {}),
        "past_performance": profile.get("past_performance", []),
        "past_performance_gaps": profile.get("past_performance_gaps", []),
        "credentials": profile.get("credentials", []),
        "differentiators": profile.get("differentiators", []),
        "methodologies": profile.get("methodologies", {}),
        "bid_preferences": profile.get("bid_preferences", {}),
        "open_blockers": profile.get("open_blockers", []),
    }
    if task in ("proposal",):
        keep["identity"] = profile.get("identity", {})
        keep["rates"] = profile.get("rates", {})
        keep["boilerplate"] = profile.get("boilerplate", {})
    if task in ("compliance", "qa"):
        keep["texas_certifications_checklist"] = profile.get("texas_certifications_checklist", [])
        keep["federal_certifications_checklist"] = profile.get("federal_certifications_checklist", [])
    return json.dumps(keep, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

VOICE = """
WRITING RULES, NON-NEGOTIABLE
- Plain, simple, direct human language. Executive tone. Conclusion first, then support.
- Use hyphens. Never use em dashes or en dashes anywhere in your output.
- Spell out acronyms on first use, then use the acronym.
- No fluff, no hype, no filler adjectives, no motivational language.
- Short paragraphs. Active voice. Informative headings.
- Never write "SDVOSB certified" or "SDVOSB verified" or "certified service-disabled
  veteran-owned." Apto has NOT completed SBA VetCert. Correct phrasings are
  "Veteran-Owned", "Texas HUB certified", and "SDVOSB (pursuing)".
- NEVER invent past performance, client names, metrics, dollar figures, dates,
  certifications, or references. If Apto lacks something, write the literal token
  [NEED: what is missing] or [VERIFY: what to confirm] and move on. A visible gap is
  correct. A fabricated fact is a disqualifying error.
- Never state or imply that anything has been submitted to any agency.
"""

MODE_NOTE = {
    "RFP": (
        "This is a Request for Proposal. The agency wants a complete solution: technical "
        "approach, methodology, qualifications, and price, evaluated on best value rather "
        "than lowest price. Demonstrate expertise and approach, not just cost."
    ),
    "RFQ": (
        "This is a Request for Quotation. The agency already knows exactly what it wants and "
        "is primarily comparing price and terms. Keep narrative short and tight. Lead with "
        "price clarity, exact conformance to the stated specification, delivery terms, and "
        "any minimum qualification proof. Do not pad with strategy essays the buyer did not "
        "ask for. Confirm line-item conformance item by item."
    ),
    "RFI": (
        "This is a Request for Information or a Sources Sought notice. No contract will be "
        "awarded from it. The objective is twofold: give the agency the market information it "
        "asked for, and shape the eventual solicitation in Apto's favor. Emphasize capability, "
        "socioeconomic status relevant to set-aside decisions, applicable NAICS codes, and "
        "suggested scope or evaluation language that would favor a small veteran-owned firm. "
        "Do not submit pricing unless explicitly requested, and do not write a full proposal."
    ),
}


def sys_fit(profile, mode, weights, ctx, lang):
    return f"""You are the bid/no-bid analyst for Apto Strategic Consulting LLC.

{MODE_NOTE.get(mode, MODE_NOTE['RFP'])}

APTO PROFILE, the only source of truth about Apto:
{profile_brief(profile, 'fit')}
{VOICE}

TODAY IS {ctx.get('today')}.

TASK
Read the solicitation. Extract the facts, then score fit on seven factors, 1 to 5.

SCORING ANCHORS. Be strict. A 5 means Apto is unusually well positioned. A 3 means
average and winnable with real work. A 1 means this factor alone should kill the bid.

1. capability_fit. Match the scope against Apto's capabilities list and their strength
   values. If the core scope hits any entry in disqualifiers.hard as a real requirement,
   score 1. Weight in the profile is {weights.get('capability_fit')} percent.
2. set_aside_advantage. Does Apto's actual status help? Veteran-owned, Texas HUB
   certified, CMBL registered, TIPS registered, SAM active, small business. CRITICAL:
   Apto has NOT completed SBA VetCert, so a federal SDVOSB set-aside is a hard
   ineligibility, not a weak score. If the solicitation is a genuine SDVOSB set-aside,
   score 1 and add a critical blocker.
3. past_performance_match. Judge only against past_performance and be honest about
   past_performance_gaps. If the solicitation demands a minimum number of similar
   completed contracts, or a minimum revenue or years in business that Apto cannot
   meet, score 1 or 2 and say exactly which requirement fails.
4. competitive_landscape. Named incumbent, unchanged scope, wired requirements,
   very large expected bidder pool, or specs written around one vendor lower this.
5. effort_vs_value. Compare estimated value to bid_preferences min_contract_value,
   sweet spot range, and max. Weigh proposal effort: page counts, forms, oral
   presentations, site visits. A high-effort low-value bid scores low.
6. timeline_feasibility. Days from today to the deadline against
   bid_preferences min_days_to_deadline and comfortable_days_to_deadline. Under the
   minimum with heavy requirements scores 1 or 2.
7. financial_contractual_risk. Bonding, insurance limits, liquidated damages,
   unlimited liability, audited financials, DCAA accounting, binding key personnel
   commitments, staffing minimums, unfavorable payment terms. Items in
   disqualifiers.soft raise risk. Items in bid_preferences.never_bid score 1.

CRITICAL BLOCKERS are conditions that make Apto ineligible or non-responsive no matter
how good the fit is: a true SDVOSB set-aside, an expired or missing required
registration, a required certification or seal Apto does not hold, a mandatory
qualification Apto provably fails, or a deadline already past. List them plainly.

Return ONLY a JSON object, no prose outside it, no markdown fences:
{{
 "summary":{{"agency":"","solicitation_no":"","type":"RFP|RFQ|RFI|IFB|RFO",
   "scope":"3 to 5 plain sentences on what the agency actually wants",
   "codes":"NAICS, NIGP class-item, or PSC codes found",
   "set_aside":"exact set-aside or small business goal language, or None stated",
   "contract_type":"","period_of_performance":"","estimated_value":"",
   "questions_due":"","pre_bid":"","due_date":"YYYY-MM-DD if determinable else as written",
   "anticipated_award":"","submission_method":"portal, email, or physical, with format and copy count",
   "evaluation_criteria":[{{"criterion":"","weight":"points or percent or Not stated"}}]}},
 "scores":{{{", ".join('"%s":{{"score":0,"rationale":"2 to 3 sentences citing the specific solicitation language or the specific Apto profile fact that drove this number"}}' % k for k in FACTOR_KEYS)}}},
 "rationale":"3 to 5 sentences. Conclusion first. Name the single biggest reason to bid and the single biggest reason not to.",
 "critical_blockers":["only genuine eligibility or responsiveness blockers, empty array if none"],
 "conditions":["if the score lands in bid-with-conditions, the specific conditions that must be true to proceed"],
 "win_themes":[{{"theme":"","feature":"","benefit":"","proof":"a real proof point from the Apto profile, or [NEED: proof point]"}}],
 "risks":[{{"title":"","mitigation":""}}],
 "gaps":["each missing document, certification, reference, data point, or decision, phrased as an action JC can take"],
 "teaming":["only if a gap is better closed by a partner. Name the capability needed, not invented company names."]
}}
Produce 3 to 5 win themes. Write all human-readable text in {'Spanish' if lang == 'es' else 'English'}. Keep JSON keys in English."""


def sys_compliance(profile, mode, ctx, lang):
    return f"""You are the compliance analyst for Apto Strategic Consulting LLC.

{MODE_NOTE.get(mode, MODE_NOTE['RFP'])}

APTO REFERENCE:
{profile_brief(profile, 'compliance')}
{VOICE}

TASK
Build the compliance matrix. Compliance beats persuasion: one missed mandatory item
disqualifies the response regardless of writing quality.

Capture EVERY one of these as its own row:
- Every "shall", "must", "is required to", "will provide", "at a minimum", and
  "responsive proposals must" statement.
- Every submission logistic: portal, email address, physical address, copy count,
  file naming, file format, file size limits, labeling of sealed envelopes.
- Every formatting rule: page limits per volume or section, font family and size,
  margins, line spacing, tab or section ordering, allowed appendices.
- Every form, affidavit, certification, and attachment by its exact name and number.
- Every deadline: questions, pre-bid conference or site visit, addenda acknowledgment,
  proposal due date and time WITH TIME ZONE, presentation windows.
- Every eligibility and minimum qualification threshold: years in business, revenue,
  licensure, insurance limits, bonding, references, key personnel credentials.
- Every evaluation factor and its point value or weight.
- For federal solicitations, map Section L instructions to volumes and Section M
  factors to response sections.
- For Texas solicitations, check each item in texas_certifications_checklist against
  the solicitation and include the ones that apply, noting the dollar threshold.

Rules: quote or closely paraphrase the actual requirement text, do not summarize it away.
Cite the section number or page where you found it. Type is "M" for mandatory,
"D" for desirable or evaluated but not pass/fail, "I" for informational. Err toward M.
Do not invent requirements that are not in the document. Sort in document order.

Return ONLY a JSON object, no markdown fences:
{{
 "requirements":[{{"section":"section number or page reference",
   "requirement":"the requirement in its own words, specific and actionable",
   "type":"M|D|I",
   "proposal_section":"where in Apto's response this gets answered",
   "notes":"threshold, risk, or an Apto-specific caution. Empty string if none."}}],
 "forms":["exact name and number of every form, affidavit, and attachment to submit"],
 "format_rules":["each page limit, font rule, margin, naming convention, and file format rule"],
 "deadlines":[{{"item":"","when":"include the time and time zone when stated"}}],
 "missing_information":["anything the solicitation references but does not include, such as an attachment not provided, or a requirement that is ambiguous and should go into the written questions"]
}}
Be exhaustive. Forty to one hundred twenty requirement rows is normal for a full RFP.
Write human-readable text in {'Spanish' if lang == 'es' else 'English'}. Keep JSON keys in English."""


def sys_proposal(profile, mode, ctx, lang, fit_summary):
    fit_block = ""
    if fit_summary:
        fit_block = (
            "\nFIT ANALYSIS ALREADY COMPLETED. Carry these win themes through the draft:\n"
            + json.dumps(fit_summary, ensure_ascii=False)[:6000]
        )

    if mode == "RFQ":
        outline = """Use a lean quotation structure:
1. Cover letter and quotation summary
2. Conformance to specification, line item by line item
3. Pricing, on the solicitation's own form or schedule when one exists
4. Minimum qualifications and required certifications
5. Delivery, performance schedule, and terms
6. Signature and required forms"""
    elif mode == "RFI":
        outline = """Use a capability response structure:
1. Cover letter and firm introduction
2. Firm profile, socioeconomic status, and applicable NAICS codes
3. Capability narrative answering each question the notice asked, in the order asked
4. Relevant experience
5. Answers to any market research questions, including recommended scope, contract
   vehicle, and evaluation approach that would favor a small veteran-owned firm
6. Point of contact
Do not include pricing unless the notice explicitly asks for it."""
    else:
        outline = """Use the seven-section framework:
1. Cover letter and executive summary. Lead with the agency's problem, not Apto's
   history. Client-focused. Differentiators early. Concrete and specific.
2. Scope of work and technical approach. Phases, deliverables per phase, timeframes,
   named methodologies from the profile, tools. A milestone table.
3. Budget and pricing. Use the solicitation's own pricing form when one exists. Otherwise
   a transparent breakdown from the rate card. State what is included and what is not.
   Net 30, no deposit. Travel at actual cost per Federal Travel Regulation per diem.
4. Company qualifications. Real past performance only, with before-and-after results
   where they exist. Certifications, credentials, principal biography.
5. Terms, assumptions, and exceptions. Every exception is a risk decision, flag it as one.
6. Closing statement. Restate strengths in two sentences, give a clear next step,
   signature block.
7. Optional sections when the solicitation calls for them: risk management plan,
   quality control, project management approach, staffing, transition in and out,
   HUB Subcontracting Plan, conflict of interest, insurance, required forms."""

    return f"""You are the proposal writer for Apto Strategic Consulting LLC.

{MODE_NOTE.get(mode, MODE_NOTE['RFP'])}

APTO PROFILE, the only source of truth. Use the boilerplate blocks verbatim where they
fit, and adapt them where the solicitation calls for something specific:
{profile_brief(profile, 'proposal')}
{VOICE}
{fit_block}

TASK
Draft the response.

THE SOLICITATION'S REQUIRED STRUCTURE ALWAYS OVERRIDES THE DEFAULT OUTLINE. If the
document dictates a response format, tab order, or volume breakdown, mirror it exactly
and say so in structure_note. Answer the evaluation criteria in the order the agency
listed them. Echo the agency's own vocabulary back to it.

Default outline when the solicitation dictates nothing:
{outline}

WRITING REQUIREMENTS
- Address the specific agency and the specific problem in this document. A section that
  could be pasted into any other proposal is a failed section. Name the agency, quote
  its stated objectives, reference its stated pain points.
- Every claim gets a proof point from the profile. No proof available means write
  [NEED: proof point for X], not a vague assertion.
- Insert [FILL: what is needed] for anything that requires JC's input, and
  [VERIFY: what to confirm] for any profile fact that carries a verify note.
- Use tables where a table is clearer than prose. Write them as markdown tables.
- Use "### Subheading" for subheadings inside a section and "**bold**" for emphasis.
- Realistic length. An executive summary is one page. A technical approach for a
  mid-size consulting engagement is three to six pages. Do not pad to look thorough.

Return ONLY a JSON object, no markdown fences:
{{
 "structure_note":"one or two sentences on the structure you followed and why, naming the solicitation section that dictated it if any",
 "sections":[{{"title":"section title as the agency would expect to see it",
   "guidance":"one short line to JC on what to check or personalize in this section before submitting",
   "content":"the full drafted text of the section"}}],
 "pricing_note":"how the price should be presented for this specific solicitation, including which form to use and what to confirm before submitting",
 "open_items":["everything JC must resolve before this can be submitted, phrased as actions"]
}}
Write all content in {'Spanish' if lang == 'es' else 'English'}. Keep JSON keys in English."""


def sys_qa(profile, mode, ctx, lang):
    return f"""You are the submission quality control reviewer for Apto Strategic Consulting LLC.

{MODE_NOTE.get(mode, MODE_NOTE['RFP'])}

APTO REFERENCE:
{profile_brief(profile, 'qa')}
{VOICE}

TODAY IS {ctx.get('today')}.

TASK
Build the final submission checklist for this specific solicitation. Not a generic one.
Every item must be verifiable by looking at the document or the response package.

Group the checklist as:
- Forms and signatures. Each required form by exact name and number, and who signs.
- Format and packaging. Page limits per volume, font, margins, file naming, file format,
  file size, copy count, envelope labeling or portal upload slots.
- Content completeness. Every mandatory requirement answered, every evaluation factor
  addressed, every question in the solicitation answered, all [FILL], [NEED], and
  [VERIFY] tokens resolved.
- Registrations and eligibility. SAM active, UEI on all forms, CMBL current, Texas HUB
  current, franchise tax active, insurance certificates, and any licensure. Flag any
  registration in the profile that is expired or unresolved as a stop item.
- Delivery. Portal account tested and a practice upload done, correct submission address,
  addenda all acknowledged, timestamp confirmation captured, submitted at least
  24 hours early because late equals rejected with no exceptions.

Return ONLY a JSON object, no markdown fences:
{{
 "due_date":"YYYY-MM-DD if determinable, else as written in the document",
 "submission_method":"exactly how and where it goes, with format and copy requirements",
 "groups":[{{"name":"group name","items":["specific verifiable check"]}}],
 "open_items":["anything that would block submission right now, most urgent first"],
 "final_note":"one or two sentences on the single biggest submission risk for this bid"
}}
Write human-readable text in {'Spanish' if lang == 'es' else 'English'}. Keep JSON keys in English."""


BUILDERS = {
    "fit": lambda p, m, w, c, l, f: sys_fit(p, m, w, c, l),
    "compliance": lambda p, m, w, c, l, f: sys_compliance(p, m, c, l),
    "proposal": lambda p, m, w, c, l, f: sys_proposal(p, m, c, l, f),
    "qa": lambda p, m, w, c, l, f: sys_qa(p, m, c, l),
}


# ---------------------------------------------------------------------------
# JSON recovery
# ---------------------------------------------------------------------------

def parse_json(text):
    """Model output to dict, tolerant of fences and trailing prose."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start == -1:
        raise ValueError("The model returned no JSON object.")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start : i + 1])
                except json.JSONDecodeError:
                    break
    # Truncated response: close what is open and salvage the complete entries.
    frag = t[start:]
    frag = re.sub(r",\s*$", "", frag.rstrip())
    for _ in range(60):
        try:
            return json.loads(frag)
        except json.JSONDecodeError as exc:
            msg = str(exc)
            if "Unterminated string" in msg:
                frag = frag[: frag.rfind('"')]
                frag = frag[: max(frag.rfind(","), frag.rfind("["), frag.rfind("{"))]
                frag = re.sub(r"[,\[{]\s*$", "", frag)
            elif frag.count("[") > frag.count("]"):
                frag = re.sub(r",\s*$", "", frag.rstrip()) + "]"
            elif frag.count("{") > frag.count("}"):
                frag = re.sub(r",\s*$", "", frag.rstrip()) + "}"
            else:
                break
    raise ValueError("Could not parse the model response as JSON.")


def normalize(task, data, weights):
    """Guarantee the shape the front end expects."""
    if not isinstance(data, dict):
        raise ValueError("The model response was not a JSON object.")

    if task == "fit":
        data.setdefault("summary", {})
        scores = data.setdefault("scores", {})
        for k in FACTOR_KEYS:
            entry = scores.get(k)
            if not isinstance(entry, dict):
                entry = {"score": 0, "rationale": "Not scored by the model."}
            try:
                s = int(round(float(entry.get("score") or 0)))
            except (TypeError, ValueError):
                s = 0
            entry["score"] = max(0, min(5, s))
            entry.setdefault("rationale", "")
            scores[k] = entry
        for k in ("critical_blockers", "conditions", "win_themes", "risks", "gaps", "teaming"):
            if not isinstance(data.get(k), list):
                data[k] = []
        data.setdefault("rationale", "")

        # Server-side recomputation so the recommendation cannot drift from the math.
        total = sum(weights.get(k, 0) for k in FACTOR_KEYS if scores[k]["score"])
        num = sum(scores[k]["score"] * weights.get(k, 0) for k in FACTOR_KEYS)
        avg = (num / total) if total else 0.0
        data["computed_score"] = round(avg, 2)
        if any(scores[k]["score"] == 1 for k in FACTOR_KEYS) or data["critical_blockers"]:
            data["recommendation"] = "NO-BID"
        elif avg >= 3.5:
            data["recommendation"] = "BID"
        elif avg >= 2.75:
            data["recommendation"] = "BID WITH CONDITIONS"
        else:
            data["recommendation"] = "NO-BID"

    elif task == "compliance":
        reqs = data.get("requirements")
        if not isinstance(reqs, list):
            reqs = []
        clean = []
        for r in reqs:
            if not isinstance(r, dict) or not str(r.get("requirement", "")).strip():
                continue
            t = str(r.get("type", "D")).strip().upper()[:1]
            r["type"] = t if t in ("M", "D", "I") else "D"
            for f in ("section", "proposal_section", "notes"):
                r.setdefault(f, "")
            r["done"] = False
            clean.append(r)
        data["requirements"] = clean
        for k in ("forms", "format_rules", "deadlines", "missing_information"):
            if not isinstance(data.get(k), list):
                data[k] = []

    elif task == "proposal":
        secs = data.get("sections")
        if not isinstance(secs, list):
            secs = []
        data["sections"] = [
            {
                "title": str(s.get("title", "Untitled section")),
                "guidance": str(s.get("guidance", "")),
                "content": str(s.get("content", "")),
            }
            for s in secs
            if isinstance(s, dict) and str(s.get("content", "")).strip()
        ]
        data.setdefault("structure_note", "")
        if not isinstance(data.get("open_items"), list):
            data["open_items"] = []

    elif task == "qa":
        groups = data.get("groups")
        if not isinstance(groups, list):
            groups = []
        data["groups"] = [
            {
                "name": str(g.get("name", "Checklist")),
                "items": [str(i.get("item") if isinstance(i, dict) else i) for i in (g.get("items") or [])],
            }
            for g in groups
            if isinstance(g, dict)
        ]
        if not isinstance(data.get("open_items"), list):
            data["open_items"] = []

    return data


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

def cors(resp):
    origin = request.headers.get("Origin", "")
    resp.headers["Access-Control-Allow-Origin"] = origin if origin in ALLOWED_ORIGINS else ALLOWED_ORIGINS[0]
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Max-Age"] = "86400"
    resp.headers["Vary"] = "Origin"
    return resp


@bid_agent_bp.route("/bid-agent", methods=["OPTIONS"])
def bid_agent_options():
    return cors(make_response("", 204))


@bid_agent_bp.route("/bid-agent/health", methods=["GET"])
def bid_agent_health():
    try:
        p = load_profile()
        ok = True
        note = "Profile loaded. Version %s, updated %s." % (
            p.get("_version"),
            p.get("_last_updated"),
        )
    except Exception as exc:  # noqa: BLE001
        ok, note = False, str(exc)
    return cors(
        make_response(
            jsonify(
                {
                    "ok": ok,
                    "code_version": CODE_VERSION,
                    "uses_prefill": False,
                    "profile": note,
                    "key_present": bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip()),
                    "sdk_present": Anthropic is not None,
                    "models": {"heavy": MODEL_HEAVY, "light": MODEL_LIGHT},
                }
            ),
            200 if ok else 500,
        )
    )


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@bid_agent_bp.route("/bid-agent", methods=["POST"])
def bid_agent():
    started = time.time()
    task = "unknown"
    try:
        if Anthropic is None:
            raise RuntimeError("The anthropic package is not installed. Add anthropic to requirements.txt.")
        # .strip() matters: the Railway variable has carried trailing whitespace before,
        # which is why the existing ask_cos handler strips it too.
        api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the Railway environment.")

        body = request.get_json(force=True, silent=True) or {}
        task = str(body.get("task", "fit")).lower()
        if task not in TASKS:
            raise ValueError("Unknown task: %s" % task)

        mode = str(body.get("mode", "RFP")).upper()
        if mode not in MODE_NOTE:
            mode = "RFP"
        lang = "es" if str(body.get("lang", "en")).lower().startswith("es") else "en"
        ctx = body.get("context") or {}
        ctx.setdefault("today", time.strftime("%Y-%m-%d"))

        weights = {}
        incoming = body.get("weights") or {}
        profile = load_profile()
        defaults = profile.get("scoring_weights", {})
        for k in FACTOR_KEYS:
            try:
                weights[k] = float(incoming.get(k, defaults.get(k, 100 / len(FACTOR_KEYS))))
            except (TypeError, ValueError):
                weights[k] = float(defaults.get(k, 100 / len(FACTOR_KEYS)))

        doc = str(body.get("document") or "")
        if len(doc.strip()) < 200:
            raise ValueError("The solicitation text is too short to analyze. Paste at least a few paragraphs.")
        doc, trimmed = condense(doc)

        system = BUILDERS[task](profile, mode, weights, ctx, lang, body.get("fit_summary"))

        ctx_lines = [
            "Solicitation type selected by the user: %s" % mode,
            "Today: %s" % ctx.get("today"),
        ]
        for label, key in (
            ("Agency", "agency"),
            ("Solicitation number", "solicitation_no"),
            ("Response due date", "due_date"),
            ("Estimated value in USD", "estimated_value"),
        ):
            if ctx.get(key):
                ctx_lines.append("%s provided by the user: %s" % (label, ctx[key]))
        if trimmed:
            ctx_lines.append(
                "NOTE: the document exceeded the size limit. It was trimmed to the front "
                "matter, the closing pages, and excerpts around high-signal headings. "
                "Where a fact is not present in what you can see, say so rather than guessing."
            )

        user = "%s\n\n===== SOLICITATION DOCUMENT BEGINS =====\n%s\n===== SOLICITATION DOCUMENT ENDS =====" % (
            "\n".join(ctx_lines),
            doc,
        )

        client = Anthropic(api_key=api_key, timeout=600.0, max_retries=2)
        cfg = TASKS[task]
        # Forced tool use: the answer comes back as an already-parsed JSON object.
        # No prefill (newer models reject it) and no text parsing in the happy path.
        msg = client.messages.create(
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[SUBMIT_TOOL],
            tool_choice={"type": "tool", "name": SUBMIT_TOOL["name"]},
        )

        payload, raw = None, ""
        for b in msg.content:
            btype = getattr(b, "type", "")
            if btype == "tool_use" and isinstance(getattr(b, "input", None), dict):
                payload = b.input
                break
            if btype == "text":
                raw += b.text

        if payload is None:
            # Fall back to text parsing if the model answered in prose anyway.
            if not raw.strip():
                raise ValueError(
                    "The model returned neither a tool call nor text. stop_reason=%s"
                    % getattr(msg, "stop_reason", None)
                )
            try:
                payload = parse_json(raw)
            except ValueError as exc:
                raise ValueError(
                    "%s stop_reason=%s output_tokens=%s. First 400 characters of the "
                    "response: %s"
                    % (
                        exc,
                        getattr(msg, "stop_reason", None),
                        getattr(msg.usage, "output_tokens", None),
                        raw[:400].replace("\n", " "),
                    )
                )

        if getattr(msg, "stop_reason", None) == "max_tokens":
            raise ValueError(
                "The response hit the %s token ceiling for task '%s' and is incomplete. "
                "Raise max_tokens for this task in bid_agent.py, or run the stages "
                "individually instead of all four at once."
                % (cfg["max_tokens"], task)
            )

        data = normalize(task, payload, weights)
        data["_meta"] = {
            "task": task,
            "mode": mode,
            "lang": lang,
            "model": cfg["model"],
            "doc_chars": len(doc),
            "trimmed": trimmed,
            "elapsed_s": round(time.time() - started, 1),
            "input_tokens": getattr(msg.usage, "input_tokens", None),
            "output_tokens": getattr(msg.usage, "output_tokens", None),
            "stop_reason": getattr(msg, "stop_reason", None),
        }
        return cors(make_response(jsonify(data), 200))

    except Exception as exc:  # noqa: BLE001
        return cors(
            make_response(
                jsonify(
                    {
                        "error": "%s: %s" % (type(exc).__name__, exc),
                        "task": task,
                        "code_version": CODE_VERSION,
                        "elapsed_s": round(time.time() - started, 1),
                    }
                ),
                500,
            )
        )
