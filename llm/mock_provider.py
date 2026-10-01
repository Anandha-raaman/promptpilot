"""Mock LLM Provider for offline portfolio demonstration and testing.

Provides realistic, deterministic prompt analyses, 5-strategy variants, completions,
and evaluations with zero API calls and zero rate limits.
"""

from typing import Optional, Type, TypeVar
from pydantic import BaseModel

from evaluation.scoring import EvaluationScore
from llm.base import BaseLLMProvider, LLMResponse
from prompts.analyzer import PromptAnalysisResult
from prompts.generator import GeneratedVariantsBundle, PromptVariant

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(BaseLLMProvider):
    """Offline demo provider generating high-fidelity mock results for testing without consuming API quota."""

    def is_available(self) -> bool:
        return True

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a realistic mock response dynamically matching the task and source data."""
        import re

        lower_prompt = prompt.lower()

        # Extract source data if delimited
        source_data = ""
        src_match = re.search(r"<SOURCE_DATA>(.*?)</SOURCE_DATA>", prompt, re.DOTALL | re.IGNORECASE)
        if not src_match:
            src_match = re.search(r"<SOURCE_DOCUMENT>(.*?)</SOURCE_DOCUMENT>", prompt, re.DOTALL | re.IGNORECASE)
        if src_match:
            source_data = src_match.group(1).strip()

        lower_src = source_data.lower()

        # Content generation strictly derived from the provided source data
        if "cve-" in lower_src or "opengateway" in lower_src or "vulnerability" in lower_src:
            content = (
                "**Security Advisory Summary:**\n"
                "• **Vulnerability:** A critical remote code execution flaw (CVE-2026-19342, CVSS v3.1 score 9.6) was discovered in OpenGateway proxy library (versions 3.2.0 to 3.4.1).\n"
                "• **Impact:** Unauthenticated remote attackers can inject arbitrary headers into internal RPC dispatchers.\n"
                "• **Mitigation:** Upgrade immediately to version 3.4.2 or disable experimental WebSocket multiplexing in gateway.conf."
            )
        elif "apex quantum" in lower_src or "nasdaq: apxq" in lower_src or ("$842 million" in lower_src and "revenue" in lower_src):
            content = (
                "**Quarterly Financial Summary:**\n"
                "• **Apex Quantum Semiconductor (NASDAQ: APXQ):** Reported Q1 2026 revenue of $842 million (+18% YoY), with net income rising to $112 million.\n"
                "• **Performance Drivers:** CEO David Sterling noted surging demand for cryogenic quantum control chips.\n"
                "• **Q2 Outlook:** CFO Maria Santos guided revenues between $890M and $915M following a delivery agreement with HyperScale Compute Labs."
            )
        elif "federal health authority" in lower_src or "clinical machine learning" in lower_src or "class ii diagnostic" in lower_src:
            content = (
                "**Clinical AI Regulatory Framework Summary:**\n"
                "• **Effective Date:** Federal Health Authority oversight framework mandates compliance beginning January 2027.\n"
                "• **Class II Requirements:** Developers must maintain training distribution audit logs, ensure demographic parity, and submit annual post-market surveillance reports.\n"
                "• **Class III Emergency Triage:** Algorithms deployed in emergency settings are classified as high-risk and require randomized prospective clinical trials."
            )
        elif "billing api" in lower_src or "deprecating rest v1" in lower_src or "graphql v2" in lower_src:
            content = (
                "**Developer Notification: API Deprecation:**\n"
                "• **Sunset Date:** REST v1 endpoints for Billing API will be deprecated on November 30, 2026.\n"
                "• **Migration Target:** Developers must transition to GraphQL v2 API, supporting webhook streaming and granular token scopes.\n"
                "• **Documentation:** Migration guides and SDK wrappers are available at docs.cloudbilling.io/v2-migration."
            )
        elif "alerting feature in promptpilot" in lower_src or ("3.0 seconds" in lower_src and "batch" in lower_src):
            content = (
                "**Product Acceptance Criteria — Latency Alerting:**\n"
                "1. **User Story:** As a developer using PromptPilot, I want real-time notifications when a model takes longer than 3.0 seconds so I can identify slow executions.\n"
                "2. **Criteria:** Alert triggers when execution latency exceeds 3.0s during batch runs.\n"
                "3. **Scope:** Detects slow models and unoptimized token lengths."
            )
        elif "techvanguard" in lower_src or "serverless compute across european" in lower_src or "$340,000" in lower_src:
            content = (
                "**Cloud Architecture Migration Post-Mortem:**\n"
                "• **Compute Cost:** European serverless migration achieved a 28% reduction in operational compute expenses.\n"
                "• **Budget Overrun:** Ingress bandwidth estimation gaps resulted in a $340,000 cost overrun against the $1.5M budget.\n"
                "• **Latency & Remedy:** Infrequently invoked services experienced cold start increases up to 480ms p99; provisioned concurrency has been mandated for auth endpoints."
            )
        elif "electric vehicle" in lower_src or "evs" in lower_src:
            content = (
                "**Electric Vehicles Summary:**\n"
                "• **Market Shift:** Electric vehicles (EVs) are transitioning to mainstream adoption, driven by regulatory targets and battery advancements.\n"
                "• **Challenges:** Charging infrastructure bottlenecks and raw material supply chain pressures remain key obstacles.\n"
                "• **Outlook:** Solid-state battery development and fleet electrification continue to accelerate."
            )
        elif "revenue grew 25%" in lower_src or ("25%" in lower_src and "$500m" in lower_src and "revenue" in lower_src):
            content = (
                "**Key Financial Metrics Summary:**\n"
                "• **Revenue Growth:** Revenue grew 25% to $500M in Q3 2026."
            )
        elif "operating expenses decreased from $210m" in lower_src or ("$180m" in lower_src and "$210m" in lower_src):
            content = (
                "**Extracted Financial Figures:**\n"
                "• Operating Expenses: Decreased from $210M to $180M\n"
                "• Revenue: Increased to $500M (Q3 2026)"
            )
        elif "18,500 customers" in lower_src or "142,000" in lower_src:
            content = (
                "**Customer Growth Summary:**\n"
                "• Added 18,500 new customers during Q3 2026.\n"
                "• Total customer base reached 142,000."
            )
        elif source_data:
            # Dynamically extract and synthesize statements from arbitrary source data
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source_data) if s.strip()]
            if len(sentences) >= 2:
                points = sentences[:3]
                content = "Summary:\n" + "\n".join(f"• {p}" for p in points)
            else:
                content = f"Summary:\n{source_data[:240]}"
        else:
            # Fallback for general prompts without explicit source data
            if "role-based" in lower_prompt or "analyst" in lower_prompt:
                content = (
                    "**Executive Analysis:**\n"
                    "The operational assessment highlights key efficiencies achieved alongside identified risk vectors. "
                    "Leadership recommends targeted mitigations to sustain stability and manage budget targets."
                )
            elif "constraint" in lower_prompt or "bullet" in lower_prompt:
                content = (
                    "• Core objective completed within specified constraints.\n"
                    "• Key parameters and performance metrics validated.\n"
                    "• Next operational milestones established for review."
                )
            elif "json" in lower_prompt or "extract" in lower_prompt:
                content = '{\n  "status": "success",\n  "key_findings": ["Objective validated", "Parameters confirmed"]\n}'
            else:
                content = (
                    "Summary:\n"
                    "The task was processed according to instructions, highlighting core observations and actionable next steps."
                )

        return LLMResponse(
            content=content,
            model_name="mock-demo-engine",
            latency_ms=180.0,
            prompt_tokens=65,
            completion_tokens=42,
            total_tokens=107,
        )

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> tuple[T, LLMResponse]:
        """Generate realistic structured data matching the requested schema."""
        import re

        if response_schema == PromptAnalysisResult:
            prompt_match = re.search(r"<(?:UNTRUSTED_)?USER_PROMPT>(.*?)</(?:UNTRUSTED_)?USER_PROMPT>", prompt, re.DOTALL | re.IGNORECASE)
            raw_user_prompt = prompt_match.group(1).strip() if prompt_match else prompt
            lower_user_prompt = raw_user_prompt.lower()

            is_weak = (
                lower_user_prompt in ["summarize this.", "summarize this", "tell me about this.", "tell me about this", "explain this.", "explain this"]
                or (len(raw_user_prompt) < 25 and not any(w in lower_user_prompt for w in ["bullet", "json", "constraint", "not", "limit"]))
            )

            if is_weak:
                mock_data = PromptAnalysisResult(
                    overall_analysis=(
                        "The prompt is minimal and open-ended. It lacks specific guidance on length, format, persona, "
                        "and negative constraints, relying entirely on default model behavior."
                    ),
                    clarity_score=5,
                    context_score=3,
                    constraint_score=1,
                    output_format_score=2,
                    specificity_score=3,
                    issues=[
                        "Prompt is open-ended and lacks task specificity.",
                        "No target length or output structure specified.",
                        "No negative constraints or boundary conditions provided.",
                    ],
                    suggestions=[
                        "Specify the desired output format (e.g. 3 bullet points, executive summary).",
                        "Define length limits (e.g. max 150 words).",
                        "Assign an expert role and specify focus areas.",
                    ],
                    has_role=False,
                    has_constraints=False,
                    has_format_specification=False,
                )
                return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=120.0)

            mock_data = PromptAnalysisResult(
                overall_analysis=(
                    "The prompt provides a general objective but lacks explicit negative constraints, output length boundaries, "
                    "and a defined domain persona. Adding structured XML tags and clear formatting directives will improve consistency."
                ),
                clarity_score=7,
                context_score=6,
                constraint_score=5,
                output_format_score=6,
                specificity_score=7,
                issues=[
                    "Output length is unconstrained, risking verbose completions.",
                    "No expert persona assigned to ground technical depth.",
                    "Negative constraints (e.g. 'do not extrapolate') are omitted.",
                ],
                suggestions=[
                    "Assign a role (e.g. 'You are a Senior Staff Architect').",
                    "Specify exact output structure (e.g. 3 bullet points with bold prefixes).",
                    "Incorporate boundary rules restricting assumptions.",
                ],
                has_role=False,
                has_constraints=False,
                has_format_specification=True,
            )
            return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=120.0)

        elif response_schema == GeneratedVariantsBundle:
            mock_data = GeneratedVariantsBundle(
                variants=[
                    PromptVariant(
                        strategy_name="Role-Based Prompting",
                        strategy_description="Assigns senior domain analyst persona to establish authority and domain depth.",
                        prompt_text=(
                            "You are an expert summarization assistant.\n\n"
                            "Summarize the provided source data for executive leadership.\n"
                            "Focus on the main benefits, challenges, and future outlook.\n"
                            "Keep the summary concise and present it in clear paragraphs.\n\n"
                            "Use only information from the provided source data.\n"
                            "Do not introduce unsupported claims."
                        ),
                        expected_benefit="Elevates analytical precision and tailors perspective to executive decision-makers.",
                        potential_limitations="May be slightly formal for general consumer audiences.",
                    ),
                    PromptVariant(
                        strategy_name="Structured Delimiters",
                        strategy_description="Structures processing stages with clear sections and explicit execution steps.",
                        prompt_text=(
                            "### INSTRUCTIONS\n"
                            "1. Analyze the provided source data thoroughly.\n"
                            "2. Extract core findings, observed risks, and corrective actions.\n"
                            "3. Output a structured executive brief with bold section headers.\n\n"
                            "### GUARDRAILS\n"
                            "- Base all statements strictly on the provided source data.\n"
                            "- Do not introduce external assumptions or speculative claims."
                        ),
                        expected_benefit="Enforces structured section hierarchy and minimizes ambiguity.",
                        potential_limitations="Slightly increases prompt token overhead.",
                    ),
                    PromptVariant(
                        strategy_name="Constraint-Based Prompting",
                        strategy_description="Imposes strict negative boundaries and bullet ceilings to eliminate fluff.",
                        prompt_text=(
                            "Summarize the provided source data according to these strict rules:\n"
                            "- Exactly 3 bullet points, each starting with an action verb.\n"
                            "- Maximum 20 words per bullet point.\n"
                            "- Use only information from the provided source data.\n"
                            "- Do NOT include conversational filler, introductory greetings, or unsupported claims."
                        ),
                        expected_benefit="Strictly bounds length and forces high information density.",
                        potential_limitations="Can omit minor contextual nuances due to length limits.",
                    ),
                    PromptVariant(
                        strategy_name="Structured-Output Prompting",
                        strategy_description="Fences processing and derives a strictly-typed JSON schema from source data.",
                        prompt_text=(
                            "Analyze the provided source data and output a JSON object with the following schema:\n"
                            "{\n"
                            '  "achievements": string[],\n'
                            '  "risks_and_issues": string[],\n'
                            '  "next_steps": string[]\n'
                            "}\n\n"
                            "Requirements:\n"
                            "- Extract values solely from the provided source data.\n"
                            "- Do not extrapolate or introduce ungrounded claims."
                        ),
                        expected_benefit="Enforces downstream programmatic parsing and zero hallucination.",
                        potential_limitations="Requires caller to parse JSON rather than read prose.",
                    ),
                ]
            )
            return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=150.0)

        elif response_schema == EvaluationScore:
            inst_match = re.search(r"<PROMPT_INSTRUCTION>(.*?)</PROMPT_INSTRUCTION>", prompt, re.DOTALL | re.IGNORECASE)
            src_match = re.search(r"<SOURCE_DOCUMENT>(.*?)</SOURCE_DOCUMENT>", prompt, re.DOTALL | re.IGNORECASE)
            resp_match = re.search(r"<GENERATED_RESPONSE>(.*?)</GENERATED_RESPONSE>", prompt, re.DOTALL | re.IGNORECASE)

            instruction = inst_match.group(1).strip() if inst_match else ""
            source_doc = src_match.group(1).strip() if src_match else ""
            gen_resp = resp_match.group(1).strip() if resp_match else ""

            lower_inst = instruction.lower()
            lower_src = source_doc.lower()
            lower_resp = gen_resp.lower()

            # ---------------------------------------------------------
            # 1. Contamination / Alien Domain Detection (TEST 3)
            # ---------------------------------------------------------
            is_source_ev = any(w in lower_src for w in ["electric vehicle", "ev ", "evs", "battery pack"])
            is_resp_ev = any(w in lower_resp for w in ["electric vehicle", "ev ", "evs", "battery pack", "charging infrastructure"])

            is_source_cve = any(w in lower_src for w in ["cve-", "vulnerability", "opengateway"])
            is_resp_cve = any(w in lower_resp for w in ["cve-", "vulnerability", "opengateway", "remote code execution"])

            is_source_fin = any(w in lower_src for w in ["apex quantum", "nasdaq: apxq", "$842 million"])
            is_resp_fin = any(w in lower_resp for w in ["apex quantum", "apxq", "$842 million", "$112 million", "david sterling"])

            is_source_health = any(w in lower_src for w in ["health authority", "diagnostic aids", "class ii", "class iii"])
            is_resp_health = any(w in lower_resp for w in ["health authority", "diagnostic aids", "class ii", "class iii", "demographic parity"])

            is_source_api = any(w in lower_src for w in ["billing api", "deprecating rest v1", "graphql v2"])
            is_resp_api = any(w in lower_resp for w in ["billing api", "rest v1", "graphql v2", "cloudbilling.io"])

            is_source_alert = any(w in lower_src for w in ["promptpilot", "3.0 second"])
            is_resp_alert = any(w in lower_resp for w in ["promptpilot", "3.0 second", "latency alert"])

            is_source_mig = any(w in lower_src for w in ["techvanguard", "$340,000", "serverless compute", "cloud migration"])
            is_resp_mig = any(w in lower_resp for w in ["techvanguard", "$340,000", "serverless compute", "cold start", "cloud migration"])

            contaminated = False
            contamination_reason = ""

            if is_resp_ev and not is_source_ev and (is_source_mig or is_source_cve or is_source_fin or is_source_health or is_source_api or is_source_alert or len(lower_src) > 50):
                contaminated = True
                contamination_reason = "Response discusses electric vehicles and automotive topics which are completely absent from the source document."
            elif is_resp_mig and not is_source_mig and (is_source_ev or is_source_cve or is_source_fin or is_source_health or is_source_api or is_source_alert or len(lower_src) > 50):
                contaminated = True
                contamination_reason = "Response discusses cloud architecture migration which is completely absent from the source document."
            elif is_resp_cve and not is_source_cve and (is_source_ev or is_source_mig or is_source_fin or is_source_health or is_source_api or is_source_alert):
                contaminated = True
                contamination_reason = "Response discusses CVE security vulnerabilities which are completely absent from the source document."
            elif is_resp_fin and not is_source_fin and (is_source_ev or is_source_mig or is_source_cve or is_source_health or is_source_api or is_source_alert):
                contaminated = True
                contamination_reason = "Response discusses semiconductor financial earnings which are completely absent from the source document."
            elif is_resp_health and not is_source_health and (is_source_ev or is_source_mig or is_source_cve or is_source_fin or is_source_api or is_source_alert):
                contaminated = True
                contamination_reason = "Response discusses clinical AI health regulatory guidelines which are completely absent from the source document."
            elif is_resp_api and not is_source_api and (is_source_ev or is_source_mig or is_source_cve or is_source_fin or is_source_health or is_source_alert):
                contaminated = True
                contamination_reason = "Response discusses API deprecation notices which are completely absent from the source document."

            if contaminated:
                mock_data = EvaluationScore(
                    relevance=1,
                    completeness=2,
                    instruction_following=2,
                    format_compliance=6,
                    conciseness=6,
                    factual_consistency=1,
                    feedback=[
                        contamination_reason,
                        "Severe factual inconsistency: response is completely ungrounded in the provided source document.",
                    ],
                    strengths=["Uses syntactically well-formed sentences."],
                )
                return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=110.0)

            # ---------------------------------------------------------
            # 2. Factual Inconsistency / Direct Contradiction (TEST 4 / TEST B)
            # ---------------------------------------------------------
            has_contradiction = False
            contradiction_detail = ""

            if ("$500m" in lower_src or "$500 million" in lower_src or "revenue of $500" in lower_src or "revenue = $500" in lower_src):
                if "$700m" in lower_resp or "$700 million" in lower_resp or "700m" in lower_resp:
                    has_contradiction = True
                    contradiction_detail = "Response asserts revenue of $700M, directly contradicting the source document which states $500M."
            elif ("3.0 second" in lower_src or "3.0s" in lower_src) and ("5.0 second" in lower_resp or "5.0s" in lower_resp or "5 seconds" in lower_resp):
                if not ("3.0" in lower_resp or "3 second" in lower_resp):
                    has_contradiction = True
                    contradiction_detail = "Response asserts threshold of 5.0 seconds, directly contradicting the source document which states 3.0 seconds."
            elif ("increased by 8%" in lower_src or "increased by 18%" in lower_src or "revenue increased" in lower_src or "revenues increased" in lower_src) and ("decreased by" in lower_resp or "revenue decreased" in lower_resp or "revenues decreased" in lower_resp):
                has_contradiction = True
                contradiction_detail = "Response asserts revenue decreased, directly contradicting the source document which states revenue increased."
            elif ("decreased" in lower_src or "dropped" in lower_src) and ("increased by" in lower_resp or "expenses increased" in lower_resp):
                has_contradiction = True
                contradiction_detail = "Response asserts expenses/metrics increased, directly contradicting the source document which states they decreased/dropped."
            elif "increased" in lower_src and "decreased" in lower_resp and any(w in lower_src for w in ["revenue", "profit", "net income", "sales", "adoption"]):
                has_contradiction = True
                contradiction_detail = "Response claims financial/operational metrics decreased when the source states they increased."

            if has_contradiction:
                mock_data = EvaluationScore(
                    relevance=4,
                    completeness=6,
                    instruction_following=5,
                    format_compliance=8,
                    conciseness=8,
                    factual_consistency=3,
                    feedback=[
                        contradiction_detail,
                        "Factual consistency severely penalized due to direct contradiction of source facts.",
                    ],
                    strengths=["Response maintains clean formatting."],
                )
                return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=110.0)

            # ---------------------------------------------------------
            # 2b. Unsupported Claims / Invented Requirements (TEST C)
            # ---------------------------------------------------------
            has_unsupported_req = False
            unsupported_req_detail = ""

            unsupported_keywords = [
                "email alert", "email alerts", "email notification", "webhook", 
                "configurable threshold", "hardcoded threshold", "audit trail", 
                "50ms", "millisecond precision", "mandatory email", "must send email"
            ]
            for kw in unsupported_keywords:
                if kw in lower_resp and kw not in lower_src:
                    has_unsupported_req = True
                    unsupported_req_detail = f"Response introduces unsupported requirement: '{kw}' is asserted but is not present in the source document."
                    break

            if has_unsupported_req:
                mock_data = EvaluationScore(
                    relevance=7,
                    completeness=8,
                    instruction_following=6,
                    format_compliance=8,
                    conciseness=8,
                    factual_consistency=6,
                    feedback=[
                        unsupported_req_detail,
                        "Factual consistency penalized for introducing ungrounded mandatory requirements not in source data.",
                    ],
                    strengths=["Core source threshold is preserved."],
                )
                return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=110.0)

            # ---------------------------------------------------------
            # 3. Format Violation (TEST 5)
            # ---------------------------------------------------------
            bullet_lines = [l.strip() for l in gen_resp.split("\n") if re.match(r"^([•\-\*]|\d+\.)\s+", l.strip())]
            wants_bullets = any(w in lower_inst for w in ["bullet point", "bullet points", "bullets", "bulleted list"])
            wants_exact_3_bullets = "3 bullet points" in lower_inst or "exactly 3 bullets" in lower_inst or "3 bullets" in lower_inst
            wants_json = "json" in lower_inst

            if wants_bullets and len(bullet_lines) == 0:
                mock_data = EvaluationScore(
                    relevance=8,
                    completeness=8,
                    instruction_following=4,
                    format_compliance=3,
                    conciseness=7,
                    factual_consistency=9,
                    feedback=[
                        "Format violation: prompt explicitly required bullet points, but the response was provided as a continuous paragraph.",
                        "Instruction following and format compliance penalized for disregarding structural constraints.",
                    ],
                    strengths=["Response accurately captures factual content from the source data."],
                )
                return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=110.0)

            if wants_exact_3_bullets and len(bullet_lines) > 0 and len(bullet_lines) != 3:
                mock_data = EvaluationScore(
                    relevance=8,
                    completeness=8,
                    instruction_following=5,
                    format_compliance=4,
                    conciseness=7,
                    factual_consistency=9,
                    feedback=[
                        f"Instruction deviation: prompt required exactly 3 bullet points, but response provided {len(bullet_lines)}.",
                    ],
                    strengths=["Grounded in source data."],
                )
                return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=110.0)

            if wants_json and not ("{" in gen_resp and "}" in gen_resp):
                mock_data = EvaluationScore(
                    relevance=7,
                    completeness=7,
                    instruction_following=4,
                    format_compliance=2,
                    conciseness=7,
                    factual_consistency=9,
                    feedback=[
                        "Format violation: prompt explicitly required JSON format, but response was plain text.",
                    ],
                    strengths=["Source facts are mentioned."],
                )
                return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=110.0)

            # ---------------------------------------------------------
            # 4. Incomplete Response (TEST 2)
            # ---------------------------------------------------------
            has_5_numbered_facts = bool(re.search(r"fact\s*1\b.*fact\s*2\b.*fact\s*3\b.*fact\s*4\b.*fact\s*5\b", lower_src, re.DOTALL))
            if not has_5_numbered_facts:
                has_5_numbered_facts = bool(re.search(r"1\..*2\..*3\..*4\..*5\.", lower_src, re.DOTALL))

            if has_5_numbered_facts:
                facts_in_resp = 0
                for f_num in ["fact 1", "fact 2", "fact 3", "fact 4", "fact 5"]:
                    if f_num in lower_resp:
                        facts_in_resp += 1
                if facts_in_resp == 0:
                    fact_keywords = [
                        ["fact 1", "first fact", "alpha", "initial"],
                        ["fact 2", "second fact", "beta"],
                        ["fact 3", "third fact", "gamma"],
                        ["fact 4", "fourth fact", "delta"],
                        ["fact 5", "fifth fact", "epsilon"],
                    ]
                    for kw_group in fact_keywords:
                        if any(kw in lower_resp for kw in kw_group):
                            facts_in_resp += 1

                if facts_in_resp == 1 or ("fact 1" in lower_resp and not any(f in lower_resp for f in ["fact 2", "fact 3", "fact 4", "fact 5"])):
                    mock_data = EvaluationScore(
                        relevance=7,
                        completeness=3,
                        instruction_following=6,
                        format_compliance=8,
                        conciseness=8,
                        factual_consistency=9,
                        feedback=[
                            "Response is severely incomplete: covers only 1 of the 5 essential facts documented in the source text.",
                            "Critical source information omitted, reducing overall completeness.",
                        ],
                        strengths=["The single included fact is factually accurate and grounded."],
                    )
                    return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=110.0)

                if facts_in_resp >= 5 or all(f in lower_resp for f in ["fact 1", "fact 2", "fact 3", "fact 4", "fact 5"]):
                    mock_data = EvaluationScore(
                        relevance=9,
                        completeness=10,
                        instruction_following=10,
                        format_compliance=10,
                        conciseness=9,
                        factual_consistency=10,
                        feedback=[
                            "Exemplary response: covers all 5 key source facts with complete fidelity and strictly satisfies all instructions.",
                        ],
                        strengths=[
                            "100% factual accuracy against source data.",
                            "Full coverage of all critical source elements.",
                            "Flawless instruction adherence.",
                        ],
                    )
                    return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=110.0)

            # ---------------------------------------------------------
            # 5. Weak Prompt with Good Response (TEST 6)
            # ---------------------------------------------------------
            is_weak_prompt = (
                lower_inst in ["summarize this.", "summarize this", "tell me about this.", "tell me about this", "explain this.", "explain this"]
                or (len(lower_inst) < 25 and not any(w in lower_inst for w in ["bullet", "json", "constraint", "not", "limit", "exact"]))
            )

            if is_weak_prompt:
                mock_data = EvaluationScore(
                    relevance=8,
                    completeness=8,
                    instruction_following=8,
                    format_compliance=8,
                    conciseness=8,
                    factual_consistency=9,
                    feedback=[
                        "Response provides an accurate, well-grounded summary of the source data.",
                        "Solid response to an open-ended prompt; scoring reflects good quality without assuming unrequested constraints.",
                    ],
                    strengths=[
                        "Direct factual alignment with source text without hallucination.",
                        "Clear and coherent synthesis.",
                    ],
                )
                return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=110.0)

            # ---------------------------------------------------------
            # 6. Calibrated Differentiated Benchmark Scores
            # ---------------------------------------------------------
            feedback = []
            strengths = []

            if is_source_cve and is_resp_cve:
                feedback.append("Response captures CVE identifier, severity, and mitigation instructions from the advisory.")
                strengths.append("High factual precision with zero hallucination of unrelated packages.")
                score_fact = 10
                score_rel = 9
                score_comp = 9
                score_inst = 9 if wants_bullets or len(bullet_lines) >= 2 else 8
                score_fmt = 9
                score_conc = 8
            elif is_source_fin and is_resp_fin:
                feedback.append("Response extracts revenue, net income, and quarterly guidance figures from the disclosure.")
                strengths.append("Strict numerical grounding against financial press release.")
                score_fact = 10
                score_rel = 9
                score_comp = 9
                score_inst = 9
                score_fmt = 9
                score_conc = 9
            elif is_source_health and is_resp_health:
                feedback.append("Response synthesizes compliance timeline, Class II audit requirements, and Class III emergency triage rules.")
                strengths.append("Thorough coverage of regulatory framework without external claims.")
                score_fact = 10
                score_rel = 9
                score_comp = 8
                score_inst = 9
                score_fmt = 9
                score_conc = 9
            elif is_source_api and is_resp_api:
                feedback.append("Response conveys the REST v1 sunset date, GraphQL v2 migration, and documentation references.")
                strengths.append("Actionable and direct developer communication.")
                score_fact = 10
                score_rel = 9
                score_comp = 8
                score_inst = 9
                score_fmt = 9
                score_conc = 8
            elif is_source_alert and is_resp_alert:
                has_invented_specs = any(w in lower_resp for w in [
                    "webhook", "email", "configuration panel", "settings panel", "50ms", "audit trail", "millisecond"
                ])
                if has_invented_specs:
                    feedback.append(
                        "Response introduces unsupported technical specifications and requirements not present in the source document "
                        "(e.g., webhook/email notification channels, settings configuration panel, 50ms monitoring overhead, and audit trail)."
                    )
                    strengths.append("Translates core 3.0s latency threshold into formal acceptance criteria structure.")
                    score_fact = 6
                else:
                    feedback.append("Response specifies the 3.0 second latency alerting threshold and scope in PromptPilot.")
                    strengths.append("Structured acceptance criteria aligned with product requirement without ungrounded requirements.")
                    score_fact = 10
                score_rel = 9
                score_comp = 8
                score_inst = 9
                score_fmt = 9
                score_conc = 9
            elif is_source_mig and is_resp_mig:
                feedback.append("Response captures compute cost reduction, budget variance, and provisioned concurrency mitigation.")
                strengths.append("Faithful extraction of engineering post-mortem trade-offs.")
                score_fact = 10
                score_rel = 9
                score_comp = 8
                score_inst = 9
                score_fmt = 9
                score_conc = 8
            elif is_source_ev and is_resp_ev:
                feedback.append("Response summarizes electric vehicle adoption trends and infrastructure constraints.")
                strengths.append("Grounded summary adhering to provided automotive article.")
                score_fact = 9
                score_rel = 9
                score_comp = 8
                score_inst = 8
                score_fmt = 8
                score_conc = 8
            else:
                feedback.append("Response captures key takeaways from the provided source document.")
                strengths.append("Maintains factual consistency with source input.")
                # If exact numerical facts or stated source details match with zero contradiction
                if ("$500m" in lower_src and "$500m" in lower_resp) or ("3.0 second" in lower_src and "3.0" in lower_resp):
                    score_fact = 10
                    score_rel = 9
                    score_comp = 9
                else:
                    score_fact = 8
                    score_rel = 8
                    score_comp = 8
                score_inst = 8
                score_fmt = 8
                score_conc = 8

            mock_data = EvaluationScore(
                relevance=score_rel,
                completeness=score_comp,
                instruction_following=score_inst,
                format_compliance=score_fmt,
                conciseness=score_conc,
                factual_consistency=score_fact,
                feedback=feedback,
                strengths=strengths,
            )
            return mock_data, LLMResponse(content="{}", model_name="mock-demo-engine", latency_ms=110.0)

        raise ValueError(f"Unknown mock response schema: {response_schema}")
