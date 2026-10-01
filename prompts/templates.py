"""Prompt Library template repository for PromptPilot.

Contains curated prompt engineering starter templates for:
- Summarization
- Information Extraction
- Text Generation
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PromptTemplate:
    """A pre-built prompt template for initiating experiments."""
    id: str
    title: str
    task_type: str
    description: str
    prompt_text: str
    sample_input: str
    desired_format: Optional[str] = None
    recommended_criteria: Optional[str] = None


PROMPT_TEMPLATES: List[PromptTemplate] = [
    # --- Summarization Templates ---
    PromptTemplate(
        id="sum-exec",
        title="Executive Briefing Summary",
        task_type="Summarization",
        description="Produces a concise high-level executive briefing highlighting decisions, risks, and next steps.",
        prompt_text=(
            "You are a Chief of Staff reviewing this briefing for C-suite leadership.\n"
            "Summarize the provided document into:\n"
            "1. Key Decision / Core Announcement (1-2 sentences)\n"
            "2. Strategic Impact & Implications (3 bullet points)\n"
            "3. Risks & Next Steps\n\n"
            "Constraints: Maximum 200 words. Maintain an objective, professional tone."
        ),
        sample_input=(
            "Acme Corp announced a shift toward cloud-native microservices architecture during Q3. "
            "The engineering initiative will modernize legacy ERP systems over 18 months, requiring an estimated "
            "$4.2M capital expenditure. While expected to yield a 35% reduction in compute overhead and improve "
            "uptime from 99.2% to 99.95%, the transition faces talent gaps in Kubernetes orchestration and potential "
            "temporary downtime during database migrations in Q4. The leadership committee has approved the pilot "
            "phase for the payments pipeline commencing next month."
        ),
        desired_format="Structured Markdown with bold headers",
        recommended_criteria="Focus on decision clarity, risk identification, and brevity.",
    ),
    PromptTemplate(
        id="sum-bullet",
        title="Key Takeaways & Action Items",
        task_type="Summarization",
        description="Extracts bulleted core findings and actionable next steps without narrative fluff.",
        prompt_text=(
            "Extract the 5 most critical takeaways from the provided source data.\n"
            "Format as bullet points, each starting with an action verb or bold keyword.\n"
            "Exclude background history and keep each bullet under 25 words."
        ),
        sample_input=(
            "During the incident post-mortem meeting for the outage on September 14th, the site reliability engineering "
            "team determined that a misconfigured ingress controller caused cascading connection timeouts. Recovery "
            "took 42 minutes because automated rollback scripts had failed silently. Action items agreed upon: 1) Implement "
            "health checks on rollback routines by Oct 5, 2) Restrict ingress modification permissions to senior DevOps engineers, "
            "3) Update status page communication protocols within two weeks."
        ),
        desired_format="Bulleted list",
        recommended_criteria="Conciseness, actionable wording, omission of non-critical details.",
    ),

    # --- Information Extraction Templates ---
    PromptTemplate(
        id="extract-json",
        title="Structured Entity & Metric Extraction",
        task_type="Information Extraction",
        description="Extracts names, dates, financial amounts, and organizations into strict JSON format.",
        prompt_text=(
            "Extract all entities, financial metrics, and dates mentioned in the provided source data.\n"
            "Output must be a valid JSON object matching this schema:\n"
            "{\n"
            '  "organizations": string[],\n'
            '  "financial_figures": [{"metric": string, "amount": string, "period": string}],\n'
            '  "dates": string[],\n'
            '  "key_findings": string[]\n'
            "}\n"
            "Constraint: If any field is not found in the text, return an empty list. Do not extrapolate."
        ),
        sample_input=(
            "BioHealth Technologies reported its Q2 2026 financial results on August 12. Revenue reached $18.4 million, "
            "representing a 22% year-over-year increase compared to Q2 2025. Net profit was recorded at $3.1 million. "
            "CEO Sarah Lin announced a partnership with Zenith Pharmaceuticals to co-develop immunology therapeutics. "
            "Clinical trial enrollment for candidate BHT-401 begins in November 2026 across 12 medical centers."
        ),
        desired_format="Strict JSON",
        recommended_criteria="Schema validity, zero hallucination, extraction completeness.",
    ),
    PromptTemplate(
        id="extract-specs",
        title="Technical Specification Parser",
        task_type="Information Extraction",
        description="Pulls technical parameters, system requirements, and hardware limits from documentation.",
        prompt_text=(
            "Analyze the provided source data and extract all technical specifications, constraints, and dependencies.\n"
            "Format as a Markdown table with columns: [Parameter, Specification / Value, Required / Optional]."
        ),
        sample_input=(
            "PromptPilot requires Python 3.11 or newer and at least 2GB of free RAM. It requires SQLite 3.35+ for "
            "foreign key and JSON support. For production deployments, an optional Redis cache can be configured for "
            "distributed rate limiting. Maximum network latency to the LLM API endpoint should not exceed 1500ms."
        ),
        desired_format="Markdown table",
        recommended_criteria="Accuracy of extracted values, proper column mapping.",
    ),

    # --- Text Generation Templates ---
    PromptTemplate(
        id="gen-customer-reply",
        title="Professional Customer Resolution Response",
        task_type="Text Generation",
        description="Drafts an empathetic, constructive resolution to a customer support ticket or inquiry.",
        prompt_text=(
            "You are a Senior Customer Experience Specialist.\n"
            "Draft a courteous, empathetic, and solution-focused email response to the customer feedback.\n"
            "Include:\n"
            "1. Empathetic acknowledgment of the inconvenience\n"
            "2. Transparent explanation of the resolution steps being taken\n"
            "3. Clear timeline and direct contact information for follow-up\n"
            "Tone: Warm, highly professional, reassuring."
        ),
        sample_input=(
            "Customer Ticket #9482: I was billed twice for my subscription this month ($49 x 2 = $98) on September 28. "
            "I contacted support yesterday and have received no response. I need the duplicate charge refunded immediately."
        ),
        desired_format="Email message with subject line and sign-off",
        recommended_criteria="Empathy, solution clarity, professional de-escalation tone.",
    ),
    PromptTemplate(
        id="gen-user-story",
        title="Agile User Story & Acceptance Criteria",
        task_type="Text Generation",
        description="Converts product feature requests into clear User Stories with Given-When-Then acceptance criteria.",
        prompt_text=(
            "Convert the product requirement into a standard Agile User Story.\n"
            "Format:\n"
            "1. User Story: 'As a [persona], I want [action], so that [business value]'\n"
            "2. Acceptance Criteria (minimum 3 criteria formatted as Given-When-Then)\n"
            "3. Edge Cases & Error States to handle."
        ),
        sample_input=(
            "Users want to export their prompt testing experiment results into a clean PDF or CSV so they can share "
            "the findings with their engineering leads during sprint planning meetings."
        ),
        desired_format="Agile User Story format",
        recommended_criteria="Format compliance, clarity of acceptance criteria, coverage of edge cases.",
    ),
]


def get_all_templates() -> List[PromptTemplate]:
    """Return all available prompt templates."""
    return PROMPT_TEMPLATES


def get_template_by_id(template_id: str) -> Optional[PromptTemplate]:
    """Retrieve template by unique ID."""
    for tmpl in PROMPT_TEMPLATES:
        if tmpl.id == template_id:
            return tmpl
    return None
