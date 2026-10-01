# 🧭 PromptPilot — AI Prompt Testing & Optimization Workbench

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Google GenAI SDK](https://img.shields.io/badge/Google%20GenAI-2.0+-green.svg)](https://ai.google.dev/)
[![SQLite](https://img.shields.io/badge/SQLite-3.35+-lightgrey.svg)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**PromptPilot** is a professional Generative AI developer workbench built to systematically analyze, optimize, execute, evaluate, and regression-test prompts for Large Language Model (LLM) applications.

---

## 1. Problem Statement

In LLM application engineering, developers and prompt engineers frequently write prompts by trial and error:
- They make subtle phrasing changes without objective baselines.
- They manually run prompts against single examples and visually eyeball responses.
- They have no reliable way to verify whether a prompt modification actually improved quality or introduced silent performance regressions.
- Generic chat interfaces do not isolate instructions from data, lack structured evaluation rubrics, and fail to track prompt version histories.

**PromptPilot solves this problem** by automating the entire prompt optimization lifecycle: from prompt health auditing and strategy-driven variant generation, to multi-prompt execution against ground truth data, multi-criteria model-based evaluation, iterative playground tuning, and cross-dataset regression detection.

---

## 2. Core Features

- **🩺 Prompt Health Analyzer**: Evaluates prompts across 5 key dimensions: Objective Clarity, Context, Negative Constraints, Output Format, and Task Specificity (0–10 scale). Flags ambiguities and suggests concrete improvements.
- **⚡ 5-Strategy Prompt Variant Generator**: Generates optimized variants applying distinct prompt engineering techniques:
  1. *Role-Based Prompting*
  2. *Structured / Delimited Prompting*
  3. *Constraint-Based Prompting*
  4. *Few-Shot Exemplar Prompting*
  5. *Structured-Output (Schema) Prompting*
- **⚖️ Side-by-Side Execution & Evaluation**: Executes the baseline prompt and all variants against identical input context. Evaluates outputs across 6 rubric criteria: Relevance, Completeness, Instruction Following, Format Compliance, Conciseness, and Factual Consistency.
- **🛠️ Prompt Playground & Regression Lab**: Edit any prompt in a live interactive sandbox, generate versioned iterations (V1 $\rightarrow$ V2 $\rightarrow$ V3), and automatically detect performance regressions.
- **🧪 Batch Evaluation & Dataset Testing**: Run prompts against synthetic multi-task test suites (`data/test_cases.json`) to compute mean scores and verify stability across diverse inputs.
- **📜 SQLite Experiment History**: Relational persistence of all experiments, prompt versions, responses, token usage, and evaluation scores with sub-millisecond retrieval.
- **📚 Curated Prompt Library**: Ready-to-run templates for Summarization, Information Extraction, and Text Generation.
- **📥 Sanitized Multi-Format Export**: Export comprehensive audit reports in JSON, CSV, and Markdown with automatic credential masking.

---

## 3. Architecture

PromptPilot uses a modular, layered architecture adhering to the Separation of Concerns (SoC) principle:

```text
Streamlit UI Layer (app.py, ui/)
       │
       ▼
Domain & Orchestration Services (services/)
       │
       ├──► Prompt Analyzer & Generator (prompts/)
       ├──► Grounded Evaluator & Regression Detector (evaluation/)
       ├──► Parameterized SQLite Repository (database/)
       └──► Provider Abstraction (llm/base.py ──► llm/gemini_provider.py)
```

See [docs/architecture.md](docs/architecture.md) for detailed architectural diagrams, data flows, and sequence specifications.

---

## 4. Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.11+ / 3.12 / 3.13 / 3.14 |
| **User Interface** | Streamlit |
| **Data Validation** | Pydantic v2 & Pydantic-Settings |
| **LLM Provider** | Google Gemini API via official `google-genai` SDK |
| **Database** | SQLite 3 with Foreign Keys & WAL journaling |
| **Data Manipulation** | Pandas |
| **Testing** | Python `unittest` suite (22+ automated tests) |

---

## 5. Installation & Setup

### Prerequisites
- Python 3.11 or higher
- A Google Gemini API Key (free tier available at [Google AI Studio](https://aistudio.google.com/))

### Step-by-Step Installation

```bash
# 1. Clone or navigate to the repository
cd promptpilot

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
```

---

## 6. Environment Variables

Configure your `.env` file:

```env
# Gemini API Key (from Google AI Studio)
GEMINI_API_KEY=your_gemini_api_key_here

# Active model identifier
LLM_MODEL=gemini-2.5-flash

# Optional: Specialized model roles (defaults to LLM_MODEL if unset)
PROMPT_ANALYZER_MODEL=gemini-2.5-flash
PROMPT_GENERATOR_MODEL=gemini-2.5-flash
RESPONSE_GENERATOR_MODEL=gemini-2.5-flash
EVALUATOR_MODEL=gemini-2.5-flash

# Database path
DATABASE_PATH=data/promptpilot.db

# Cost & Safety Limits
MAX_INPUT_CHARS=10000
MAX_PROMPT_CHARS=3000
MAX_VARIANTS_PER_RUN=5
MAX_BATCH_TEST_CASES=10
RATE_LIMIT_COOLDOWN_SECONDS=2.0
LOG_LEVEL=INFO
```

---

## 7. Running Locally

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 8. Usage Walkthrough

1. **Dashboard**: Inspect overall experiment statistics, API connectivity, and recent runs.
2. **New Experiment**:
   - Select task category (e.g. *Summarization*).
   - Enter your original prompt and ground source text.
   - Click **Run Full Experiment** to trigger prompt health analysis, variant generation, execution, and evaluation.
3. **Inspect Comparison**:
   - Compare composite scores and rubric dimensions across all prompt variants.
   - Review qualitative feedback and strengths.
4. **Playground**: Click **Open in Playground to Tune** on any variant to make manual edits and test for regressions.
5. **Batch Testing**: Run prompts against the synthetic test suite to evaluate consistency.
6. **Export**: Download audit reports as clean JSON, CSV, or Markdown.

---

## 9. Evaluation Methodology

PromptPilot evaluates completions using a multi-criteria model-based rubric scored from 0 to 10:

- **Instruction Following (25%)**: Adherence to explicit negative constraints, tone, and directives.
- **Relevance (20%)**: Focus on the prompt's intent without wandering or conversational filler.
- **Factual Consistency (20%)**: Faithfulness to the provided ground context without hallucination.
- **Completeness (15%)**: Coverage of essential requirements and key details.
- **Format Compliance (10%)**: Conformance with requested structures (JSON, tables, bullet points).
- **Conciseness (10%)**: Density of information and economy of expression.

> [!IMPORTANT]
> **Evaluation Disclaimer**: Model-based evaluation is experimental and should be treated as a directional engineering metric rather than an absolute ground truth. Always combine automated metrics with human verification on high-stakes tasks.

---

## 10. Security & Safety Considerations

- **API Key Protection**: Credentials are read strictly from environment variables or active session memory. They are masked in logs, string representations, and exports.
- **Prompt Injection Defense**: Untrusted source data is strictly fenced in `<SOURCE_DATA>` XML blocks, and the evaluator is explicitly instructed that source documents are data, not instructions.
- **Input Boundaries**: Configurable length ceilings (`MAX_INPUT_CHARS`, `MAX_PROMPT_CHARS`) and null-byte defenses prevent denial-of-service and buffer abuse.
- **SQL Security**: All database interactions utilize parameterized queries (`?` placeholders). User input is never concatenated into raw SQL strings.
- **Cost Transparency**: Estimated API calls are computed and displayed before multi-variant or batch experiments execute.

---

## 11. Automated Test Suite

PromptPilot includes a comprehensive automated test suite with deterministic mocks (zero external API calls required during automated testing):

```bash
# Run all unit and integration tests
python -m unittest discover -s tests
```

### Test Coverage Highlights
- `test_validation.py`: Input length enforcement, null-byte rejection, and prompt injection signal detection.
- `test_database.py`: Relational schema integrity, foreign key cascading delete, and hierarchical query fetching.
- `test_scoring.py`: Pydantic schema validation and weighted rubric scoring calculations.
- `test_regression.py`: Delta computations and regression vs. improvement classification.
- `test_services.py`: End-to-end mocked pipeline execution for experiments and playground iterations.
- `test_batch_evaluation.py`: Multi-case dataset testing and batch regression reporting.
- `test_export.py`: Sanitization and structural verification of JSON, CSV, and Markdown exports.

---

## 12. Project Structure

```text
promptpilot/
├── app.py                      # Main Streamlit application & router
├── requirements.txt            # Pinned dependencies
├── .env.example                # Environment configuration template
├── .gitignore                  # Git secrets and artifact rules
├── README.md                   # Project documentation
│
├── config/
│   └── settings.py             # Pydantic Settings with validation & masking
│
├── llm/
│   ├── base.py                 # Abstract BaseLLMProvider interface
│   └── gemini_provider.py      # Google Gemini provider (google-genai SDK)
│
├── prompts/
│   ├── analyzer.py             # Prompt quality and weakness auditor
│   ├── generator.py            # 5-strategy prompt variant generator
│   ├── evaluator.py            # Grounded response evaluator
│   └── templates.py            # Reusable prompt library
│
├── evaluation/
│   ├── scoring.py              # Rubric definitions and composite scoring
│   └── regression.py           # Delta calculations and regression detection
│
├── database/
│   ├── db.py                   # SQLite connection & schema initialization
│   ├── models.py               # Data transfer objects
│   └── repository.py           # Parameterized CRUD operations
│
├── services/
│   ├── experiment_service.py   # Primary pipeline orchestrator
│   └── evaluation_service.py   # Batch evaluation and regression service
│
├── ui/
│   ├── components.py           # Reusable badges, CSS, and disclaimer banners
│   ├── dashboard.py            # Home overview & stats
│   ├── experiment_page.py      # New Experiment flow & comparison view
│   ├── analyzer_page.py        # Standalone prompt auditor
│   ├── playground_page.py      # Interactive prompt tuning lab
│   ├── batch_page.py           # Batch testing across datasets
│   ├── history_page.py         # SQLite history browser
│   ├── library_page.py         # Starter prompt template catalog
│   └── settings_page.py        # Configuration & API diagnostics
│
├── utils/
│   ├── validation.py           # Input constraints & injection defense
│   ├── logging.py              # Sanitized structured logging
│   └── export.py               # JSON, CSV, Markdown report exporters
│
├── data/
│   └── test_cases.json         # Synthetic test suite for batch testing
│
├── tests/                      # 22+ automated unit & integration tests
└── docs/
    └── architecture.md         # Architecture specifications & Mermaid diagrams
```

---

## 13. Limitations & Roadmap

### Current Limitations
- Supports text-based prompts and documents (multimodal file parsing planned).
- Initial provider implementation is Google Gemini (provider protocol is ready for OpenAI/Anthropic/Ollama).
- Evaluator scores depend on model inference and prompt phrasing.

### Future Roadmap
- [ ] Add OpenAI and Anthropic provider implementations via `BaseLLMProvider`.
- [ ] Multi-turn conversational prompt testing.
- [ ] Semantic clustering of response variations.
- [ ] PDF, Word, and CSV document upload parsers.
- [ ] CI/CD CLI runner for automated prompt regression testing in GitHub Actions.

---

## 14. License

Distributed under the MIT License. See `LICENSE` for details.
