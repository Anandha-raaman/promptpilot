"""Comprehensive security test suite for PromptPilot.

Verifies:
1. API keys and secrets are never rendered in the UI or prefilled into input fields.
2. The UI displays only approved status indicators ("API key configured", "Configured", "Not configured").
3. Secrets are excluded from string representations, logs, and exported experiment data.
4. Missing credentials produce clear, safe error messages without leaking internal data.
5. The application functions properly with configured providers and supports Streamlit secrets.
"""

import io
import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from config.settings import Settings, _get_streamlit_secret, get_settings
from database.models import (
    EvaluationRecord,
    ExperimentRecord,
    FullExperimentDetail,
    PromptVersionRecord,
    ResponseRecord,
)
from llm.base import LLMAuthError, LLMResponse
from llm.gemini_provider import GeminiProvider
from llm.mock_provider import MockLLMProvider
from ui.settings_page import render_settings_page
from utils.export import ExperimentExporter
from utils.logging import SensitiveDataSanitizingFilter, get_logger, redact_secrets


class TestSecurityExposure(unittest.TestCase):
    """Test suite ensuring credentials cannot be exposed through any surface."""

    def setUp(self):
        self.dummy_key = "AQ.SecurityTestSecretKey9876543210zyxwv"
        self.aiza_key = "AIzaSySecurityTestDummySecretKey12345"

    # --- 1. UI RENDERING & INPUT FIELD TESTS ---

    @patch("ui.settings_page.st")
    def test_api_key_never_rendered_in_settings_ui(self, mock_st):
        """Verify API key is never rendered into the DOM, markdown, or input value."""
        mock_cols = []

        def fake_columns(spec):
            count = spec if isinstance(spec, int) else len(spec)
            cols = [MagicMock() for _ in range(count)]
            mock_cols.extend(cols)
            return cols

        mock_st.columns.side_effect = fake_columns
        mock_st.button.return_value = False

        settings = Settings(gemini_api_key=self.dummy_key)
        provider = GeminiProvider(settings=settings)

        render_settings_page(provider=provider, settings=settings)

        # Verify st.text_input calls
        text_input_calls = mock_st.text_input.call_args_list
        self.assertTrue(len(text_input_calls) > 0, "st.text_input should be called for Gemini API Key")

        # Find the Gemini API Key input call
        key_input_call = None
        for call in text_input_calls:
            if "Gemini API Key" in call.args or call.kwargs.get("label") == "Gemini API Key":
                key_input_call = call
                break

        self.assertIsNotNone(key_input_call, "Gemini API Key input field must be present")
        _, kwargs = key_input_call

        # REQUIREMENT: Password type input
        self.assertEqual(kwargs.get("type"), "password", "API Key input must be type='password'")

        # REQUIREMENT: Never prefill with existing secret
        self.assertEqual(kwargs.get("value"), "", "API Key input value must NOT be prefilled with existing secret")
        self.assertNotIn(self.dummy_key, str(kwargs.get("value")), "Secret key found in input value!")

        # Verify no call to st.markdown or column.markdown contains the secret or partial key
        all_markdown_calls = list(mock_st.markdown.call_args_list)
        for col in mock_cols:
            all_markdown_calls.extend(col.markdown.call_args_list)

        for m_call in all_markdown_calls:
            rendered_text = " ".join(str(a) for a in m_call.args)
            self.assertNotIn(self.dummy_key, rendered_text, "Secret key exposed in markdown call!")
            self.assertNotIn(self.dummy_key[-6:], rendered_text, "Partial secret key exposed in markdown call!")

        # REQUIREMENT: If key exists, display only 'API key configured'
        success_calls = [str(c.args) for c in mock_st.success.call_args_list]
        self.assertTrue(
            any("API key configured" in text for text in success_calls),
            "Expected 'API key configured' notification when key is set",
        )

    @patch("ui.settings_page.st")
    def test_settings_page_metadata_only(self, mock_st):
        """Verify API configuration section displays only API status, Provider name, and Model name."""
        mock_cols = []

        def fake_columns(spec):
            count = spec if isinstance(spec, int) else len(spec)
            cols = [MagicMock() for _ in range(count)]
            mock_cols.extend(cols)
            return cols

        mock_st.columns.side_effect = fake_columns
        mock_st.button.return_value = False

        settings = Settings(gemini_api_key=self.dummy_key, llm_model="gemini-3.5-flash")
        provider = GeminiProvider(settings=settings)

        render_settings_page(provider=provider, settings=settings)

        # Content inside 'with col1:' is dispatched to st.markdown
        markdown_calls = [str(c.args[0]) for c in mock_st.markdown.call_args_list]
        all_text = " ".join(markdown_calls)

        self.assertIn("API status:", all_text)
        self.assertIn("Configured", all_text)
        self.assertIn("Provider name:", all_text)
        self.assertIn("Google Gemini", all_text)
        self.assertIn("Model name:", all_text)
        self.assertIn("gemini-3.5-flash", all_text)

        # No secret or partial secret in rendered markdown
        self.assertNotIn(self.dummy_key, all_text)
        self.assertNotIn(self.dummy_key[-4:], all_text)

    # --- 2. STRING REPRESENTATION & SECRETS MASKING ---

    def test_settings_repr_and_str_never_reveal_secrets(self):
        """Verify Settings __repr__ and __str__ do not expose keys or partial keys."""
        settings = Settings(gemini_api_key=self.dummy_key)
        repr_str = repr(settings)
        str_str = str(settings)

        self.assertNotIn(self.dummy_key, repr_str)
        self.assertNotIn(self.dummy_key, str_str)
        # Verify no partial key (last 4 characters) is revealed
        self.assertNotIn(self.dummy_key[-4:], repr_str)
        self.assertNotIn(self.dummy_key[-4:], str_str)
        self.assertIn("api_key_configured=True", repr_str)

    # --- 3. LOGGING SANITIZATION TESTS ---

    def test_logs_redact_api_keys(self):
        """Verify logging filter strips and redacts credentials from console and file logs."""
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.addFilter(SensitiveDataSanitizingFilter())
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("promptpilot.test_security")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Attempting connection with key %s", self.dummy_key)
        logger.info("Google key: %s and token: %s", self.aiza_key, "sk-1234567890abcdef1234567890")

        output = log_stream.getvalue()
        self.assertNotIn(self.dummy_key, output)
        self.assertNotIn(self.aiza_key, output)
        self.assertNotIn("sk-1234567890abcdef1234567890", output)
        self.assertIn("[REDACTED_SECRET]", output)

    # --- 4. EXPORT SANITIZATION TESTS ---

    def test_exports_redact_secrets(self):
        """Verify exports redact accidental secrets in prompts, inputs, or model responses."""
        exp = ExperimentRecord(
            id="exp-sec-1",
            task_type="Extraction",
            original_prompt=f"Extract data using authorization: {self.dummy_key}",
            input_text=f"Customer record with token: {self.aiza_key}",
            model_name="test-model",
            created_at="2026-10-01T12:00:00Z",
            title="Security Audit",
        )
        resp = ResponseRecord(
            id="resp-sec-1",
            prompt_version_id="ver-sec-1",
            response_text=f"Model echoed back secret {self.dummy_key} in output.",
            latency_ms=120.0,
        )
        ver = PromptVersionRecord(
            id="ver-sec-1",
            experiment_id="exp-sec-1",
            version_number=0,
            strategy="Baseline",
            prompt_text=f"Prompt with key {self.dummy_key}",
            response=resp,
        )
        detail = FullExperimentDetail(experiment=exp, versions=[ver])

        # Test JSON Export
        json_out = ExperimentExporter.to_json(detail)
        self.assertNotIn(self.dummy_key, json_out)
        self.assertNotIn(self.aiza_key, json_out)
        self.assertIn("[REDACTED_SECRET]", json_out)

        # Test CSV Export
        csv_out = ExperimentExporter.to_csv(detail)
        self.assertNotIn(self.dummy_key, csv_out)
        self.assertNotIn(self.aiza_key, csv_out)

        # Test Markdown Export
        md_out = ExperimentExporter.to_markdown(detail)
        self.assertNotIn(self.dummy_key, md_out)
        self.assertNotIn(self.aiza_key, md_out)

    # --- 5. MISSING CREDENTIALS & SAFE ERROR MESSAGES ---

    def test_missing_credentials_safe_error_message(self):
        """Verify unconfigured provider raises a safe exception without internal leak."""
        empty_settings = Settings(gemini_api_key=None, llm_api_key_fallback=None)
        provider = GeminiProvider(settings=empty_settings, api_key=None)

        self.assertFalse(provider.is_available())
        with self.assertRaises(LLMAuthError) as ctx:
            provider._get_client()

        err_msg = str(ctx.exception)
        self.assertIn("Gemini API Key is not configured", err_msg)
        self.assertNotIn("password", err_msg.lower())
        self.assertNotIn("token", err_msg.lower())

    def test_api_error_mapping_redacts_credentials(self):
        """Verify that provider error mapping strips any API key returned by upstream."""
        provider = GeminiProvider(settings=Settings(gemini_api_key=self.dummy_key))
        leaked_error = Exception(f"HTTP 401 Unauthorized for request with key {self.dummy_key}")

        mapped_err = provider._map_api_error(leaked_error, "gemini-3.5-flash")
        err_str = str(mapped_err)

        self.assertNotIn(self.dummy_key, err_str)
        self.assertIn("[API_KEY_REDACTED]", err_str)

    # --- 6. PROVIDER CONNECTIVITY & STREAMLIT SECRETS SUPPORT ---

    def test_mock_provider_runs_normally(self):
        """Verify Offline Mock Provider operates seamlessly without any API keys."""
        mock_p = MockLLMProvider()
        self.assertTrue(mock_p.is_available())
        resp = mock_p.generate_text("Test prompt")
        self.assertIsInstance(resp, LLMResponse)
        self.assertTrue(len(resp.content) > 0)

    def test_streamlit_secrets_support(self):
        """Verify Settings can load credentials from Streamlit secrets dictionary."""
        mock_secrets = {"GEMINI_API_KEY": "AQ.StreamlitSecretLoadedKey12345"}
        with patch("config.settings._get_streamlit_secret", side_effect=lambda k: mock_secrets.get(k)):
            settings = Settings(gemini_api_key=None, llm_api_key_fallback=None)
            self.assertEqual(settings.effective_api_key, "AQ.StreamlitSecretLoadedKey12345")
            self.assertTrue(settings.has_valid_api_key)

    # --- 7. SAFE ERROR HANDLING TESTS ---

    def test_safe_error_handling_sanitizes_paths_and_traces(self):
        """Verify internal filesystem paths, stack traces, and database strings are sanitized from user view."""
        from utils.logging import format_safe_user_error

        # Path leak
        path_err = Exception("FileNotFoundError: [Errno 2] No such file or directory: 'F:\\My projects\\promptpilot\\data\\secret.db'")
        safe_msg = format_safe_user_error(path_err, "The experiment could not be completed. Please try again.")
        self.assertNotIn("F:\\My projects", safe_msg)
        self.assertNotIn("secret.db", safe_msg)
        self.assertEqual(safe_msg, "The experiment could not be completed. Please try again.")

        # Stack trace leak
        trace_err = Exception('Traceback (most recent call last):\n  File "C:\\Users\\User\\app.py", line 42, in <module>\nRuntimeError')
        safe_msg2 = format_safe_user_error(trace_err, "Unable to complete the AI request. Please check your API configuration and try again.")
        self.assertNotIn("Traceback", safe_msg2)
        self.assertNotIn("C:\\Users\\User", safe_msg2)
        self.assertEqual(safe_msg2, "Unable to complete the AI request. Please check your API configuration and try again.")

        # Database internal leak
        db_err = Exception("sqlite3.OperationalError: no such column: secret_key")
        safe_msg3 = format_safe_user_error(db_err)
        self.assertNotIn("sqlite3", safe_msg3)
        self.assertNotIn("secret_key", safe_msg3)
        self.assertIn("A database operation could not be completed", safe_msg3)

        # API Auth / Credential error
        auth_err = Exception(f"403 Forbidden: Invalid API Key {self.dummy_key}")
        safe_msg4 = format_safe_user_error(auth_err)
        self.assertNotIn(self.dummy_key, safe_msg4)
        self.assertEqual(safe_msg4, "Unable to complete the AI request. Please check your API configuration and try again.")


if __name__ == "__main__":
    unittest.main()

