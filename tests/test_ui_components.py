"""Unit tests for UI components and exports."""

import unittest
from ui.components import (
    render_api_warning,
    render_empty_state,
    render_evaluation_disclaimer,
    render_horizontal_score_bar,
    render_metric_card,
    render_page_header,
    render_radar_or_bar_metrics,
    render_score_badge,
    render_status_badge,
)
from ui.analyzer_page import render_prompt_analyzer_page


class TestUIComponents(unittest.TestCase):
    """Test suite ensuring all UI components and page modules are cleanly exported and callable."""

    def test_component_exports(self):
        """Verify all critical UI components are exported without ImportError."""
        self.assertTrue(callable(render_radar_or_bar_metrics))
        self.assertTrue(callable(render_page_header))
        self.assertTrue(callable(render_api_warning))
        self.assertTrue(callable(render_prompt_analyzer_page))

    def test_render_score_badge(self):
        """Verify score badges generate proper CSS classes."""
        high_badge = render_score_badge(9.0)
        self.assertIn("badge-score-high", high_badge)
        self.assertIn("9.0 / 10.0", high_badge)

        med_badge = render_score_badge(6.5)
        self.assertIn("badge-score-med", med_badge)

        low_badge = render_score_badge(3.0)
        self.assertIn("badge-score-low", low_badge)

    def test_render_status_badge(self):
        """Verify status badge formatting."""
        badge = render_status_badge("Completed")
        self.assertIn("badge-status-completed", badge)
        self.assertIn("Completed", badge)

    def test_render_page_header_signature_compatibility(self):
        """Verify render_page_header accepts title, subtitle, icon, action_button, and extra kwargs."""
        import inspect
        sig = inspect.signature(render_page_header)
        params = list(sig.parameters.keys())
        self.assertIn("title", params)
        self.assertIn("subtitle", params)
        self.assertIn("icon", params)
        self.assertIn("action_button", params)


if __name__ == "__main__":
    unittest.main()
