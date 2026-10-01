"""UI flow automated tests for Test Datasets and Test Case Management."""

import unittest
from streamlit.testing.v1 import AppTest


class TestBatchPageUIFlow(unittest.TestCase):
    """Test Streamlit UI flow for datasets, test cases, and execution."""

    def test_app_loads_and_batch_page_renders(self):
        import os
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        self.assertFalse(at.exception)

        # Navigate to Test Datasets page
        nav_radio = at.sidebar.radio[0]
        at.sidebar.radio[0].set_value("Test Datasets")
        at.run()
        self.assertFalse(at.exception)

        # Verify dataset dropdown exists
        self.assertTrue(len(at.selectbox) >= 1)

        # Verify "+ Add Test Case" button exists
        add_btn = None
        for b in at.button:
            if "+ Add Test Case" in b.label:
                add_btn = b
                break
        self.assertIsNotNone(add_btn, "Could not find '+ Add Test Case' button")

        # Verify "Run All Tests" button exists
        run_all_btn = None
        for b in at.button:
            if "Run All Tests" in b.label:
                run_all_btn = b
                break
        self.assertIsNotNone(run_all_btn, "Could not find 'Run All Tests' button")

        # Click "+ Add Test Case"
        add_btn.click()
        at.run()
        self.assertFalse(at.exception)

        # Verify editor fields appear
        # Test Case ID, Name, Prompt, Source Data, Criteria
        labels = [ti.label for ti in at.text_input]
        self.assertTrue(any("Test Case ID" in l for l in labels))
        self.assertTrue(any("Test Case Name" in l for l in labels))

        area_labels = [ta.label for ta in at.text_area]
        self.assertTrue(any("Prompt / Instructions" in l for l in area_labels))
        self.assertTrue(any("Source Data / Input" in l for l in area_labels))
        self.assertTrue(any("Evaluation Criteria" in l for l in area_labels))

    def test_save_and_run_test_case_ui_flow(self):
        import os
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()

        # Set mock mode in session state to avoid external API calls during UI test
        at.session_state["use_mock_mode"] = True
        at.sidebar.radio[0].set_value("Test Datasets")
        at.run()
        self.assertFalse(at.exception)

        # Open editor
        for b in at.button:
            if "+ Add Test Case" in b.label:
                b.click()
                break
        at.run()
        self.assertFalse(at.exception)

        # Fill in test case fields
        for ti in at.text_input:
            if "Test Case Name" in ti.label:
                ti.set_value("UI Automated Test Case")
                break

        for ta in at.text_area:
            if "Prompt / Instructions" in ta.label:
                ta.set_value("Summarize the key metrics from the input.")
            elif "Source Data / Input" in ta.label:
                ta.set_value("Revenue grew 25% to $500M in Q3 2026.")
            elif "Evaluation Criteria" in ta.label:
                ta.set_value("Must mention 25% growth and $500M revenue.")

        # Click Save Test Case
        for b in at.button:
            if "Save Test Case" in b.label:
                b.click()
                break
        at.run()
        self.assertFalse(at.exception)

        # Verify test case is now visible in the list
        self.assertTrue(any("UI Automated Test Case" in exp.label for exp in at.expander))

        # Clean up created test case from repo to prevent benchmark pollution
        repo = at.session_state.get("repo")
        if repo:
            cases = repo.get_test_cases(dataset_id="default-benchmark")
            for c in cases:
                if c.name == "UI Automated Test Case":
                    repo.delete_test_case(c.id)


if __name__ == "__main__":
    unittest.main()
