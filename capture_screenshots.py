import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import time
from playwright.sync_api import sync_playwright

STREAMLIT_URL = "http://localhost:8501"
OUT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(OUT_DIR, exist_ok=True)

PAGES = [
    ("Dashboard",         "01_dashboard.png"),
    ("Single Prediction", "02_single_prediction.png"),
    ("Batch Prediction",  "03_batch_prediction.png"),
    ("EDA",               "04_eda_insights.png"),
    ("Model Performance", "05_model_performance.png"),
]

def wait_for_streamlit(page, timeout=30):
    try:
        page.wait_for_selector('[data-testid="stSpinner"]', timeout=3000)
        page.wait_for_selector('[data-testid="stSpinner"]', state="detached", timeout=timeout * 1000)
    except Exception:
        pass
    page.wait_for_timeout(2500)


def capture():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1.5,
        )
        page = context.new_page()

        print("Navigating to Streamlit app...")
        page.goto(STREAMLIT_URL, wait_until="networkidle", timeout=60000)
        wait_for_streamlit(page)
        page.wait_for_timeout(4000)

        for label, filename in PAGES:
            print("  Capturing page: " + label)
            try:
                sidebar = page.locator('[data-testid="stSidebar"]')
                radio = sidebar.get_by_text(label, exact=False).first
                radio.click()
                wait_for_streamlit(page)
                page.wait_for_timeout(3000)
            except Exception as e:
                print("    WARNING: Could not click '" + label + "': " + str(e))

            out_path = os.path.join(OUT_DIR, filename)
            page.screenshot(path=out_path, full_page=True)
            print("    Saved: " + out_path)

        browser.close()
        print("All screenshots captured.")


if __name__ == "__main__":
    capture()
