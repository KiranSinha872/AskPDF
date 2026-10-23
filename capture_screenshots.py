"""
capture_screenshots.py — Open each of the 5 preset questions in a 1440x900 window,
click New chat between each, wait for the answer, and save screenshots/q1.png ... q5.png.
"""
import os
import time
from playwright.sync_api import sync_playwright

QUESTIONS = [
    "What are the three main types of inspection?",
    "What does a maintenance factor of two mean?",
    "What is the purchase price of this gas turbine?",
    "What is the melting point of titanium?",
    "What is the recommended combustion-inspection interval in hours?",
]

os.makedirs("screenshots", exist_ok=True)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})

        for idx, q in enumerate(QUESTIONS, 1):
            print(f"[{idx}/5] {q}")
            page = context.new_page()
            page.goto("http://127.0.0.1:8000", wait_until="networkidle")

            # Click the matching preset pill (text match)
            pill = page.locator(".sample-pill", has_text=q.strip())
            pill.click()

            # Wait for .answer-text or .answer-text.not-found to appear
            page.wait_for_selector(".answer-text", timeout=45000)
            page.wait_for_timeout(800)   # let any layout settle

            out = f"screenshots/q{idx}.png"
            page.screenshot(path=out, full_page=False)
            print(f"   Saved {out}")
            page.close()

        browser.close()
    print("All 5 screenshots saved.")


if __name__ == "__main__":
    main()
