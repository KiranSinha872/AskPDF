"""
capture_screenshots.py — Automate browser to ask each of the 5 questions
and save screenshots/q1.png ... screenshots/q5.png.
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
        context = browser.new_context(viewport={"width": 960, "height": 720})
        
        for idx, q in enumerate(QUESTIONS, 1):
            print(f"[{idx}/5] Opening page for: {q}")
            page = context.new_page()
            page.goto("http://127.0.0.1:8000")
            page.wait_for_selector(".preset-btn")
            
            # Click the preset button
            buttons = page.query_selector_all(".preset-btn")
            buttons[idx - 1].click()
            
            # Wait for result card
            page.wait_for_selector(".result-card", timeout=30000)
            page.wait_for_timeout(1000)
            
            out_path = f"screenshots/q{idx}.png"
            page.screenshot(path=out_path)
            print(f"  Saved screenshot to {out_path}")
            page.close()
            
        browser.close()
    print("All screenshots successfully captured!")

if __name__ == "__main__":
    main()
