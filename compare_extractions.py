import os
import pymupdf
from pypdf import PdfReader

PDF_PATH = "data/Binary-Trees-Complete-Course.pdf"
PAGE_INDEX = 54  # 0-indexed page 54 = page 55

def extract_a():
    doc = pymupdf.open(PDF_PATH)
    page = doc[PAGE_INDEX]
    return page.get_text()

def extract_b():
    doc = pymupdf.open(PDF_PATH)
    page = doc[PAGE_INDEX]
    d = page.get_text("dict")

    # Collect text segments with bounding boxes and font info
    raw_lines = []
    mono_char_widths = []

    for b in d.get("blocks", []):
        if "lines" not in b:
            continue
        for line in b["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            txt = "".join(s["text"] for s in line["spans"]).rstrip("\n")
            if not txt.strip():
                continue
            
            # Check for monospace spans to calculate char width dynamically
            for s in line["spans"]:
                font_name = s.get("font", "").lower()
                text_len = len(s.get("text", ""))
                span_w = s.get("bbox", [0, 0, 0, 0])[2] - s.get("bbox", [0, 0, 0, 0])[0]
                if ("mono" in font_name or "courier" in font_name or "typewriter" in font_name) and text_len >= 5:
                    mono_char_widths.append(span_w / text_len)

            raw_lines.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "text": txt,
                "spans": line["spans"],
            })

    # Fallback monospace char width if none detected
    char_w = sum(mono_char_widths) / len(mono_char_widths) if mono_char_widths else 5.23

    # Group lines by vertical proximity (within 3.5 points)
    raw_lines.sort(key=lambda item: (round(item["y0"] / 4) * 4, item["x0"]))
    visual_lines = []
    for item in raw_lines:
        if not visual_lines:
            visual_lines.append([item])
        else:
            last_group = visual_lines[-1]
            if abs(item["y0"] - last_group[0]["y0"]) <= 3.5:
                last_group.append(item)
            else:
                visual_lines.append([item])

    # Find minimum code x0 (typically ~74.8 on code lines)
    code_x0s = [
        it["x0"] for group in visual_lines for it in group
        if any("mono" in s.get("font", "").lower() for s in it.get("spans", []))
    ]
    base_x0 = min(code_x0s) if code_x0s else 74.8

    rebuilt = []
    for group in visual_lines:
        group.sort(key=lambda it: it["x0"])
        line_str = ""
        is_code_line = any("mono" in s.get("font", "").lower() for it in group for s in it.get("spans", []))
        
        for idx, it in enumerate(group):
            if is_code_line:
                col = max(0, int(round((it["x0"] - base_x0) / char_w)))
            else:
                col = 0

            if idx == 0:
                line_str = (" " * col) + it["text"]
            else:
                needed = max(1, col - len(line_str))
                line_str += (" " * needed) + it["text"]
        rebuilt.append(line_str)

    return "\n".join(rebuilt)

def extract_c():
    reader = PdfReader(PDF_PATH)
    page = reader.pages[PAGE_INDEX]
    return page.extract_text(extraction_mode="layout")

if __name__ == "__main__":
    print("================================================================================")
    print(" METHOD (a): PyMuPDFLoader default get_text()")
    print("================================================================================")
    text_a = extract_a()
    # Print lines containing the solution
    lines_a = [l for l in text_a.splitlines() if any(k in l for k in ["class Solution", "best", "height", "return", "//", "};"])]
    print("\n".join(lines_a[:18]))

    print("\n================================================================================")
    print(" METHOD (b): PyMuPDF get_text('dict') rebuilt with x-origin converted to spaces")
    print("================================================================================")
    text_b = extract_b()
    lines_b = [l for l in text_b.splitlines() if any(k in l for k in ["class Solution", "best", "height", "return", "//", "};"])]
    print("\n".join(lines_b[:18]))

    print("\n================================================================================")
    print(" METHOD (c): PyPDFLoader(extraction_mode='layout')")
    print("================================================================================")
    text_c = extract_c()
    lines_c = [l for l in text_c.splitlines() if any(k in l for k in ["Solution", "best", "height", "return", "//", "};"])]
    print("\n".join(lines_c[:18]))
