import os
import sys
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

print("=== 1. Health check ===")
res = client.get("/health")
assert res.status_code == 200, f"Health check failed: {res.text}"
print("Health check OK:", res.json())

print("\n=== 2. Uploading data/GER-3620-P.pdf ===")
with open("data/GER-3620-P.pdf", "rb") as f:
    upload_res = client.post(
        "/upload",
        files={"file": ("GER-3620-P.pdf", f, "application/pdf")}
    )

assert upload_res.status_code == 200, f"Upload failed: {upload_res.status_code} {upload_res.text}"
upload_data = upload_res.json()
print("Upload result:", upload_data)
doc_id_1 = upload_data["doc_id"]
assert upload_data["pages"] == 60, f"Expected 60 pages, got {upload_data['pages']}"
assert upload_data["chunks"] > 0, "No chunks generated"

print(f"\n=== 3. Querying Q1 on doc_id {doc_id_1} ===")
q1 = "What are the three main types of inspection?"
ask_res1 = client.post("/ask", json={"question": q1, "doc_id": doc_id_1})
assert ask_res1.status_code == 200, f"Ask failed: {ask_res1.text}"
q1_data = ask_res1.json()
print("Q1 result:", q1_data)
assert q1_data["found"] is True, f"Expected found=True for Q1, got {q1_data}"
assert len(q1_data["pages"]) > 0, "Expected non-empty pages list"
assert q1_data["quote"], "Expected non-empty quote"

print(f"\n=== 4. Querying Titanium question on doc_id {doc_id_1} ===")
q_ti = "What is the melting point of titanium?"
ask_res_ti = client.post("/ask", json={"question": q_ti, "doc_id": doc_id_1})
assert ask_res_ti.status_code == 200, f"Ask failed: {ask_res_ti.text}"
ti_data = ask_res_ti.json()
print("Titanium result:", ti_data)
assert ti_data["found"] is False, f"Expected found=False for titanium, got {ti_data}"
assert ti_data["answer"] == "Not found", f"Expected 'Not found', got {ti_data['answer']}"

print("\n=== 5. Uploading second PDF (data/test_solar.pdf) ===")
with open("data/test_solar.pdf", "rb") as f:
    upload_res2 = client.post(
        "/upload",
        files={"file": ("test_solar.pdf", f, "application/pdf")}
    )

assert upload_res2.status_code == 200, f"Second upload failed: {upload_res2.status_code} {upload_res2.text}"
upload_data2 = upload_res2.json()
print("Second upload result:", upload_data2)
doc_id_2 = upload_data2["doc_id"]
assert doc_id_2 != doc_id_1, "doc_id should be unique"
assert upload_data2["pages"] == 2, f"Expected 2 pages, got {upload_data2['pages']}"

print(f"\n=== 6. Querying solar PDF on doc_id {doc_id_2} ===")
q_solar = "What is the maximum theoretical efficiency for single-junction silicon solar cells?"
ask_solar = client.post("/ask", json={"question": q_solar, "doc_id": doc_id_2})
assert ask_solar.status_code == 200, f"Ask failed: {ask_solar.text}"
solar_data = ask_solar.json()
print("Solar query result:", solar_data)
assert solar_data["found"] is True, f"Expected found=True, got {solar_data}"
assert "29.4" in solar_data["answer"] or "29.4" in solar_data["quote"], "Expected 29.4% in solar answer"

print(f"\n=== 7. Confirming cross-document isolation: asking gas turbine Q1 on solar doc_id {doc_id_2} ===")
ask_cross = client.post("/ask", json={"question": q1, "doc_id": doc_id_2})
assert ask_cross.status_code == 200, f"Ask cross failed: {ask_cross.text}"
cross_data = ask_cross.json()
print("Cross-document query result on solar doc:", cross_data)
assert cross_data["found"] is False, f"Expected found=False on solar doc for turbine question, got {cross_data}"
assert cross_data["answer"] == "Not found"

print("\n=== 8. Testing 404 for nonexistent doc_id ===")
bad_doc = client.post("/ask", json={"question": "hello", "doc_id": "nonexistent123"})
assert bad_doc.status_code == 404, f"Expected 404, got {bad_doc.status_code}"
print("404 check OK:", bad_doc.json())

print("\n=== 9. Testing 400 for empty question ===")
empty_q = client.post("/ask", json={"question": "   ", "doc_id": doc_id_1})
assert empty_q.status_code == 400, f"Expected 400, got {empty_q.status_code}"
print("Empty question 400 check OK:", empty_q.json())

print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
