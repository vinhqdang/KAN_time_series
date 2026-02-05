import pypdf
import json
import os

pdf_path = "2025437659.pdf"
notebook_path = "Extended ICOIN.ipynb"

print("=== PDF CONTENT START ===")
try:
    if os.path.exists(pdf_path):
        reader = pypdf.PdfReader(pdf_path)
        # Read up to 5 pages to get Introduction and Conclusion
        pages_to_read = list(range(min(5, len(reader.pages))))
        if len(reader.pages) > 5:
            pages_to_read.append(len(reader.pages) - 1)
        
        for i in pages_to_read:
            if i < len(reader.pages):
                print(f"--- Page {i+1} ---")
            text = reader.pages[i].extract_text()
            print(text.encode('ascii', errors='ignore').decode('ascii'))
    else:
        print("PDF file not found.")
except Exception as e:
    print(f"Error reading PDF: {e}")
print("=== PDF CONTENT END ===")

print("\n=== NOTEBOOK MARKDOWN START ===")
try:
    if os.path.exists(notebook_path):
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'markdown':
                    print("--- Markdown Cell ---")
                    print(''.join(cell.get('source', [])))
    else:
        print("Notebook file not found.")
except Exception as e:
    print(f"Error reading notebook: {e}")
print("=== NOTEBOOK MARKDOWN END ===")
