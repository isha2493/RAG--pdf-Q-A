import pymupdf4llm

pages = pymupdf4llm.to_markdown("data/rag-ios.pdf", page_chunks=True)

print(type(pages))     # what kind of object is this?
print(len(pages))      # how many items?
print(pages[0].keys()) # what's inside one item?
