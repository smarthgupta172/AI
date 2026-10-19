from langchain_text_splitters import(
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

text="""
The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.

The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."""

#FIXED-SIZE CHUNKING

fixed=CharacterTextSplitter(
    separator="",
    chunk_size=50,
    chunk_overlap=0,
)
print("Fixed-size chunking")

for i, chunk in enumerate(fixed.split_text(text)):
    print(f"Chunk {i}: {chunk}")


# PARAGRAPH CHUNKING

paragraph = CharacterTextSplitter(
separator="\n\n",
chunk_size=100,
chunk_overlap=0
)
print("\n========== PARAGRAPH =============")
for i, chunk in enumerate (paragraph.split_text(text), 1):
    print (f"\nChunk {i}:")
    print(chunk) 

# RECURSIVE CHUNKING

recursive = RecursiveCharacterTextSplitter(
chunk_size=100,
chunk_overlap=20
)

for i, chunk in enumerate (recursive.split_text(text), 1):
    print (f"\nChunk {i}:")
    print(chunk)
    print("\n========== RECURSIVE =============")