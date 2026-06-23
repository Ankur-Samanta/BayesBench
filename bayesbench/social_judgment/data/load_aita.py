"""Load AITA dataset from HuggingFace."""

from datasets import load_dataset
from typing import List, Dict, Literal
import random


def load_aita_dataset(
    mode: Literal["binary", "full"] = "binary",
    max_samples: int = 1000,
    min_length: int = 200,
    max_length: int = 5000,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Dict]:
    """Load AITA dataset from HuggingFace.

    Args:
        mode: "binary" (YTA/NTA only) or "full" (all 4 verdicts)
        max_samples: Maximum number of samples to load
        min_length: Minimum post text length (filter out very short posts)
        max_length: Maximum post text length (filter out very long posts)
        seed: Random seed for shuffling
        shuffle: Whether to shuffle the dataset

    Returns:
        List of dicts with: id, title, text, verdict, is_yta, score
    """
    print(f"Loading AITA dataset from HuggingFace (mode={mode})...")
    ds = load_dataset("OsamaBsher/AITA-Reddit-Dataset")

    samples = []
    for item in ds["train"]:
        # Filter by mode - verdicts are lowercase in the dataset
        verdict = item.get("verdict", "").upper()
        if mode == "binary" and verdict not in ["YTA", "NTA"]:
            continue

        # Filter by text length
        text = item.get("text", "")
        if len(text) < min_length or len(text) > max_length:
            continue

        # Skip items with missing fields
        if not item.get("id") or not item.get("title"):
            continue

        samples.append({
            "id": item["id"],
            "title": item["title"],
            "text": text,
            "verdict": verdict,
            "is_yta": verdict in ["YTA", "ESH"],  # True for asshole verdicts
            "score": item.get("score", 0),
        })

    print(f"Found {len(samples)} valid samples")

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(samples)

    # Limit samples
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    print(f"Returning {len(samples)} samples")
    return samples


def split_into_chunks(text: str, n_chunks: int = 5) -> List[str]:
    """Split text into approximately equal chunks by sentences.

    Args:
        text: The full post text
        n_chunks: Number of chunks to split into

    Returns:
        List of text chunks
    """
    # Split by sentence-ending punctuation
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) <= n_chunks:
        # If fewer sentences than chunks, return one sentence per chunk
        return sentences

    # Distribute sentences across chunks
    chunk_size = len(sentences) / n_chunks
    chunks = []
    current_chunk = []
    current_count = 0

    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        current_count += 1

        # Start new chunk when we've accumulated enough sentences
        target_sentences = (len(chunks) + 1) * chunk_size
        if current_count >= target_sentences or i == len(sentences) - 1:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Ensure we have exactly n_chunks
    while len(chunks) < n_chunks:
        chunks.append("")
    while len(chunks) > n_chunks:
        # Merge last two chunks
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks = chunks[:-1]

    return chunks


def get_verdict_label(verdict: str, mode: str = "binary") -> str:
    """Get human-readable verdict label.

    Args:
        verdict: The verdict code (YTA, NTA, ESH, NAH)
        mode: "binary" or "full"

    Returns:
        Human-readable label
    """
    labels = {
        "YTA": "You're The Asshole",
        "NTA": "Not The Asshole",
        "ESH": "Everyone Sucks Here",
        "NAH": "No Assholes Here",
    }
    return labels.get(verdict, verdict)


if __name__ == "__main__":
    # Test loading
    print("Testing binary mode:")
    samples = load_aita_dataset(mode="binary", max_samples=10)
    for s in samples[:3]:
        print(f"  [{s['verdict']}] {s['title'][:50]}...")
        print(f"    Text length: {len(s['text'])}, is_yta: {s['is_yta']}")

    print("\nTesting full mode:")
    samples = load_aita_dataset(mode="full", max_samples=10)
    for s in samples[:3]:
        print(f"  [{s['verdict']}] {s['title'][:50]}...")

    print("\nTesting chunk splitting:")
    test_text = "First sentence. Second sentence! Third sentence? Fourth. Fifth. Sixth. Seventh. Eighth. Ninth. Tenth."
    chunks = split_into_chunks(test_text, n_chunks=3)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk}")
