
import re
import spacy


# ============================================================
# 1. LOAD THE SPACY MODEL
# ============================================================

nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner", "lemmatizer", "attribute_ruler"]
)


# ============================================================
# 2. NORMALISE PUNCTUATION SPACING
# ============================================================

def normalize_text(text: str) -> str:

    """
    Repair targeted punctuation-spacing errors.

    Example:
        "Hello .World" -> "Hello. World"

        "Hello.World" -> "Hello. World"
    """

    text = re.sub(
        r'(?<=[a-z])\s*([.!?])(?=[A-Z])',
        r'\1 ',
        text
    )

    return text


# ============================================================
# 3. SPLIT TEXT INTO SENTENCES
# ============================================================

def split_sentences(text: str) -> list[str]:

    """
    Split a complete passage into individual sentences
    using spaCy.

    Returns:
        A list of sentence strings.
    """

    # Handle missing input.
    if text is None:
        return []

    text = str(text).strip()

    if not text:
        return []

    # Repair targeted punctuation-spacing errors.
    text = normalize_text(text)

    # Process the passage with spaCy.
    doc = nlp(text)

    # Extract the identified sentences.
    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]

    return sentences

