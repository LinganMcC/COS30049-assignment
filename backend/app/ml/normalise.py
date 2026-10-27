"""Unicode normalisation - the defence against detection evasion (section 2.8).

The attack: change characters so a person still reads the same text but the
model does not. Our tokenizer was trained from scratch on plain English, so a
Cyrillic 'а' in place of a Latin 'a', or a zero-width space inside a word,
splits the word into rare or unknown tokens and pulls the score towards human.

The defence runs before sentence splitting, in three steps:
    1. NFKC        fullwidth 'ｔｅｘｔ' -> 'text', ligature 'ﬁ' -> 'fi',
                   maths letters '𝐭𝐞𝐱𝐭' -> 'text', no-break space -> space
    2. strip Cf    invisible "format" characters: zero-width space/joiners,
                   soft hyphen, byte-order mark, bidirectional controls
    3. confusables Cyrillic and Greek letters that look like Latin ones
                   (NFKC leaves these alone because they are real letters)

Cost: genuine Cyrillic or Greek words are turned into Latin look-alikes, so
the detector only suits English text - which is already its stated scope
(A1 assumptions: English-language focus).
"""
import unicodedata
from dataclasses import dataclass

# Look-alikes that NFKC does not change. Not every Unicode confusable - the
# letters that are visually identical to Latin in common fonts.
CONFUSABLES = {
    # Cyrillic lower case
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
    'і': 'i', 'ј': 'j', 'ѕ': 's', 'ԁ': 'd', 'һ': 'h', 'ԛ': 'q', 'ԝ': 'w',
    'ӏ': 'l', 'ү': 'y', 'ɡ': 'g',
    # Cyrillic upper case
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
    'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X', 'У': 'Y', 'І': 'I', 'Ј': 'J',
    'Ѕ': 'S', 'Ԛ': 'Q', 'Ԝ': 'W',
    # Greek
    'ο': 'o', 'ν': 'v', 'α': 'a', 'ι': 'i', 'κ': 'k', 'ρ': 'p', 'τ': 't',
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H', 'Ι': 'I', 'Κ': 'K',
    'Μ': 'M', 'Ν': 'N', 'Ο': 'O', 'Ρ': 'P', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
}
_CONFUSABLES_TABLE = str.maketrans(CONFUSABLES)


@dataclass(frozen=True)
class NormalisationResult:
    text: str
    characters_changed: int   # shown to the user, and logged as an evasion signal


def normalise(text: str) -> NormalisationResult:
    nfkc = unicodedata.normalize('NFKC', text)
    visible = ''.join(ch for ch in nfkc if unicodedata.category(ch) != 'Cf')
    latin = visible.translate(_CONFUSABLES_TABLE)
    return NormalisationResult(latin, sum(_is_suspicious(ch) for ch in text))


def _is_suspicious(ch: str) -> bool:
    """True for a character one of the three steps above would change or remove."""
    return (unicodedata.category(ch) == 'Cf' or ch in CONFUSABLES
            or unicodedata.normalize('NFKC', ch) != ch)
