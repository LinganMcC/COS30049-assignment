from app.ml.normalise import normalise


def test_plain_text_is_unchanged():
    result = normalise('A perfectly ordinary sentence.')
    assert result.text == 'A perfectly ordinary sentence.'
    assert result.characters_changed == 0


def test_cyrillic_homoglyphs_become_latin():
    result = normalise('Thе cоmputеr')        # е, о, е are Cyrillic
    assert result.text == 'The computer'
    assert result.characters_changed == 3


def test_zero_width_characters_are_removed():
    result = normalise('com​pu‍ter﻿')
    assert result.text == 'computer'
    assert result.characters_changed == 3


def test_fullwidth_and_maths_letters_fold_to_ascii():
    assert normalise('ｔｅｘｔ').text == 'text'
    assert normalise('𝐭𝐞𝐱𝐭').text == 'text'
