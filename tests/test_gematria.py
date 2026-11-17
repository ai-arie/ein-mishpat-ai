import pytest

from ein_mishpat_ai.gen.translate import batch_messages, gematria_digits


@pytest.mark.parametrize(
    "src, out",
    [
        ("ועיין בסי' קע\"ג", "ועיין בסי' 173"),
        ('בנדרים דף ח"י', "בנדרים דף 18"),
        ('דף כ"ב ע"א', 'דף 22 ע"א'),
        ("פרק ד' משנה כ'", "פרק 4 משנה 20"),
        ('סעיף י"ד וס"ק ט"ו', 'סעיף 14 וס"ק 15'),
        ('סימן ת"ח', "סימן 408"),
        ('סי\' ר"ה סק"א', "סי' 205 סק\"א"),
        ('פרק קי"ט', "פרק 119"),
        # abbreviations after a marker stay as they are
        ("סי' הנ\"ל", "סי' הנ\"ל"),
        ('פרק ר"ע במסכת', 'פרק ר"ע במסכת'),
        ('משנה ר"פ דם', 'משנה ר"פ דם'),
        ('ס"ק כ"כ', 'ס"ק כ"כ'),
        ('מאמר ר"י אמת', 'מאמר ר"י אמת'),
        ("בתוס' ד\"ה והשאר", "בתוס' ד\"ה והשאר"),
        ("פרק המקבל", "פרק המקבל"),
    ],
)
def test_gematria_digits(src: str, out: str) -> None:
    assert gematria_digits(src) == out


def test_numerals_flag_off_keeps_prompt() -> None:
    # blocks before numerals_from_block must produce the same requests (cache keys) as before
    a = batch_messages("T", ["T 1:1"], ["סי' קע\"ג"], True, "")
    b = batch_messages("T", ["T 1:1"], ["סי' קע\"ג"], True, "", numerals=False)
    assert a == b and 'קע"ג' in a[1]["content"]
    assert "173" in batch_messages("T", ["T 1:1"], ["סי' קע\"ג"], True, "", True)[1]["content"]
