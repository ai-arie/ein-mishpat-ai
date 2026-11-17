from ein_mishpat_ai.data.unknown_report import (
    TitleResolver,
    excluded_hebrew_works,
    parse_publisher,
    parse_year,
)


def test_parse_year_prefers_title_and_earliest():
    assert (
        parse_year("Tractate Berakot by A. Cohen, Cambridge University Press, 1921", None) == 1921
    )
    assert parse_year("Translated 1880-1890", "x") == 1880
    assert parse_year("No year here", "https://archive.org/details/x1913y") is None
    assert parse_year("No year", "https://example.org/1913/book") == 1913
    assert parse_year("NNL_ALEPH001741739", None) is None


def test_parse_publisher():
    assert parse_publisher(
        "Tractate Berakot by A. Cohen, Cambridge University Press, 1921", ""
    ) == ("Cambridge University Press")
    assert (
        parse_publisher("Some Translation", "https://www.sacred-texts.com/jud/")
        == "sacred-texts.com"
    )


def test_title_resolver_drops_node_parts():
    r = TitleResolver({"Tur", "Berakhot", "Introductions to Amoraic Literature"})
    assert r("Tur, Orach Chayim") == "Tur"
    assert r("Berakhot") == "Berakhot"
    assert r("Introductions to Amoraic Literature, Part I (X), Pesachim") == (
        "Introductions to Amoraic Literature"
    )
    assert r("Unknown Work, Part 2") is None


def test_excluded_hebrew_works():
    recs = [
        {"title": "A", "language": "Hebrew", "actualLanguage": "he", "status": "kept"},
        {"title": "A", "language": "Hebrew", "actualLanguage": "he", "status": "skipped"},
        {"title": "B", "language": "Hebrew", "actualLanguage": "he", "status": "skipped"},
        {"title": "B", "language": "Hebrew", "actualLanguage": "yi", "status": "kept"},  # not orig
        {"title": "C", "language": "English", "actualLanguage": "en", "status": "skipped"},
    ]
    lost = excluded_hebrew_works(recs)
    assert set(lost) == {"B"} and len(lost["B"]) == 1
