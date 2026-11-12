from judgegate.readme import README, RESULTS_DOC, render, render_results_text


def test_readme_matches_results():
    """README.md is rendered from results/*.json; regenerate with `judgegate readme` if this fails."""
    assert README.read_text() == render(write=False)


def test_results_doc_matches_results():
    """docs/RESULTS.md is the long tables. Same render path as the README."""
    assert RESULTS_DOC.read_text() == render_results_text()
