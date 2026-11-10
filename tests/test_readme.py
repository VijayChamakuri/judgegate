from judgegate.readme import README, render


def test_readme_matches_results():
    """README.md is rendered from results/*.json; regenerate with `judgegate readme` if this fails."""
    assert README.read_text() == render(write=False)
