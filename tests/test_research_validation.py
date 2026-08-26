from nncpu.research_validation import validate_research_artifact


def test_committed_holdout_research_artifact_is_self_consistent():
    report = validate_research_artifact()
    failures = [
        f"{check.name}: {check.detail}" for check in report.checks if not check.passed
    ]
    assert report.ok, "\n".join(failures)
