from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

import pytest

import scripts.reproducibility.verify_pytest_junit as verifier


def _write_report(path, body):
    path.write_text(body, encoding="utf-8")
    return path


def _freshness_boundary():
    return datetime.now(timezone.utc) - timedelta(minutes=1)


def test_junit_verifier_accepts_fresh_passing_report(tmp_path):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="test_sample" name="test_passes"/>'
        "</testsuite>",
    )

    summary = verifier.verify_pytest_junit(
        path, created_after_utc=_freshness_boundary(), max_skips=0
    )

    assert summary == {"errors": 0, "failures": 0, "skipped": 0, "tests": 1}


def test_junit_verifier_requires_report_path():
    with pytest.raises(SystemExit) as exc:
        verifier.main(["--created-after-utc", "2026-07-27T00:00:00Z"])

    assert exc.value.code == 2


def test_junit_verifier_rejects_missing_report(tmp_path):
    with pytest.raises(verifier.JUnitVerificationError, match="does not exist"):
        verifier.verify_pytest_junit(
            tmp_path / "missing.xml",
            created_after_utc=_freshness_boundary(),
            max_skips=0,
        )


def test_junit_verifier_rejects_malformed_xml(tmp_path):
    path = _write_report(tmp_path / "junit.xml", "<testsuite>")

    with pytest.raises(verifier.JUnitVerificationError, match="valid XML"):
        verifier.verify_pytest_junit(
            path, created_after_utc=_freshness_boundary(), max_skips=0
        )


def test_junit_verifier_rejects_zero_tests(tmp_path):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuite tests="0" failures="0" errors="0" skipped="0"/>',
    )

    with pytest.raises(verifier.JUnitVerificationError, match="zero tests"):
        verifier.verify_pytest_junit(
            path, created_after_utc=_freshness_boundary(), max_skips=0
        )


@pytest.mark.parametrize(("child", "message"), [("<failure/>", "failure"), ("<error/>", "error")])
def test_junit_verifier_rejects_failures_and_errors(tmp_path, child, message):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuite tests="1" failures="1" errors="0" skipped="0">'
        f'<testcase classname="test_sample" name="test_bad">{child}</testcase>'
        "</testsuite>",
    )
    if child == "<error/>":
        path.write_text(
            path.read_text(encoding="utf-8")
            .replace('failures="1"', 'failures="0"')
            .replace('errors="0"', 'errors="1"'),
            encoding="utf-8",
        )

    with pytest.raises(verifier.JUnitVerificationError, match=message):
        verifier.verify_pytest_junit(
            path, created_after_utc=_freshness_boundary(), max_skips=0
        )


def test_junit_verifier_rejects_excessive_skips(tmp_path):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuite tests="1" failures="0" errors="0" skipped="1">'
        '<testcase classname="test_sample" name="test_skip"><skipped/></testcase>'
        "</testsuite>",
    )

    with pytest.raises(verifier.JUnitVerificationError, match="skipped"):
        verifier.verify_pytest_junit(
            path, created_after_utc=_freshness_boundary(), max_skips=0
        )


def test_junit_verifier_rejects_stale_report(tmp_path):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="test_sample" name="test_passes"/>'
        "</testsuite>",
    )
    stale_time = datetime.now(timezone.utc) - timedelta(hours=2)
    os.utime(path, (stale_time.timestamp(), stale_time.timestamp()))

    with pytest.raises(verifier.JUnitVerificationError, match="stale"):
        verifier.verify_pytest_junit(
            path,
            created_after_utc=datetime.now(timezone.utc) - timedelta(hours=1),
            max_skips=0,
        )
