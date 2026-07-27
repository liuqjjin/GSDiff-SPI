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


def test_junit_verifier_accepts_fully_namespaced_report(tmp_path):
    path = _write_report(
        tmp_path / "junit.xml",
        '<j:testsuites xmlns:j="urn:junit" tests="1" failures="0" '
        'errors="0" skipped="0">'
        '<j:testsuite name="suite" tests="1" failures="0" errors="0" skipped="0">'
        '<j:testcase classname="test_sample" name="test_passes"/>'
        "</j:testsuite>"
        "</j:testsuites>",
    )

    summary = verifier.verify_pytest_junit(
        path, created_after_utc=_freshness_boundary(), max_skips=0
    )

    assert summary == {"errors": 0, "failures": 0, "skipped": 0, "tests": 1}


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        ("failure", "failure"),
        ("error", "error"),
        ("skipped", "skipped"),
    ],
)
def test_junit_verifier_rejects_namespace_qualified_outcomes(
    tmp_path, outcome, message
):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuite xmlns:j="urn:junit" tests="1" failures="0" '
        'errors="0" skipped="0">'
        '<testcase classname="test_sample" name="test_bad">'
        f"<j:{outcome}/>"
        "</testcase>"
        "</testsuite>",
    )

    with pytest.raises(verifier.JUnitVerificationError, match=message):
        verifier.verify_pytest_junit(
            path, created_after_utc=_freshness_boundary(), max_skips=0
        )


def test_junit_verifier_rejects_nested_child_suite_count_mismatch(tmp_path):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuites tests="2" failures="0" errors="0" skipped="0">'
        '<testsuite name="parent" tests="2" failures="0" errors="0" skipped="0">'
        '<testsuite name="child-a" tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="a" name="one"/>'
        "</testsuite>"
        '<testsuite name="child-b" tests="0" failures="0" errors="0" skipped="0">'
        '<testcase classname="b" name="two"/>'
        "</testsuite>"
        "</testsuite>"
        "</testsuites>",
    )

    with pytest.raises(verifier.JUnitVerificationError, match="child-a.*tests"):
        verifier.verify_pytest_junit(
            path, created_after_utc=_freshness_boundary(), max_skips=0
        )


def test_junit_verifier_rejects_root_aggregate_count_mismatch(tmp_path):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuites tests="1" failures="0" errors="0" skipped="0">'
        '<testsuite name="suite" tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="a" name="one"/>'
        '<testcase classname="a" name="two"/>'
        "</testsuite>"
        "</testsuites>",
    )

    with pytest.raises(verifier.JUnitVerificationError, match="tests.*mismatch"):
        verifier.verify_pytest_junit(
            path, created_after_utc=_freshness_boundary(), max_skips=0
        )


def test_junit_verifier_accepts_consistent_nested_suites_without_double_counting(
    tmp_path,
):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuites tests="2" failures="0" errors="0" skipped="0">'
        '<testsuite name="parent" tests="2" failures="0" errors="0" skipped="0">'
        '<testsuite name="child-a" tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="a" name="one"/>'
        "</testsuite>"
        '<testsuite name="child-b" tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="b" name="two"/>'
        "</testsuite>"
        "</testsuite>"
        "</testsuites>",
    )

    summary = verifier.verify_pytest_junit(
        path, created_after_utc=_freshness_boundary(), max_skips=0
    )

    assert summary["tests"] == 2


@pytest.mark.parametrize("boundary_offset_ns", [0, 1])
def test_junit_verifier_rejects_exact_or_one_nanosecond_stale_boundary(
    tmp_path, boundary_offset_ns
):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="test_sample" name="test_passes"/>'
        "</testsuite>",
    )
    modified_ns = path.stat().st_mtime_ns

    with pytest.raises(verifier.JUnitVerificationError, match="stale"):
        verifier.verify_pytest_junit(
            path,
            created_after_utc=modified_ns + boundary_offset_ns,
            max_skips=0,
        )


def test_junit_verifier_accepts_one_nanosecond_fresh_boundary(tmp_path):
    path = _write_report(
        tmp_path / "junit.xml",
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="test_sample" name="test_passes"/>'
        "</testsuite>",
    )
    modified_ns = path.stat().st_mtime_ns

    summary = verifier.verify_pytest_junit(
        path, created_after_utc=modified_ns - 1, max_skips=0
    )

    assert summary["tests"] == 1


def test_parse_utc_preserves_nanosecond_precision():
    later = verifier.parse_utc("2026-07-27T04:24:44.123456789Z")
    earlier = verifier.parse_utc("2026-07-27T04:24:44.123456788Z")

    assert later - earlier == 1
