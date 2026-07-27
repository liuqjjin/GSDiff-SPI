from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Sequence
import xml.etree.ElementTree as ElementTree


class JUnitVerificationError(RuntimeError):
    pass


def parse_utc(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--created-after-utc must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "--created-after-utc must include a UTC offset or Z"
        )
    return parsed.astimezone(timezone.utc)


def _declared_count(element: ElementTree.Element, name: str) -> int | None:
    raw = element.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise JUnitVerificationError(
            f"JUnit {name} count is not an integer: {raw!r}"
        ) from exc
    if value < 0:
        raise JUnitVerificationError(f"JUnit {name} count is negative")
    return value


def verify_pytest_junit(
    report_path: Path | str,
    *,
    created_after_utc: datetime,
    max_skips: int = 0,
) -> dict[str, int]:
    path = Path(report_path)
    if not path.is_file():
        raise JUnitVerificationError(f"JUnit report does not exist: {path}")
    if created_after_utc.tzinfo is None:
        raise JUnitVerificationError("created_after_utc must be timezone-aware")
    if max_skips < 0:
        raise JUnitVerificationError("max_skips must be non-negative")
    modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    boundary = created_after_utc.astimezone(timezone.utc)
    if modified < boundary:
        raise JUnitVerificationError(
            f"JUnit report is stale: modified {modified.isoformat()} "
            f"before {boundary.isoformat()}"
        )

    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise JUnitVerificationError(f"JUnit report is not valid XML: {path}") from exc
    if root.tag not in {"testsuite", "testsuites"}:
        raise JUnitVerificationError(f"unexpected JUnit root element: {root.tag}")

    testcases = root.findall(".//testcase")
    tests = len(testcases)
    failures = sum(len(case.findall("./failure")) for case in testcases)
    errors = sum(len(case.findall("./error")) for case in testcases)
    skipped = sum(len(case.findall("./skipped")) for case in testcases)
    observed = {
        "errors": errors,
        "failures": failures,
        "skipped": skipped,
        "tests": tests,
    }

    declared_source = root
    if root.tag == "testsuites" and not any(root.get(name) for name in observed):
        suites = root.findall("./testsuite")
        declared = {
            name: sum(_declared_count(suite, name) or 0 for suite in suites)
            for name in observed
        }
    else:
        declared = {
            name: _declared_count(declared_source, name) for name in observed
        }
    for name, value in declared.items():
        if value is not None and value != observed[name]:
            raise JUnitVerificationError(
                f"JUnit {name} count mismatch: declared {value}, observed {observed[name]}"
            )

    if tests == 0:
        raise JUnitVerificationError("JUnit report contains zero tests")
    if failures:
        raise JUnitVerificationError(f"JUnit report contains {failures} failure(s)")
    if errors:
        raise JUnitVerificationError(f"JUnit report contains {errors} error(s)")
    if skipped > max_skips:
        raise JUnitVerificationError(
            f"JUnit report contains {skipped} skipped test(s), maximum is {max_skips}"
        )
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reject incomplete, failing, skipped, or stale pytest JUnit evidence."
    )
    parser.add_argument("report_path", type=Path)
    parser.add_argument("--created-after-utc", type=parse_utc, required=True)
    parser.add_argument("--max-skips", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = verify_pytest_junit(
            args.report_path,
            created_after_utc=args.created_after_utc,
            max_skips=args.max_skips,
        )
    except JUnitVerificationError as exc:
        print(f"pytest_junit_verification=failed: {exc}", file=sys.stderr)
        return 1
    print(
        "pytest_junit_verification=passed "
        f"tests={summary['tests']} failures={summary['failures']} "
        f"errors={summary['errors']} skipped={summary['skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
