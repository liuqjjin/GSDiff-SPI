from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Sequence
import xml.etree.ElementTree as ElementTree


class JUnitVerificationError(RuntimeError):
    pass


_UTC_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?"
    r"(?P<zone>Z|[+-]\d{2}:\d{2})$"
)
_NANOSECONDS_PER_SECOND = 1_000_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def parse_utc(value: str) -> int:
    match = _UTC_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "--created-after-utc must be an ISO-8601 timestamp with up to "
            "nine fractional digits"
        )
    zone = "+00:00" if match.group("zone") == "Z" else match.group("zone")
    try:
        parsed = datetime.fromisoformat(match.group("date") + zone)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--created-after-utc must be an ISO-8601 timestamp"
        ) from exc
    delta = parsed.astimezone(timezone.utc) - _EPOCH
    whole_seconds = delta.days * 86_400 + delta.seconds
    fraction_ns = int((match.group("fraction") or "").ljust(9, "0"))
    return whole_seconds * _NANOSECONDS_PER_SECOND + fraction_ns


def _boundary_nanoseconds(value: datetime | int) -> int:
    if isinstance(value, bool):
        raise JUnitVerificationError(
            "created_after_utc must be a timezone-aware datetime or integer nanoseconds"
        )
    if isinstance(value, int):
        return value
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise JUnitVerificationError("created_after_utc must be timezone-aware")
    delta = value.astimezone(timezone.utc) - _EPOCH
    return (
        (delta.days * 86_400 + delta.seconds) * _NANOSECONDS_PER_SECOND
        + delta.microseconds * 1_000
    )


def _format_utc_nanoseconds(value: int) -> str:
    seconds, nanoseconds = divmod(value, _NANOSECONDS_PER_SECOND)
    whole = datetime.fromtimestamp(seconds, timezone.utc)
    return whole.strftime("%Y-%m-%dT%H:%M:%S") + f".{nanoseconds:09d}Z"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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


def _observed_counts(testcases: list[ElementTree.Element]) -> dict[str, int]:
    def has_outcome(testcase: ElementTree.Element, outcome: str) -> bool:
        return any(
            element is not testcase and _local_name(element.tag) == outcome
            for element in testcase.iter()
        )

    return {
        "errors": sum(has_outcome(case, "error") for case in testcases),
        "failures": sum(has_outcome(case, "failure") for case in testcases),
        "skipped": sum(has_outcome(case, "skipped") for case in testcases),
        "tests": len(testcases),
    }


def _descendant_testcases(element: ElementTree.Element) -> list[ElementTree.Element]:
    return [
        descendant
        for descendant in element.iter()
        if _local_name(descendant.tag) == "testcase"
    ]


def _validate_declared_counts(
    element: ElementTree.Element, observed: dict[str, int]
) -> None:
    label = element.get("name") or _local_name(element.tag)
    for name, observed_value in observed.items():
        declared = _declared_count(element, name)
        if declared is not None and declared != observed_value:
            raise JUnitVerificationError(
                f"JUnit {label!r} {name} count mismatch: "
                f"declared {declared}, observed {observed_value}"
            )


def verify_pytest_junit(
    report_path: Path | str,
    *,
    created_after_utc: datetime | int,
    max_skips: int = 0,
) -> dict[str, int]:
    path = Path(report_path)
    if not path.is_file():
        raise JUnitVerificationError(f"JUnit report does not exist: {path}")
    if max_skips < 0:
        raise JUnitVerificationError("max_skips must be non-negative")
    modified_ns = path.stat().st_mtime_ns
    boundary_ns = _boundary_nanoseconds(created_after_utc)
    if modified_ns <= boundary_ns:
        raise JUnitVerificationError(
            f"JUnit report is stale: modified {_format_utc_nanoseconds(modified_ns)} "
            f"is not after {_format_utc_nanoseconds(boundary_ns)}"
        )

    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise JUnitVerificationError(f"JUnit report is not valid XML: {path}") from exc
    root_name = _local_name(root.tag)
    if root_name not in {"testsuite", "testsuites"}:
        raise JUnitVerificationError(f"unexpected JUnit root element: {root.tag}")

    observed = _observed_counts(_descendant_testcases(root))
    if root_name == "testsuites":
        _validate_declared_counts(root, observed)
    for suite in (
        element for element in root.iter() if _local_name(element.tag) == "testsuite"
    ):
        _validate_declared_counts(
            suite, _observed_counts(_descendant_testcases(suite))
        )

    if observed["tests"] == 0:
        raise JUnitVerificationError("JUnit report contains zero tests")
    if observed["failures"]:
        raise JUnitVerificationError(
            f"JUnit report contains {observed['failures']} failure(s)"
        )
    if observed["errors"]:
        raise JUnitVerificationError(
            f"JUnit report contains {observed['errors']} error(s)"
        )
    if observed["skipped"] > max_skips:
        raise JUnitVerificationError(
            f"JUnit report contains {observed['skipped']} skipped test(s), "
            f"maximum is {max_skips}"
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
