from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

import gsdiff.experiments.source_snapshot as snapshot_module
import gsdiff.data._artifact_persistence as artifact_persistence_module
from gsdiff.experiments.identity import canonical_json_bytes
from gsdiff.experiments.source_snapshot import (
    materialize_source_snapshot,
    selected_source_evidence,
    verify_source_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SOURCE_ROOTS = (
    Path("gsdiff"),
    Path("scripts"),
    Path("configs"),
    Path("schemas"),
    Path("assets"),
    Path("train.py"),
    Path("requirements-lock.txt"),
    Path("docs/reproducibility/environment-lock.json"),
)


def _init_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "snapshot@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Snapshot Test"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "core.ignorecase", "false"],
        cwd=path,
        check=True,
    )
    return path


def _commit(path: Path, message: str = "snapshot") -> str:
    subprocess.run(["git", "commit", "-m", message], cwd=path, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _index_blob(repo: Path, mode: str, path: str, payload: bytes = b"payload\n") -> None:
    oid = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=payload,
        capture_output=True,
        check=True,
    ).stdout.decode("ascii").strip()
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"{mode},{oid},{path}"],
        cwd=repo,
        check=True,
    )


def _snapshot_fixture(tmp_path: Path):
    repo = _init_repo(tmp_path / "repo")
    _index_blob(repo, "100644", "gsdiff/module.py", b"committed\n")
    commit = _commit(repo)
    snapshot = materialize_source_snapshot(
        repo,
        tmp_path / "artifacts",
        commit,
        (Path("gsdiff"),),
    )
    return repo, commit, snapshot


def test_source_snapshot_uses_claimed_commit_bytes_after_live_tree_changes(
    tmp_path: Path,
):
    repo = _init_repo(tmp_path / "repo")
    source = repo / "gsdiff"
    source.mkdir()
    tracked = source / "module.py"
    tracked.write_text("COMMITTED = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "gsdiff/module.py"], cwd=repo, check=True)
    commit = _commit(repo)

    snapshot = materialize_source_snapshot(
        repo,
        tmp_path / "artifacts",
        commit,
        (Path("gsdiff"),),
    )
    tracked.write_text("LIVE_ATTACK = 2\n", encoding="utf-8")
    verify_source_snapshot(snapshot)
    reused = materialize_source_snapshot(
        repo,
        tmp_path / "artifacts",
        commit,
        (Path("gsdiff"),),
    )

    assert (snapshot.root / "gsdiff/module.py").read_text("utf-8") == (
        "COMMITTED = 1\n"
    )
    assert reused == snapshot
    assert not list(snapshot.root.parent.glob("*.tmp-*"))
    assert snapshot.root.parent == tmp_path / "artifacts/source-snapshots"


@pytest.mark.parametrize("unsafe_kind", ["symlink", "case-collision"])
def test_source_snapshot_rejects_unsafe_claimed_git_tree(
    tmp_path: Path,
    unsafe_kind: str,
):
    repo = _init_repo(tmp_path / "repo")
    if unsafe_kind == "symlink":
        entries = [("120000", "gsdiff/link.py")]
    else:
        entries = [
            ("100644", "gsdiff/Module.py"),
            ("100644", "gsdiff/module.py"),
        ]
    for mode, path in entries:
        _index_blob(repo, mode, path)
    commit = _commit(repo, "unsafe tree")

    with pytest.raises(ValueError, match="symlink|gitlink|collision|mode"):
        materialize_source_snapshot(
            repo,
            tmp_path / "artifacts",
            commit,
            (Path("gsdiff"),),
        )

    assert not (tmp_path / "artifacts/source-snapshots").exists()


@pytest.mark.parametrize("commit", ["deadbeef", "f" * 40])
def test_source_snapshot_rejects_wrong_or_nonexistent_commit(
    tmp_path: Path,
    commit: str,
):
    repo = _init_repo(tmp_path / "repo")
    _index_blob(repo, "100644", "gsdiff/module.py")

    with pytest.raises(ValueError, match="commit|enumerate"):
        materialize_source_snapshot(
            repo,
            tmp_path / "artifacts",
            commit,
            (Path("gsdiff"),),
        )


@pytest.mark.parametrize("object_kind", ["tree", "blob", "tag"])
def test_source_snapshot_rejects_noncommit_git_object_ids(
    tmp_path: Path,
    object_kind: str,
):
    repo = _init_repo(tmp_path / "repo")
    _index_blob(repo, "100644", "gsdiff/module.py")
    _commit(repo)
    if object_kind == "tree":
        revision = "HEAD^{tree}"
    elif object_kind == "blob":
        revision = "HEAD:gsdiff/module.py"
    else:
        subprocess.run(
            ["git", "tag", "-a", "snapshot-tag", "-m", "snapshot tag"],
            cwd=repo,
            check=True,
        )
        revision = "refs/tags/snapshot-tag"
    object_id = subprocess.check_output(
        ["git", "rev-parse", revision],
        cwd=repo,
        text=True,
    ).strip()

    with pytest.raises(ValueError, match="commit object"):
        materialize_source_snapshot(
            repo,
            tmp_path / "artifacts",
            object_id,
            (Path("gsdiff"),),
        )

    assert not (tmp_path / "artifacts").exists()


def test_source_snapshot_rejects_gitlink(tmp_path: Path):
    repo = _init_repo(tmp_path / "repo")
    _index_blob(repo, "100644", "base.txt")
    target_commit = _commit(repo, "base")
    subprocess.run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{target_commit},gsdiff/submodule",
        ],
        cwd=repo,
        check=True,
    )
    commit = _commit(repo, "gitlink")

    with pytest.raises(ValueError, match="gitlink|mode"):
        materialize_source_snapshot(
            repo,
            tmp_path / "artifacts",
            commit,
            (Path("gsdiff"),),
        )


@pytest.mark.parametrize(
    "paths",
    [
        ["../escape.py"],
        [r"gsdiff\module.py"],
        [r"gsdiff\..\outside.txt"],
        [r"\\server\share\payload.py"],
        ["gsdiff/duplicate.py", "gsdiff/duplicate.py"],
        ["gsdiff/CON.py"],
        ["gsdiff/CONIN$"],
        ["gsdiff/CONOUT$.txt"],
        ["gsdiff/COM¹.py"],
        ["gsdiff/LPT³"],
        ["gsdiff/trailing. "],
        ["gsdiff/Module.py", "gsdiff/module.py"],
        ["gsdiff/Straße.py", "gsdiff/STRASSE.py"],
    ],
)
def test_source_snapshot_manifest_rejects_ambiguous_or_escaping_paths(paths):
    inventory = [
        {
            "path": path,
            "mode": "100644",
            "git_blob": "a" * 40,
            "sha256": "b" * 64,
            "size_bytes": 1,
        }
        for path in sorted(paths, key=lambda value: value.encode("utf-8"))
    ]
    identity = {
        "schema": "source-snapshot-identity-v1",
        "commit": "c" * 40,
        "inventory": inventory,
    }
    manifest = {
        "schema": "source-snapshot-v1",
        "commit": "c" * 40,
        "snapshot_sha256": __import__("hashlib").sha256(
            canonical_json_bytes(identity)
        ).hexdigest(),
        "inventory": inventory,
    }

    with pytest.raises(ValueError, match="path|collision|reserved|ambiguous|sorted"):
        snapshot_module._load_canonical_manifest(canonical_json_bytes(manifest))


@pytest.mark.parametrize(
    "unsafe_path",
    [r"gsdiff\..\outside.txt", r"\\server\share\payload.py"],
)
def test_source_snapshot_rejects_unsafe_manifest_path_before_derived_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_path: str,
):
    inventory = [
        {
            "path": unsafe_path,
            "mode": "100644",
            "git_blob": "a" * 40,
            "sha256": "b" * 64,
            "size_bytes": 1,
        }
    ]
    identity = {
        "schema": "source-snapshot-identity-v1",
        "commit": "c" * 40,
        "inventory": inventory,
    }
    manifest = {
        "schema": "source-snapshot-v1",
        "commit": "c" * 40,
        "snapshot_sha256": __import__("hashlib").sha256(
            canonical_json_bytes(identity)
        ).hexdigest(),
        "inventory": inventory,
    }
    snapshot_root = tmp_path / manifest["snapshot_sha256"]
    snapshot_root.mkdir()
    manifest_path = snapshot_root / "source-snapshot.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    original_read = snapshot_module._read_stable_regular_bytes
    reads: list[Path] = []

    def manifest_only_read(path: Path, *, noun: str) -> bytes:
        reads.append(path)
        if path != manifest_path:
            raise AssertionError(f"unsafe manifest triggered derived I/O: {path}")
        return original_read(path, noun=noun)

    monkeypatch.setattr(
        snapshot_module,
        "_read_stable_regular_bytes",
        manifest_only_read,
    )

    with pytest.raises(ValueError, match="path|ambiguous|unsafe"):
        verify_source_snapshot(snapshot_root)

    assert reads == [manifest_path]


def test_source_snapshot_ignores_git_replace_refs(tmp_path: Path):
    repo = _init_repo(tmp_path / "repo")
    _index_blob(repo, "100644", "gsdiff/module.py", b"original\n")
    claimed_commit = _commit(repo, "claimed")
    _index_blob(repo, "100644", "gsdiff/module.py", b"replacement\n")
    replacement_commit = _commit(repo, "replacement")
    subprocess.run(
        ["git", "replace", claimed_commit, replacement_commit],
        cwd=repo,
        check=True,
    )

    snapshot = materialize_source_snapshot(
        repo,
        tmp_path / "artifacts",
        claimed_commit,
        (Path("gsdiff"),),
    )

    assert snapshot.commit == claimed_commit
    assert (snapshot.root / "gsdiff/module.py").read_bytes() == b"original\n"


def test_source_snapshot_ignores_git_blob_replace_refs(tmp_path: Path):
    repo = _init_repo(tmp_path / "repo")
    _index_blob(repo, "100644", "gsdiff/module.py", b"original\n")
    claimed_commit = _commit(repo, "claimed")
    claimed_blob = subprocess.check_output(
        ["git", "rev-parse", f"{claimed_commit}:gsdiff/module.py"],
        cwd=repo,
        text=True,
    ).strip()
    replacement_blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=b"attacked\n",
        capture_output=True,
        check=True,
    ).stdout.decode("ascii").strip()
    subprocess.run(
        ["git", "replace", claimed_blob, replacement_blob],
        cwd=repo,
        check=True,
    )

    snapshot = materialize_source_snapshot(
        repo,
        tmp_path / "artifacts",
        claimed_commit,
        (Path("gsdiff"),),
    )

    assert (snapshot.root / "gsdiff/module.py").read_bytes() == b"original\n"


@pytest.mark.parametrize(
    "mutation",
    [
        "content",
        "manifest",
        "extra",
        "empty-dir",
        "missing",
        "hardlink",
        "wrong-dir",
    ],
)
def test_source_snapshot_strict_reuse_rejects_physical_drift(
    tmp_path: Path,
    mutation: str,
):
    _repo, _commit_id, snapshot = _snapshot_fixture(tmp_path)
    payload = snapshot.root / "gsdiff/module.py"
    if mutation == "content":
        payload.write_bytes(b"changed\n")
    elif mutation == "manifest":
        manifest = snapshot.root / "source-snapshot.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
    elif mutation == "extra":
        (snapshot.root / "extra.txt").write_text("extra", encoding="ascii")
    elif mutation == "empty-dir":
        (snapshot.root / "extra-empty").mkdir()
    elif mutation == "missing":
        payload.unlink()
    elif mutation == "hardlink":
        victim = tmp_path / "victim.py"
        victim.write_bytes(payload.read_bytes())
        payload.unlink()
        os.link(victim, payload)
    else:
        wrong = snapshot.root.with_name("0" * 64)
        os.rename(snapshot.root, wrong)
        snapshot = snapshot_module.SourceSnapshot(
            wrong,
            snapshot.commit,
            snapshot.snapshot_sha256,
            snapshot.inventory,
        )

    with pytest.raises(ValueError, match="snapshot|file|inventory|identity|regular"):
        verify_source_snapshot(snapshot)


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse contract")
def test_source_snapshot_rejects_reparse_entry(tmp_path: Path):
    _repo, _commit_id, snapshot = _snapshot_fixture(tmp_path)
    payload = snapshot.root / "gsdiff/module.py"
    target = tmp_path / "target.py"
    target.write_bytes(payload.read_bytes())
    payload.unlink()
    try:
        os.symlink(target, payload)
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="linked|reparse|regular"):
        verify_source_snapshot(snapshot)


def test_snapshot_directory_tree_accepts_a_valid_concurrent_creator(
    tmp_path: Path,
    monkeypatch,
):
    parent = tmp_path / "artifacts"
    target = parent / "source-snapshots"
    real_mkdir = os.mkdir
    injected = False

    def inject_concurrent_creator(path, mode=0o777, *, dir_fd=None):
        nonlocal injected
        if Path(path) == parent and not injected:
            injected = True
            real_mkdir(path, mode, dir_fd=dir_fd)
            raise FileExistsError("injected concurrent directory winner")
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", inject_concurrent_creator)

    resolved = snapshot_module._ensure_real_directory_tree(target)

    assert injected is True
    assert resolved == target.resolve(strict=True)


@pytest.mark.parametrize("winner_valid", [True, False])
def test_source_snapshot_concurrent_winner_is_strictly_reused(
    tmp_path: Path,
    monkeypatch,
    winner_valid: bool,
):
    repo = _init_repo(tmp_path / "repo")
    _index_blob(repo, "100644", "gsdiff/module.py")
    commit = _commit(repo)

    def inject_winner(source, destination, **_identity):
        if winner_valid:
            shutil.copytree(source, destination)
        else:
            destination.mkdir()
            (destination / "source-snapshot.json").write_bytes(b"{}")
        raise FileExistsError("injected winner")

    monkeypatch.setattr(
        snapshot_module,
        "_promote_exact_directory_no_clobber",
        inject_winner,
    )

    if winner_valid:
        result = materialize_source_snapshot(
            repo,
            tmp_path / "artifacts",
            commit,
            (Path("gsdiff"),),
        )
        assert verify_source_snapshot(result) == result
        assert not list(result.root.parent.glob("*.tmp-*"))
    else:
        with pytest.raises(ValueError, match="manifest|snapshot"):
            materialize_source_snapshot(
                repo,
                tmp_path / "artifacts",
                commit,
                (Path("gsdiff"),),
            )
        winner = next((tmp_path / "artifacts/source-snapshots").glob("[0-9a-f]*"))
        assert (winner / "source-snapshot.json").read_bytes() == b"{}"


def test_source_snapshot_cleanup_never_deletes_swapped_unowned_tmp(
    tmp_path: Path,
    monkeypatch,
):
    repo = _init_repo(tmp_path / "repo")
    _index_blob(repo, "100644", "gsdiff/module.py")
    commit = _commit(repo)
    preserved = None

    def swap_tmp_then_fail(source, destination, **_identity):
        del destination
        nonlocal preserved
        preserved = source.with_name(source.name + ".preserved")
        os.rename(source, preserved)
        source.mkdir()
        (source / "replacement-victim.txt").write_text("victim", encoding="ascii")
        raise OSError("injected promotion failure")

    monkeypatch.setattr(
        snapshot_module,
        "_promote_exact_directory_no_clobber",
        swap_tmp_then_fail,
    )

    with pytest.raises(OSError, match="promotion"):
        materialize_source_snapshot(
            repo,
            tmp_path / "artifacts",
            commit,
            (Path("gsdiff"),),
        )

    assert preserved is not None
    replacement = preserved.with_name(preserved.name.removesuffix(".preserved"))
    assert (replacement / "replacement-victim.txt").read_text("ascii") == "victim"
    assert (preserved / "gsdiff/module.py").is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound promotion")
def test_source_snapshot_promotes_pinned_directory_not_swapped_path(
    tmp_path: Path,
    monkeypatch,
):
    repo = _init_repo(tmp_path / "repo")
    _index_blob(repo, "100644", "gsdiff/module.py", b"committed\n")
    commit = _commit(repo)
    swapped_paths: list[Path] = []

    def swap_after_handle_open(source, _destination):
        displaced = source.with_name(source.name + ".displaced")
        os.rename(source, displaced)
        source.mkdir()
        (source / "poison.txt").write_text("poison", encoding="ascii")
        swapped_paths.append(source)

    monkeypatch.setattr(
        artifact_persistence_module,
        "_handle_bound_promotion_barrier",
        swap_after_handle_open,
    )

    snapshot = materialize_source_snapshot(
        repo,
        tmp_path / "artifacts",
        commit,
        (Path("gsdiff"),),
    )

    assert verify_source_snapshot(snapshot) == snapshot
    assert (snapshot.root / "gsdiff/module.py").read_bytes() == b"committed\n"
    assert not (snapshot.root / "poison.txt").exists()
    assert len(swapped_paths) == 1
    assert (swapped_paths[0] / "poison.txt").read_text("ascii") == "poison"


def test_real_repo_snapshot_avoids_default_artifact_work_overlap():
    artifacts = REPO_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="task1-source-snapshot-",
        dir=artifacts,
    ) as temporary:
        artifact_root = Path(temporary)
        snapshot = materialize_source_snapshot(
            REPO_ROOT,
            artifact_root,
            commit,
            REPO_SOURCE_ROOTS,
        )
        inventory, digest = selected_source_evidence(snapshot)
        work = artifact_root / "work" / ("a" * 32)

        assert snapshot.root.parent == artifact_root / "source-snapshots"
        assert work.parent.parent == artifact_root
        assert snapshot.root not in work.parents
        assert work not in snapshot.root.parents
        assert inventory
        assert len(digest) == 64
