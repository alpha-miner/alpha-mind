from __future__ import annotations

import io
import re
import sys
from hashlib import sha256
from pathlib import Path
from tarfile import TarFile, TarInfo
from zipfile import ZipFile

import pytest
from scripts import verify_release

ROOT = Path(__file__).resolve().parents[1]


def _add_tar_text(archive: TarFile, name: str, text: str) -> None:
    payload = text.encode("utf-8")
    info = TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _write_distributions(directory: Path, config, include_python_314=True):
    distribution = config.name.replace("-", "_")
    wheel = directory / f"{distribution}-{config.version}-py3-none-any.whl"
    sdist = directory / f"{distribution}-{config.version}.tar.gz"
    dist_info = f"{distribution}-{config.version}.dist-info"
    classifiers = sorted(config.classifiers)
    if not include_python_314:
        classifiers.remove("Programming Language :: Python :: 3.14")
    metadata = "\n".join(
        [
            "Metadata-Version: 2.3",
            f"Name: {config.name}",
            f"Version: {config.version}",
            f"Requires-Python: {config.requires_python}",
            *(f"Classifier: {classifier}" for classifier in classifiers),
            "",
            "Alpha-Mind package documentation",
        ]
    )
    with ZipFile(wheel, "w") as archive:
        archive.writestr("alphamind/__init__.py", f'__version__ = "{config.version}"\n')
        archive.writestr("alphamind/api.py", "# Public API\n")
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")

    prefix = f"{distribution}-{config.version}"
    with TarFile.open(sdist, "w:gz") as archive:
        _add_tar_text(
            archive,
            f"{prefix}/pyproject.toml",
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        )
        _add_tar_text(archive, f"{prefix}/README.md", "# Alpha-Mind\n")
        _add_tar_text(archive, f"{prefix}/LICENSE.txt", "MIT\n")
        _add_tar_text(
            archive,
            f"{prefix}/src/alphamind/__init__.py",
            f'__version__ = "{config.version}"\n',
        )
    return wheel, sdist


def test_project_and_source_versions_match():
    config = verify_release.project_configuration()
    source = (ROOT / "src/alphamind/__init__.py").read_text(encoding="utf-8")
    assert config.name == "alpha-mind"
    assert config.requires_python == ">=3.9"
    assert f'__version__ = "{config.version}"' in source
    assert verify_release.REQUIRED_PYTHON_CLASSIFIERS <= config.classifiers
    assert verify_release.validate_versions(tag=f"alpha-mind-v{config.version}") == config


def test_wrong_release_tag_is_rejected():
    config = verify_release.project_configuration()
    with pytest.raises(ValueError, match="release tag"):
        verify_release.validate_versions(tag=f"alpha-mind-v{config.version}.1")


def test_pypi_availability_is_checked_when_requested(monkeypatch):
    calls = []
    monkeypatch.setattr(
        verify_release,
        "ensure_version_is_new_on_pypi",
        lambda name, version: calls.append((name, version)),
    )
    config = verify_release.validate_versions(check_pypi=True)
    assert calls == [(config.name, config.version)]


@pytest.mark.parametrize(("status", "error"), [(404, None), (200, ValueError), (503, OSError)])
def test_pypi_response_handling(monkeypatch, status, error):
    class Response:
        def __init__(self, status):
            self.status = status

        def read(self):
            return b""

    class Connection:
        def __init__(self, host, timeout):
            assert host == "pypi.org"
            assert timeout == 20
            self.closed = False

        def request(self, method, path, headers):
            assert method == "GET"
            assert "/pypi/alpha-mind/" in path

        def getresponse(self):
            return Response(status)

        def close(self):
            self.closed = True

    monkeypatch.setattr(verify_release, "HTTPSConnection", Connection)
    if error is None:
        verify_release.ensure_version_is_new_on_pypi("alpha-mind", "0.4.0")
    else:
        with pytest.raises(error):
            verify_release.ensure_version_is_new_on_pypi("alpha-mind", "0.4.0")


def test_distribution_validation_and_hash_manifest(tmp_path):
    config = verify_release.project_configuration()
    artifacts = _write_distributions(tmp_path, config)
    manifest = verify_release.validate_release(tmp_path)
    assert manifest == [
        f"{sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in artifacts
    ]


def test_distribution_validation_rejects_unexpected_files(tmp_path):
    config = verify_release.project_configuration()
    _write_distributions(tmp_path, config)
    (tmp_path / "extra.whl").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="release artifacts must be exactly"):
        verify_release.validate_release(tmp_path)


def test_distribution_validation_requires_python_314_metadata(tmp_path):
    config = verify_release.project_configuration()
    _write_distributions(tmp_path, config, include_python_314=False)
    with pytest.raises(ValueError, match="wheel metadata is missing"):
        verify_release.validate_release(tmp_path)


def test_version_only_cli(capsys):
    config = verify_release.project_configuration()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", ["verify_release.py", "--version-only"])
        assert verify_release.main() == 0
    assert capsys.readouterr().out.strip() == config.version


def test_release_workflow_gates_publication():
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    jobs = re.findall(r"^  ([a-z-]+):$", workflow.split("jobs:\n", 1)[1], re.MULTILINE)
    assert jobs == [
        "prepare-release",
        "build-distributions",
        "test-wheel",
        "test-sdist",
        "publish-pypi",
    ]
    publish = workflow.split("  publish-pypi:\n", 1)[1]
    assert "needs: [build-distributions, test-wheel, test-sdist]" in publish
    assert "if: github.event_name == 'push' && github.ref_type == 'tag'" in publish
    assert "name: pypi" in publish
    assert "id-token: write" in publish
    assert "sha256sum --check ../release-manifest.txt" in publish
    assert publish.count("uses: pypa/gh-action-pypi-publish@") == 1
    assert "packages-dir: dist" in publish
    assert "PYPI_API_TOKEN" not in workflow
    assert 'python-version: ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]' in workflow
    assert 'python-version: "3.14"' in workflow
    assert "python -m pytest -q -o pythonpath= tests" in workflow
    assert 'test "$(git rev-parse FETCH_HEAD)" = "${GITHUB_SHA}"' in workflow
    assert 'test "$(git cat-file -t "${GITHUB_REF}")" = "tag"' in workflow


def test_release_actions_are_pinned_to_full_commits():
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    actions = re.findall(r"uses:\s*([^\s]+)", workflow)
    assert actions
    for action in actions:
        name, reference = action.rsplit("@", 1)
        assert name
        assert re.fullmatch(r"[0-9a-f]{40}", reference), action


def test_ci_covers_python_314():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'python-version: ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]' in workflow
