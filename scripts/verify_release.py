#!/usr/bin/env python3
"""Validate Alpha-Mind distributions before a PyPI upload."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from email.parser import BytesParser
from hashlib import sha256
from http.client import HTTPSConnection
from pathlib import Path, PurePosixPath
from tarfile import TarFile
from urllib.parse import quote
from zipfile import ZipFile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10 development environments
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
FINAL_VERSION_RE = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))?"
)
SOURCE_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$', re.MULTILINE)
REQUIRED_PYTHON_CLASSIFIERS = frozenset(
    {
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    }
)


@dataclass(frozen=True)
class ReleaseConfig:
    name: str
    version: str
    requires_python: str
    classifiers: frozenset[str]


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def project_configuration(root: Path = ROOT) -> ReleaseConfig:
    project = _read_toml(root / "pyproject.toml")["project"]
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml project must be a table")
    name = project.get("name")
    version = project.get("version")
    requires_python = project.get("requires-python")
    classifiers = project.get("classifiers")
    if name != "alpha-mind":
        raise ValueError(f"project.name must be 'alpha-mind', got {name!r}")
    if not isinstance(version, str) or FINAL_VERSION_RE.fullmatch(version) is None:
        raise ValueError(
            f"project.version must be a final X.Y.Z or X.Y.Z.N version, got {version!r}"
        )
    if requires_python != ">=3.9":
        raise ValueError(f"project.requires-python must be '>=3.9', got {requires_python!r}")
    if not isinstance(classifiers, list) or not all(isinstance(item, str) for item in classifiers):
        raise ValueError("project.classifiers must be a list of strings")
    missing = REQUIRED_PYTHON_CLASSIFIERS.difference(classifiers)
    if missing:
        raise ValueError(f"project.classifiers is missing {sorted(missing)}")

    source = (root / "src" / "alphamind" / "__init__.py").read_text(encoding="utf-8")
    match = SOURCE_VERSION_RE.search(source)
    if match is None:
        raise ValueError("src/alphamind/__init__.py does not declare __version__")
    if match.group(1) != version:
        raise ValueError(f"alphamind.__version__={match.group(1)!r} does not match {version!r}")
    return ReleaseConfig(name, version, requires_python, frozenset(classifiers))


def ensure_version_is_new_on_pypi(project: str, version: str) -> None:
    path = f"/pypi/{quote(project, safe='')}/{quote(version, safe='')}/json"
    connection = HTTPSConnection("pypi.org", timeout=20)
    try:
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        response.read()
    finally:
        connection.close()
    if response.status == 404:
        return
    if response.status != 200:
        raise OSError(f"PyPI version check for {project} failed with HTTP {response.status}")
    raise ValueError(f"{project} {version} already exists on PyPI and cannot be overwritten")


def validate_versions(
    root: Path = ROOT,
    tag: str | None = None,
    check_pypi: bool = False,
) -> ReleaseConfig:
    config = project_configuration(root)
    expected_tag = f"alpha-mind-v{config.version}"
    if tag is not None and tag != expected_tag:
        raise ValueError(f"release tag {tag!r} must equal {expected_tag!r}")
    if check_pypi:
        ensure_version_is_new_on_pypi(config.name, config.version)
    return config


def _single_header(path: Path, metadata, key: str) -> str:
    values = metadata.get_all(key, [])
    if len(values) != 1:
        raise ValueError(
            f"{path.name}: metadata has {len(values)} {key} headers; expected exactly one"
        )
    return values[0]


def _wheel_metadata(archive: ZipFile, path: Path, dist_info: str, filename: str):
    expected = f"{dist_info}/{filename}"
    matches = [
        name
        for name in archive.namelist()
        if PurePosixPath(name).parent == PurePosixPath(dist_info)
        and PurePosixPath(name).name == filename
    ]
    if matches != [expected]:
        raise ValueError(f"{path.name}: expected exactly one {expected}, found {matches}")
    return BytesParser().parsebytes(archive.read(expected))


def validate_wheel(path: Path, config: ReleaseConfig) -> None:
    distribution = config.name.replace("-", "_")
    expected_name = f"{distribution}-{config.version}-py3-none-any.whl"
    if path.name != expected_name:
        raise ValueError(f"wheel filename must be {expected_name}, got {path.name}")
    dist_info = f"{distribution}-{config.version}.dist-info"
    with ZipFile(path) as archive:
        names = archive.namelist()
        for required in ("alphamind/__init__.py", "alphamind/api.py", f"{dist_info}/RECORD"):
            if required not in names:
                raise ValueError(f"{path.name}: missing {required}")
        if any(name.startswith(("tests/", "notebooks/", "examples/")) for name in names):
            raise ValueError(f"{path.name}: wheel contains repository-only content")

        metadata = _wheel_metadata(archive, path, dist_info, "METADATA")
        expected = {
            "Name": config.name,
            "Version": config.version,
            "Requires-Python": config.requires_python,
        }
        for key, value in expected.items():
            actual = _single_header(path, metadata, key)
            if actual != value:
                raise ValueError(f"{path.name}: {key}={actual!r}, expected {value!r}")
        actual_classifiers = set(metadata.get_all("Classifier", []))
        missing = REQUIRED_PYTHON_CLASSIFIERS.difference(actual_classifiers)
        if missing:
            raise ValueError(f"{path.name}: wheel metadata is missing {sorted(missing)}")
        if not str(metadata.get_payload()).strip():
            raise ValueError(f"{path.name}: package description is empty")

        wheel = _wheel_metadata(archive, path, dist_info, "WHEEL")
        if _single_header(path, wheel, "Root-Is-Purelib") != "true":
            raise ValueError(f"{path.name}: Alpha-Mind must be a pure-Python wheel")
        if wheel.get_all("Tag", []) != ["py3-none-any"]:
            raise ValueError(f"{path.name}: wheel tag must be py3-none-any")
        source = archive.read("alphamind/__init__.py").decode("utf-8")
        match = SOURCE_VERSION_RE.search(source)
        if match is None or match.group(1) != config.version:
            raise ValueError(f"{path.name}: packaged alphamind.__version__ does not match")


def validate_sdist(path: Path, config: ReleaseConfig) -> None:
    distribution = config.name.replace("-", "_")
    expected_name = f"{distribution}-{config.version}.tar.gz"
    if path.name != expected_name:
        raise ValueError(f"sdist filename must be {expected_name}, got {path.name}")
    prefix = PurePosixPath(f"{distribution}-{config.version}")
    with TarFile.open(path, "r:gz") as archive:
        names = {PurePosixPath(member.name) for member in archive.getmembers() if member.isfile()}
        required = {
            prefix / "pyproject.toml",
            prefix / "README.md",
            prefix / "LICENSE.txt",
            prefix / "src/alphamind/__init__.py",
        }
        missing = required.difference(names)
        if missing:
            raise ValueError(f"{path.name}: missing {sorted(map(str, missing))}")
        pyproject_member = archive.getmember(str(prefix / "pyproject.toml"))
        stream = archive.extractfile(pyproject_member)
        if stream is None:
            raise ValueError(f"{path.name}: cannot read pyproject.toml")
        project = tomllib.loads(stream.read().decode("utf-8"))["project"]
        if project.get("name") != config.name or project.get("version") != config.version:
            raise ValueError(f"{path.name}: packaged project identity does not match")


def validate_release(
    dist_dir: Path,
    root: Path = ROOT,
    tag: str | None = None,
    check_pypi: bool = False,
) -> list[str]:
    config = validate_versions(root, tag, check_pypi)
    distribution = config.name.replace("-", "_")
    wheel = dist_dir / f"{distribution}-{config.version}-py3-none-any.whl"
    sdist = dist_dir / f"{distribution}-{config.version}.tar.gz"
    actual = sorted(path.name for path in dist_dir.iterdir() if path.is_file())
    expected = sorted([wheel.name, sdist.name])
    if actual != expected:
        raise ValueError(f"release artifacts must be exactly {expected}, found {actual}")
    validate_wheel(wheel, config)
    validate_sdist(sdist, config)
    return [f"{sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in (wheel, sdist)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--check-pypi", action="store_true")
    parser.add_argument("--version-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.version_only:
            print(validate_versions(tag=args.tag, check_pypi=args.check_pypi).version)
            return 0
        if args.dist_dir is None:
            parser.error("--dist-dir is required unless --version-only is used")
        manifest = validate_release(args.dist_dir, tag=args.tag, check_pypi=args.check_pypi)
    except (OSError, ValueError) as error:
        print(f"release verification failed: {error}", file=sys.stderr)
        return 1
    print("\n".join(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
