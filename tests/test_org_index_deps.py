"""Dependency harvest in org_index: declaration parsers and the consumers query."""
from __future__ import annotations

from cairn import org_index as oi


def test_package_xml_records_provides_and_every_depend_tag():
    xml = """<package format="3"><name>nav_node</name>
      <depend>rclpy</depend><build_depend>vehicle_msgs</build_depend>
      <exec_depend>payload</exec_depend><test_depend>pytest</test_depend>
      <depend>nav_node</depend></package>"""
    got = oi.parse_package_xml(xml)
    assert ("provides", "nav_node", None) in got
    assert ("ros", "vehicle_msgs", "build_depend") in got
    assert ("ros", "payload", "exec_depend") in got
    assert ("ros", "pytest", "test_depend") in got
    # A package never depends on itself.
    assert ("ros", "nav_node", "depend") not in got


def test_package_xml_that_does_not_parse_yields_nothing():
    assert oi.parse_package_xml("<package><name>x</name>") == []


def test_gitmodules_keeps_owner_so_third_party_is_distinguishable():
    text = """[submodule "a"]
\tpath = src/intern/nav-msgs
\turl = git@github.com:example-org/nav-msgs.git
[submodule "b"]
\tpath = src/extern/matplotlib-cpp
\turl = https://github.com/lava/matplotlib-cpp.git
"""
    assert oi.parse_gitmodules(text) == [
        ("submodule", "example-org/nav-msgs", "src/intern/nav-msgs"),
        ("submodule", "lava/matplotlib-cpp", "src/extern/matplotlib-cpp"),
    ]


def test_requirements_splits_index_packages_from_git_refs():
    text = """numpy==1.26  # pinned
Foo_Bar[extra]>=2
-r other.txt
nav-msgs @ git+https://github.com/example-org/nav-msgs.git@v1.4.0
"""
    got = oi.parse_requirements(text)
    assert ("python", "numpy", "==1.26") in got
    assert ("python", "foo-bar", ">=2") in got
    assert ("git", "example-org/nav-msgs", "v1.4.0") in got
    assert not any(t == "nav-msgs" for k, t, _ in got if k == "python")


def test_pyproject_reads_project_optional_and_poetry_dependencies():
    text = """[project]
dependencies = ["requests>=2", "lib_slow_sync==0.3.1"]
[project.optional-dependencies]
dev = ["pytest"]
[tool.poetry.dependencies]
python = "^3.10"
PyYAML = "*"
"""
    targets = {t for k, t, _ in oi.parse_pyproject(text) if k == "python"}
    assert targets == {"requests", "lib-slow-sync", "pytest", "pyyaml"}


def test_repos_file_and_dockerfile_git_refs():
    repos = """repositories:
  src/nav: {type: git, url: https://github.com/example-org/nav.git, version: develop}
"""
    assert oi.parse_repos_file(repos) == [("git", "example-org/nav", "develop")]
    docker = "RUN git clone -b feat/x git@github.com:example-org/ros2-app.git /src\n"
    assert oi._git_refs(docker) == [("git", "example-org/ros2-app", "feat/x")]


def test_dep_file_kind_selects_declaration_files_only():
    assert oi._dep_file_kind("package.xml") == "package.xml"
    assert oi._dep_file_kind("requirements-dev.txt") == "requirements"
    assert oi._dep_file_kind("Dockerfile.jetson") == "dockerfile"
    assert oi._dep_file_kind("deps.repos") == "repos"
    assert oi._dep_file_kind("README.md") is None


def test_consumers_matches_exactly_despite_underscores(tmp_path, capsys):
    db = str(tmp_path / "idx.db")
    con = oi.connect(db)
    con.executescript(oi.DEPS_SCHEMA)
    con.executemany(
        "INSERT INTO dependencies(org, repo, path, kind, target, detail) VALUES (?,?,?,?,?,?)",
        [("o", "geo", ".gitmodules", "submodule", "o/nav-msgs", "src/msgs"),
         ("o", "nav", "a/package.xml", "ros", "vehicle_msgs", "depend"),
         ("o", "tool", "requirements.txt", "python", "lib-slow-sync", "==0.3")])
    con.commit()
    # '_' is a LIKE wildcard: a pattern match would wrongly hit the repo row.
    assert oi.consumers(db, "nav_msgs") == []
    assert [r[1] for r in oi.consumers(db, "nav-msgs")] == ["geo"]
    assert [r[1] for r in oi.consumers(db, "o/nav-msgs")] == ["geo"]
    assert [r[1] for r in oi.consumers(db, "vehicle_msgs")] == ["nav"]
    assert [r[1] for r in oi.consumers(db, "lib_slow_sync")] == ["tool"]


def test_comments_labels_and_project_urls_are_not_dependencies():
    assert oi.parse_requirements("# see https://github.com/x/y\nrequests>=2\n") == [
        ("python", "requests", ">=2")]
    assert oi.parse_requirements("git+https://github.com/o/r.git#egg=r\n") == [
        ("git", "o/r", None)]
    assert oi._git_refs('LABEL source="https://github.com/o/self"\n'
                        "RUN git clone https://github.com/o/dep\n") == [("git", "o/dep", None)]
    toml = ('[project]\nname="self"\ndependencies=["requests>=2"]\n'
            '[project.urls]\nHomepage = "https://github.com/o/self"\n')
    assert oi.parse_pyproject(toml) == [("python", "requests", ">=2")]


def test_pyproject_vcs_dependency_fields_are_still_recorded():
    toml = ('[project]\nname="a"\ndependencies=["lib @ git+https://github.com/o/lib.git@v1"]\n'
            '[tool.uv.sources]\nother = { git = "https://github.com/o/other" }\n')
    targets = {t for k, t, _ in oi.parse_pyproject(toml) if k == "git"}
    assert targets == {"o/lib", "o/other"}


def test_deps_without_configured_orgs_fails_loudly(monkeypatch, tmp_path):
    import pytest
    from cairn import config
    monkeypatch.setattr(config, "ORG_INDEX_ORGS", [])
    monkeypatch.setattr("sys.argv", ["org_index", "--db", str(tmp_path / "x.db"), "deps"])
    with pytest.raises(SystemExit) as e:
        oi.main()
    assert "no orgs" in str(e.value)
