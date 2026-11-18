# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the specialized-directory rules: imports and mutations match
EXCEPTIONS.py (Rule 1), declarations are import-light, and every directory
has mirrored tests (Rule 4)."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from vllm.specialized import _lint
from vllm.specialized._routing import LANGUAGE_MODEL

DIRS = _lint.directories()
REPO = _lint.VLLM_ROOT.parent


def _id(d: Path) -> str:
    return str(d.relative_to(_lint.VLLM_ROOT / "models"))


@pytest.mark.parametrize("directory", DIRS, ids=_id)
def test_rule1_matches_exceptions(directory):
    found = _lint.lint(directory)
    exceptions, mutations = _lint.declared(directory)
    assert set(found.imports) == set(exceptions), (
        "imports of core internals differ from EXCEPTIONS.py: "
        f"new={sorted(set(found.imports) - set(exceptions))} "
        f"stale={sorted(set(exceptions) - set(found.imports))}"
    )
    assert set(found.mutations) == set(mutations), (
        f"new={sorted(set(found.mutations) - set(mutations))} "
        f"stale={sorted(set(mutations) - set(found.mutations))}"
    )


@pytest.mark.parametrize("directory", DIRS, ids=_id)
def test_declaration_is_import_light(directory):
    module = _lint.module_of(directory / "__init__.py")
    family = ".".join(module.split(".")[:3])
    code = f"""
import sys, torch
import {module} as d
from vllm.specialized.declare import Support
assert isinstance(d.SUPPORTS, Support)
heavy = [m for m in sys.modules if m.startswith((
    "vllm.model_executor", "vllm.distributed", "vllm.v1", "cutlass", "flashinfer",
    "{family}.common", "{family}.nvidia.model", "{family}.amd.model"))]
assert not heavy, heavy
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def _class_exists(directory: Path, ref: str) -> bool:
    module, _, cls = ref.partition(":")
    path = directory / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text())
    return any(isinstance(n, ast.ClassDef) and n.name == cls for n in tree.body)


@pytest.mark.parametrize("directory", DIRS, ids=_id)
def test_model_refs_and_routing_entry(directory):
    module = _lint.module_of(directory / "__init__.py")
    d = importlib.import_module(module)
    refs = d.MODEL.values() if isinstance(d.MODEL, dict) else [d.MODEL]
    refs = [*refs, *getattr(d, "DRAFT_MODEL", {}).values()]
    for ref in refs:
        assert ref == LANGUAGE_MODEL or _class_exists(directory, ref), ref
    family = ".".join(module.split(".")[:3])
    table = importlib.import_module(f"{family}.routing").SPECIALIZED
    listed = {f"{family}.{rel}" for rels in table.values() for rel in rels}
    assert module in listed, f"{module} missing from {family}/routing.py"


@pytest.mark.parametrize("directory", DIRS, ids=_id)
def test_rule4_tests_mirror_source(directory):
    rel = directory.relative_to(_lint.VLLM_ROOT)
    tests = REPO / "tests" / rel
    assert any(tests.glob("test_*.py")), f"no tests under {tests}"
    # Nothing outside a directory (and its tests) imports into it.
    module = _lint.module_of(directory / "__init__.py")
    out = subprocess.run(
        ["git", "grep", "-l", module, "--", "vllm"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.split()
    outside = [p for p in out if not p.startswith(str(Path("vllm") / rel))]
    assert not outside, outside
