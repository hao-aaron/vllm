# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rule 1 lints for specialized directories: imports and mutations, by AST.

`python -m vllm.specialized._lint` prints each directory's current findings in
EXCEPTIONS.py form.
"""

import ast
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

VLLM_ROOT = Path(__file__).resolve().parents[1]
# Core modules every specialized directory may import.
ALLOWED = (
    "vllm.specialized.declare",
    "vllm.specialized.model",
    "vllm.specialized.services",
    "vllm.specialized",
    "vllm.models.common.ops",
    "vllm.logger",
)
# Attributes that, assigned on an object the directory did not create, change
# how shared code behaves.
PATCHED_ATTRS = frozenset({"forward", "quant_method", "process_weights_after_loading"})


@dataclass
class Findings:
    imports: dict[str, list[str]] = field(default_factory=dict)  # module -> files
    mutations: dict[str, list[str]] = field(default_factory=dict)  # what -> files


def directories() -> list[Path]:
    return sorted(
        p
        for p in VLLM_ROOT.glob("models/*/*/specialized/*")
        if p.is_dir() and (p / "__init__.py").exists()
    )


def module_of(path: Path) -> str:
    rel = path.relative_to(VLLM_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    base = package.split(".")
    base = base[: len(base) - node.level + 1]
    return ".".join(base + ([node.module] if node.module else []))


def _allowed(module: str, directory: str, family: str) -> bool:
    if module.startswith(ALLOWED) and not module.startswith("vllm.specialized._"):
        return True
    if module == directory or module.startswith(directory + "."):
        return True
    # The family's general tier, read-only; never another specialized directory.
    if module.startswith(family + ".") or module == family:
        return ".specialized." not in module[len(family) :] + "."
    return not module.startswith("vllm")


def lint(directory: Path) -> Findings:
    dmod = module_of(directory / "__init__.py")
    family = ".".join(dmod.split(".")[:3])
    out = Findings()
    for f in sorted(directory.rglob("*.py")):
        if f.name == "EXCEPTIONS.py":
            continue
        mod = module_of(f)
        package = mod if f.name == "__init__.py" else mod.rsplit(".", 1)[0]
        tree = ast.parse(f.read_text(), str(f))
        rel = str(f.relative_to(directory))
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
                imported_names |= {
                    (a.asname or a.name).split(".")[0] for a in node.names
                }
            elif isinstance(node, ast.ImportFrom):
                base = _resolve(node, package)
                # `from pkg import submodule` imports pkg.submodule.
                names = []
                for a in node.names:
                    sub = f"{base}.{a.name}"
                    names.append(sub if _is_module(sub) else base)
                    imported_names.add(a.asname or a.name)
            for name in names:
                if not _allowed(name, dmod, family):
                    out.imports.setdefault(name, []).append(rel)
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                if not isinstance(t, ast.Attribute):
                    continue
                root = t.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                on_module = isinstance(root, ast.Name) and root.id in imported_names
                if t.attr in PATCHED_ATTRS or on_module:
                    what = f"{ast.unparse(t)} = ..."
                    if t.attr in PATCHED_ATTRS and ast.unparse(t) == f"self.{t.attr}":
                        continue  # its own attribute
                    out.mutations.setdefault(what, []).append(rel)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in PATCHED_ATTRS
            ):
                out.mutations.setdefault(ast.unparse(node), []).append(rel)
    return out


def _is_module(name: str) -> bool:
    if not name.startswith("vllm."):
        return False
    rel = Path(*name.split("."))
    root = VLLM_ROOT.parent
    return (root / rel).is_dir() or (root / rel).with_suffix(".py").exists()


def declared(directory: Path) -> tuple[dict[str, str], dict[str, str]]:
    path = directory / "EXCEPTIONS.py"
    if not path.exists():
        return {}, {}
    spec = importlib.util.spec_from_file_location("_exceptions", path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return dict(getattr(m, "EXCEPTIONS", {})), dict(getattr(m, "MUTATIONS", {}))


def main() -> None:
    for d in directories():
        f = lint(d)
        print(
            f"# {d.relative_to(VLLM_ROOT.parent)}: {len(f.imports)} imports, "
            f"{len(f.mutations)} mutations"
        )
        if "-v" in sys.argv:
            for k, v in sorted(f.imports.items()):
                print(f"  import {k}  <- {sorted(set(v))[:3]}")
            for k, v in sorted(f.mutations.items()):
                print(f"  mutate {k}  <- {sorted(set(v))[:3]}")


if __name__ == "__main__":
    main()
