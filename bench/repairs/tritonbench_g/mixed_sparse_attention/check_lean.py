"""Build the five public roots and all local imports in dependency order."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[4]
base = Path(__file__).resolve().parent
modules = {str(p.relative_to(root).with_suffix("")).replace("/", "."): p
           for p in base.rglob("*.lean")}
seen = set()
order = []


def visit(name):
    if name in seen:
        return
    seen.add(name)
    for line in modules[name].read_text().splitlines():
        if line.startswith("import ") and line.split()[1] in modules:
            visit(line.split()[1])
    order.append(modules[name])


for entry in ("Kernels", "Source", "Contract", "Correct", "FPEquiv"):
    visit(str((base / entry).relative_to(root)).replace("/", "."))
for source in order:
    relative = source.relative_to(root)
    output = root / ".lake/build/lib/lean" / relative.with_suffix(".olean")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["lake", "env", "lean", "-s", "32768", "--root=" + str(root), str(source),
                    "-o", str(output)], cwd=root, check=True)
    print(relative, "PASS", flush=True)
print(len(order), "Lean modules PASS")
