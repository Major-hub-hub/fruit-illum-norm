"""Run from the fruit-illum-norm repository root to integrate stress.py."""
import ast
from pathlib import Path

path = Path('src/evaluate.py')
stress_path = Path('src/preprocess/stress.py')
if not stress_path.is_file():
    raise SystemExit('Install the updated src/preprocess/stress.py first.')
source = path.read_text(encoding='utf-8')
new_import = 'from src.preprocess.stress import STRESSES, apply_stress'
if new_import in source:
    raise SystemExit('Stress import already present; inspect evaluate.py before proceeding.')
old_import = 'from src.preprocess.pipeline import PIPELINES'
old_constant = 'STRESSES = ("normal", "dark", "bright", "warm_cast")'
assert source.count(old_import) == 1 and source.count(old_constant) == 1, 'Unexpected evaluator version; no changes made.'
tree = ast.parse(source)
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TestPreprocessing']
assert len(classes) == 1, 'Expected one TestPreprocessing class.'
node = classes[0]
start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
lines = source.splitlines(keepends=True)
replacement = '''@dataclass(frozen=True)
class TestPreprocessing:
    """Apply illumination stress before model preprocessing."""

    condition: str
    stress: str

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        observed = apply_stress(rgb, self.stress)
        return PIPELINES[self.condition](observed)
'''
updated = ''.join(lines[:start]) + replacement + ''.join(lines[node.end_lineno:])
updated = updated.replace(old_import, old_import + '\n' + new_import, 1).replace(old_constant + '\n', '', 1)
ast.parse(updated)
backup = path.with_suffix('.py.bak')
if backup.exists():
    raise SystemExit(f'Backup already exists: {backup}; no changes made.')
backup.write_text(source, encoding='utf-8')
path.write_text(updated, encoding='utf-8')
print(f'Updated {path}; original backed up to {backup}')
