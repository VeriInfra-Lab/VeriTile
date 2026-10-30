"""Full-body normalized AST correspondence and pinned offline compiler audit."""
import ast
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
OUT = HERE / 'kernel-evidence'
SOURCE = HERE / 'fused_recurrent_hgrn_repaired.py'
QUOTE = HERE / 'Kernels.lean'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(SOURCE) == 'f920993bd8b61bc71450ab7a1775ab8dc4c350435e81b83728834df1af922cde'

class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id='_i' if node.id == '_' else node.id, ctx=node.ctx), node)
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'range' and len(node.args) == 2:
            node.args.append(ast.Constant(1))
        return node
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            return [self.visit(ast.Assign(targets=[t], value=v)) for t, v in zip(node.targets[0].elts, node.value.elts)]
        return self.generic_visit(node)

def normalized(body):
    return ast.dump(Normalize().visit(ast.Module(body=body, type_ignores=[])), include_attributes=False)

bodies = re.findall(r'\(triton \{(.*?)\}\)\.surfaceBody', QUOTE.read_text(), re.S)
assert len(bodies) == 2
functions = {n.name: n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef)}
comparisons = []
for phase, body in zip(('fwd', 'bwd'), bodies):
    body = body[body.index('  i_d ='):]
    body = re.sub(r'\$\(([^()]*)\)', r'\1', body)
    flags = {'U': 'USE_INITIAL_STATE', 'S': 'STORE_FINAL_STATE', 'V': 'USE_FINAL_GRAD', 'DO': 'do'}
    for old, new in flags.items():
        body = re.sub(r'\b'+old+r'\b', new, body)
    indent, lines = 0, []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('}'):
            indent -= 1
            line = line[1:].strip()
        if not line:
            continue
        opens = line.endswith('{')
        if opens:
            line = line[:-1].rstrip() + ':'
        lines.append('    '*indent + line)
        if opens:
            indent += 1
    assert indent == 0
    translated = '\n'.join(lines) + '\n'
    expected = normalized(functions[f'fused_recurrent_hgrn_{phase}_kernel'].body)
    actual = normalized(ast.parse(translated).body)
    assert actual == expected, (phase, translated, actual, expected)
    comparisons.append(dict(phase=phase, complete_body_equal=True,
        normalized_ast_sha256=hashlib.sha256(actual.encode()).hexdigest(),
        normalizations=['split tuple program_id assignment', 'range default step 1',
            'unused loop variable alpha rename', 'constexpr splice substitution',
            'flag and do region alpha rename', 'elif as nested else/if', 'parenthesis syntax']))

compiler_checks = []
for file in ('profiles.json', 'loop-profiles.json'):
    profiles = json.loads((OUT/file).read_text())
    assert profiles['actual_binder'] and profiles['actual_wrapper_arguments']
    assert not profiles['gpu_execution'] and not profiles['numerical_experiments']
    for record in profiles['results']:
        assert record['status'] == 'compiled' and record['source_sha256'] == sha(SOURCE)
        for artifact in record['artifacts']:
            assert sha(Path(artifact['path'])) == artifact['sha256']
        ptx = next(Path(a['path']).read_text() for a in record['artifacts'] if a['path'].endswith('.ptx'))
        assert '.ftz' not in ptx
        constants = record['constants']
        t = int(constants['T'])
        phase = record['name'].split('_')[0]
        if t > 0 and phase == 'fwd':
            assert 'fma.rn.f32' in ptx
        if phase == 'bwd' and t in (2, 3, 4):
            if record['name'] == 'bwd_t3_d65_u1_v1_bd128_w1':
                assert 'fma.rn.f32' not in ptx and 'Inner Loop Header' in ptx
            else:
                assert 'fma.rn.f32' in ptx, record['name']
        if phase == 'bwd' and t in (17, 1024):
            assert 'fma.rn.f32' not in ptx and 'add.f32' in ptx and 'mul.f32' in ptx
        if phase == 'fwd' and t == 0 and constants['USE_INITIAL_STATE'] == 'True' and constants['STORE_FINAL_STATE'] == 'True':
            assert 'add.f32' in ptx and '0f00000000' in ptx
        compiler_checks.append(dict(name=record['name'], no_ftz=True,
            add=ptx.count('add.f32'), mul=ptx.count('mul.f32'), fma=ptx.count('fma.rn.f32')))
assert len(compiler_checks) == 64
audit = dict(source_sha256=sha(SOURCE), quotation_sha256=sha(QUOTE),
    complete_source_ast=comparisons, compiler_checks=compiler_checks,
    gpu_execution=False, native_numerical_experiment=False,
    compiler_correspondence='Profile observations; universal compiler reification remains an explicit model boundary.')
(OUT/'source-and-compiler-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print('PASS: both complete source bodies match the readable DSL; 64 pinned actual-binder profiles audited.')
