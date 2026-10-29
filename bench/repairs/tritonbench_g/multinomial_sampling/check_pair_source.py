"""Check that the existing candidate only specializes and unrolls the token loop."""
import ast
import copy
from pathlib import Path

SOURCE = Path(__file__).with_name('multinomial_sampling_repaired.py')


def check_pair(source_text=None):
    text = SOURCE.read_text() if source_text is None else source_text
    module = ast.parse(text)
    original = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_multinomial_sampling_kernel')
    candidate = copy.deepcopy(next(n for n in module.body if isinstance(n, ast.FunctionDef)
                                  and n.name == '_multinomial_sampling_unrolled_kernel'))
    dump = lambda n: ast.dump(n, include_attributes=False)
    loops = [n for n in candidate.body if isinstance(n, ast.For)]
    reference = [n for n in original.body if isinstance(n, ast.For)]
    if len(loops) != 1 or len(reference) != 1:
        raise ValueError('expected one complete token loop in each source')
    loop, old = loops[0], reference[0]
    if dump(loop.iter) != dump(ast.parse('tl.static_range(tl.cdiv(num_tokens, BLOCK_N))', mode='eval').body):
        raise ValueError('candidate token count differs')
    if dump(loop.body[0]) != dump(ast.parse('b_idx = b_idx * BLOCK_N').body[0]):
        raise ValueError('candidate token index differs')
    if dump(old.iter) != dump(ast.parse('range(0, num_tokens, BLOCK_N)', mode='eval').body):
        raise ValueError('original loop differs')
    loop.iter = copy.deepcopy(old.iter)
    loop.body = loop.body[1:]
    arg = next(a for a in candidate.args.args if a.arg == 'num_tokens')
    if dump(arg.annotation) != dump(ast.parse('tl.constexpr', mode='eval').body):
        raise ValueError('candidate token count must be constexpr')
    arg.annotation = copy.deepcopy(next(a for a in original.args.args if a.arg == 'num_tokens').annotation)
    candidate.name = original.name
    if dump(candidate) != dump(original):
        raise ValueError('unexpected candidate change outside token specialization/unrolling')
    return {'status': 'PASS', 'scope': 'Complete existing JIT pair, including all decorators, arguments and statements; no numerical claim.'}


if __name__ == '__main__':
    import json
    print(json.dumps(check_pair(), sort_keys=True))
