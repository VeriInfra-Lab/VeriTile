"""One exact integer rewrite and full-storage source/candidate CPU checks."""
import ast
import copy
from pathlib import Path
import torch
import check_contract as check

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / 'token_softmax_bloom_index_commuted.py'


def main():
    canonical = check.function(check.SOURCE, '_fwd_kernel_token_softmax')
    candidate = check.function(CANDIDATE, '_fwd_kernel_token_softmax')

    class Commute(ast.NodeTransformer):
        count = 0
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Name)
                    and node.left.id == 'cur_batch' and isinstance(node.right, ast.Name)
                    and node.right.id == 'stride_length'):
                node.left, node.right = node.right, node.left
                self.count += 1
            return node

    transform = Commute()
    assert ast.dump(transform.visit(copy.deepcopy(canonical))) == ast.dump(candidate)
    assert transform.count == 1
    source_kernel = check.Kernel()
    source_path = check.SOURCE
    check.SOURCE = CANDIDATE
    candidate_kernel = check.Kernel()
    check.SOURCE = source_path
    pairs = 0

    def compare(x, starts, lengths, maximum):
        nonlocal pairs
        before = [bytes(t.untyped_storage()) for t in (x, starts, lengths)]
        results = []
        for kernel in (source_kernel, candidate_kernel):
            storage = torch.full((2 * x.shape[1] + 7, 5), 73.)
            out = storage[3:3 + 2 * x.shape[1]:2, 1:1 + x.shape[0]].T
            untouched = storage.view(torch.int32).clone()
            wrapped = list(map(check.Tensor, (x, starts, lengths, out)))
            block = check.next_power(maximum)
            kernel[(starts.numel(), x.shape[0])](*wrapped, starts.stride(0), lengths.stride(0),
                *x.stride(), *out.stride(), BLOCK_SIZE=block,
                num_warps=16 if block >= 4096 else 8 if block >= 2048 else 4)
            touched = set()
            for start, count in zip(starts.tolist(), lengths.tolist()):
                for h in range(x.shape[0]):
                    for n in range(count):
                        touched.add(out.storage_offset()+h*out.stride(0)+(start+n)*out.stride(1))
            flat, initial = storage.view(torch.int32).flatten(), untouched.flatten()
            for i in range(flat.numel()):
                if i not in touched:
                    assert flat[i].item() == initial[i].item()
            for tensor, saved in zip((x, starts, lengths), before):
                assert bytes(tensor.untyped_storage()) == saved
            results.append(flat)
        assert torch.equal(*results)
        pairs += 1

    for width in (1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                  1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 32769):
        for exceptional in (False, True):
            base = torch.full((3, 2 * (width + 5) + 7), 91.)
            x = base[1:3, 3:3 + 2 * (width + 5):2]
            if exceptional:
                words = torch.tensor([0xff800000, 0, 0x80000000, 0x7f800000,
                    0x7fc00001, 0x7fa00003, 1, 0x7f7fffff], dtype=torch.int64).to(torch.int32)
                x.view(torch.int32).copy_(words.repeat((x.numel()+7)//8)[:x.numel()].reshape(x.shape))
            else:
                x.copy_(torch.linspace(-4, 4, x.numel()).reshape(x.shape))
            starts = torch.tensor([99, 2, 99, width+5, 99], dtype=torch.int32)[1:4:2]
            lengths = torch.tensor([99, width, 99, 0, 99], dtype=torch.int32)[1:4:2]
            compare(x, starts, lengths, width)
    # Read allocations may alias each other, including different source dtypes.
    words = torch.zeros(16, dtype=torch.int32)
    words[:4] = torch.tensor([0, 3, 5, 2], dtype=torch.int32)
    compare(words.view(torch.float32).reshape(2, 8), words[:4:2], words[1:4:2], 3)
    compare(torch.ones(2, 5), torch.tensor([5], dtype=torch.int32).expand(4),
            torch.tensor([0], dtype=torch.int32).expand(4), 8)
    print(f'PASS: exact one-product AST rewrite; {pairs} complete-storage bit comparisons.')
    print('All warp boundaries, exceptional words, offset/strided/transposed views, aliased read allocations, zero-stride empty metadata, and caller canaries; CPU only.')


if __name__ == '__main__':
    main()
