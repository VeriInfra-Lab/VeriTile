"""Actual source-AST kernel calls with shared input storage and guarded outputs.
This checks addressing and mathematical smoke cases on CPU, not GPU numerics.
"""
import numpy as np
import torch
import warnings
import check_source as source


def window(count, offset=5):
    data = np.full(count + offset + 7, -771.25, dtype=np.float32)
    writes = np.zeros(data.size, dtype=np.int64)
    return source.base.Pointer(data, offset, writes=writes), data, writes


def check():
    ns, _, _ = source.environment()
    forward = ns['_layer_norm_forward_kernel']
    backward = ns['_layer_norm_backward_kernel']
    cases = [(1, 1, 3), (2, 3, 5), (5, 7, 2), (4, 16, 3)]
    for rows, cols, programs in cases:
        group = (rows + programs - 1) // programs
        block = 1 << (cols - 1).bit_length()
        xp, xa, _ = window(rows * cols)
        rng = np.random.default_rng(rows * 100 + cols)
        xa[xp.offset:xp.offset + rows * cols] = rng.normal(size=rows * cols)
        original = xa.copy()
        # W aliases the first X row; B aliases W at precisely the same pointer.
        wp = source.base.Pointer(xa, xp.offset)
        bp = source.base.Pointer(xa, xp.offset)
        yp, ya, yw = window(rows * cols, 3)
        mp, ma, mw = window(rows, 2)
        rp, ra, rw = window(rows, 4)
        for pid in range(rows):
            forward.tl.row = pid
            forward.fn(yp, cols, xp, cols, wp, 1, bp, 1,
                       mp, 1, rp, 1, cols, 0.03125, block)
        x = torch.tensor(original[xp.offset:xp.offset + rows * cols].reshape(rows, cols), requires_grad=True)
        w = torch.tensor(original[xp.offset:xp.offset + cols], requires_grad=True)
        b = w.detach().clone().requires_grad_()
        reference = torch.nn.functional.layer_norm(x, (cols,), w, b, 0.03125)
        np.testing.assert_allclose(ya[yp.offset:yp.offset + rows * cols].reshape(rows, cols),
                                   reference.detach().numpy(), rtol=3e-5, atol=3e-6)
        # dY shares the exact X storage, permitted by the read-only contract.
        dxp, dxa, dxw = window(rows * cols, 6)
        dwp, dwa, dww = window(programs * cols, 7)
        dbp, dba, dbw = window(programs * cols, 8)
        for pid in range(programs):
            backward.tl.row = pid
            backward.fn(xp, wp, mp, rp, dxp, dwp, dbp, xp,
                        cols, cols, cols, cols, cols, rows, cols,
                        group, block, np.float32)
        gradient = torch.tensor(original[xp.offset:xp.offset + rows * cols].reshape(rows, cols))
        targets = torch.autograd.grad(reference, (x, w, b), gradient)
        actual = (dxa[dxp.offset:dxp.offset + rows * cols].reshape(rows, cols),
                  dwa[dwp.offset:dwp.offset + programs * cols].reshape(programs, cols).sum(0),
                  dba[dbp.offset:dbp.offset + programs * cols].reshape(programs, cols).sum(0))
        for got, expected in zip(actual, targets):
            np.testing.assert_allclose(got, expected.numpy(), rtol=6e-4, atol=2e-5)
        for ptr, data, writes, size in [(yp, ya, yw, rows * cols), (mp, ma, mw, rows),
                                       (rp, ra, rw, rows), (dxp, dxa, dxw, rows * cols),
                                       (dwp, dwa, dww, programs * cols), (dbp, dba, dbw, programs * cols)]:
            active = slice(ptr.offset, ptr.offset + size)
            assert np.all(writes[active] == 1)
            assert np.all(writes[:ptr.offset] == 0) and np.all(writes[ptr.offset + size:] == 0)
            assert np.all(data[:ptr.offset] == -771.25) and np.all(data[ptr.offset + size:] == -771.25)
        assert np.array_equal(xa, original)
    # Contiguous singleton views retain noncanonical strides through the
    # actual wrapper. Inactive backward prelude products may wrap int64.
    for stride in (0, 2**62, 2**63 - 1):
        view = torch.as_strided(torch.tensor([2.0]), (1, 1), (stride, stride))
        assert view.is_contiguous()
        assert view.contiguous().view(-1, 1).stride() == (stride, stride)
        ns, sm, _ = source.environment()
        sm.multi_processor_count = 3
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            y, saved, mean, rstd, _, _ = ns['layer_norm_forward'](
                source.Tensor(view), source.Tensor(torch.ones(1)),
                source.Tensor(torch.zeros(1)), 0.25)
            assert saved.stride(0) == stride
            dx, dw, db = ns['layer_norm_backward'](
                source.Tensor(view), saved, source.Tensor(torch.ones(1)),
                source.Tensor(torch.zeros(1)), mean, rstd)
        torch.testing.assert_close(y.value, torch.zeros((1, 1)), rtol=0, atol=0)
        torch.testing.assert_close(dx.value, torch.zeros((1, 1)), rtol=0, atol=0)
        torch.testing.assert_close(dw.value, torch.zeros(1), rtol=0, atol=0)
        torch.testing.assert_close(db.value, torch.full((1,), 2.0), rtol=0, atol=0)
        assert torch.equal(view, torch.full((1, 1), 2.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        assert np.int64(2) * np.int64(2**62) == np.int64(-(2**63))
    print('PASS: retained singleton strides 0, 2^62, 2^63-1; actual forward/backward source; inactive signed64 wrap')
    print(f'PASS: {len(cases)} actual-source kernel cases; nonzero offsets, X/W/B/dY aliases, empty groups, all six output frames')
    print('CPU AST execution with adapted primitives; no native GPU numerical conformance claim.')


if __name__ == '__main__':
    check()
