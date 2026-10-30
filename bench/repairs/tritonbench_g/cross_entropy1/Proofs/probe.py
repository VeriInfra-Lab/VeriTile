"""Actual Triton CPU interpreter, Torch autograd and bounded collective controls."""
import os
os.environ["TRITON_INTERPRET"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True
import concurrent.futures
import contextlib
import hashlib
import importlib.util
import itertools
import json
import pathlib
import shutil
import threading
import traceback
import types
import warnings

import torch
import triton

D = pathlib.Path(__file__).resolve().parents[1]
R = D.parents[3]
E = pathlib.Path("/tmp/veritile-cross-entropy1")
SOURCE = D / "cross_entropy1_conditional.py"
LOCK = threading.RLock()
ACTIVE_OUTPUT = None
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def info(t):
    return {"shape": list(t.shape), "stride": list(t.stride()), "offset": t.storage_offset(),
            "dtype": str(t.dtype), "bytes": t.untyped_storage().nbytes(), "is_neg": t.is_neg()}


def storage_bytes(t):
    return torch.empty(0, dtype=torch.uint8).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes(),), (1,)).clone()


def footprint(t):
    size = t.element_size()
    base = t.untyped_storage().data_ptr()
    return {base + size * (t.storage_offset() + sum(i * s for i, s in zip(index, t.stride()))) + b
            for index in itertools.product(*(range(n) for n in t.shape)) for b in range(size)}


class Collectives:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.gathered = {}
        self.reduced = {}
        self.events = []

    def group(self, rank):
        return types.SimpleNamespace(rank=rank, coordinator=self)

    def gather(self, output, tensor, rank):
        with self.condition:
            self.gathered[rank] = tensor.detach().clone()
            self.events.append({"rank": rank, "op": "all_gather_into_tensor", "input": info(tensor), "output": info(output)})
            self.condition.notify_all()
            assert self.condition.wait_for(lambda: len(self.gathered) == self.size, timeout=40)
            output.copy_(torch.stack([self.gathered[i] for i in range(self.size)]))

    def reduce(self, tensor, rank):
        with self.condition:
            self.reduced[rank] = tensor.detach().clone()
            self.events.append({"rank": rank, "op": "all_reduce_async", "input": info(tensor)})
            self.condition.notify_all()

        def wait():
            with self.condition:
                assert self.condition.wait_for(lambda: len(self.reduced) == self.size, timeout=40)
                tensor.copy_(torch.stack([self.reduced[i] for i in range(self.size)]).sum(dim=0))
                self.events.append({"rank": rank, "op": "wait"})
        return types.SimpleNamespace(wait=wait)


class TorchHost:
    cuda = types.SimpleNamespace(device=lambda *_: contextlib.nullcontext())

    def __init__(self, runner):
        self.runner = runner
        self.distributed = types.SimpleNamespace(
            get_world_size=lambda group: group.coordinator.size,
            get_rank=lambda group: group.rank,
            all_gather_into_tensor=lambda output, tensor, group: group.coordinator.gather(output, tensor, group.rank),
            all_reduce=lambda tensor, op, group, async_op: group.coordinator.reduce(tensor, group.rank),
            ReduceOp=types.SimpleNamespace(SUM="sum"))

    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, *args, **kwargs):
        out = torch.empty(*args, **kwargs)
        out.fill_(42.5)
        self.runner.allocations.append(out)
        return out

    def empty_like(self, *args, **kwargs):
        out = torch.empty_like(*args, **kwargs)
        out.fill_(42.5)
        self.runner.allocations.append(out)
        return out


class Kernel:
    def __init__(self, runner, kernel, name):
        self.runner, self.kernel, self.name = runner, kernel, name

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            names = self.kernel.fn.arg_names
            bound = dict(zip(names, args))
            bound.update(kwargs)
            shape = grid(bound) if callable(grid) else grid
            tensors = {name: value for name, value in bound.items() if isinstance(value, torch.Tensor)}
            self.runner.launch_tensors.append(tensors)
            self.runner.launches.append({"kernel": self.name, "grid": list(shape),
                "BLOCK_SIZE": kwargs["BLOCK_SIZE"], "SPLIT": kwargs.get("SPLIT"),
                "tensor_arguments": {name: info(value) for name, value in tensors.items()}})
            with LOCK:
                return self.kernel[grid](*args, **kwargs)
        return run


class Runner:
    def __init__(self, suffix):
        self.allocations, self.launches, self.launch_tensors = [], [], []
        spec = importlib.util.spec_from_file_location("ce_repair_" + suffix, SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.torch = TorchHost(self)
        for name in ["cross_entropy_fwd_kernel", "cross_entropy_bwd_kernel"]:
            setattr(self.module, name, Kernel(self, getattr(self.module, name), name))
        forward = self.module.CrossEntropyLoss.forward
        backward = self.module.CrossEntropyLoss.backward

        def capture_forward(ctx, *args):
            result = forward(ctx, *args)
            self.saved = ctx.to_save
            self.forward_result = result
            return result

        def capture_backward(ctx, grad):
            self.actual_upstream = grad
            self.backward_allocations_start = len(self.allocations)
            self.backward_result = backward(ctx, grad)
            return self.backward_result

        self.module.CrossEntropyLoss.forward = staticmethod(capture_forward)
        self.module.CrossEntropyLoss.backward = staticmethod(capture_backward)


def reference(x, labels, smoothing, scale, ignored, dy):
    x = x.detach().to(torch.float64)
    labels = labels.detach().clone()
    lse = torch.logsumexp(x, dim=1)
    selected = labels != ignored
    safe = torch.where(selected, labels, 0)
    loss = lse - (1 - smoothing) * x.gather(1, safe[:, None]).squeeze(1)
    loss = loss - smoothing * x.mean(dim=1) + scale * lse.square()
    loss = torch.where(selected, loss, 0.)
    grad = (1 + 2 * scale * lse[:, None]) * torch.softmax(x, dim=1) - smoothing / x.shape[1]
    grad.scatter_add_(1, safe[:, None], torch.full((x.shape[0], 1), -(1-smoothing), dtype=torch.float64))
    grad = grad * dy.detach().to(torch.float64)[:, None]
    grad = torch.where(selected[:, None], grad, 0.)
    return loss, grad


def run_case(name, logits, labels, *, smoothing=0., scale=0., ignored=-100,
             dy=None, inplace=False, sum_loss=False, group=None, expected=None, exceptional=False):
    if dy is None:
        dy = torch.ones(logits.shape[0])
    inputs = [logits, labels, dy]
    before = {t.untyped_storage().data_ptr(): (t, storage_bytes(t)) for t in inputs}
    labels_before = labels.detach().resolve_neg().clone()
    if not exceptional and expected is None:
        expected = reference(logits, labels, smoothing, scale, ignored, dy)
    runner = Runner(name)
    logits.requires_grad_(True)
    loss = runner.module.cross_entropy_loss(logits, labels, smoothing, scale, ignored, inplace, group)
    grad, = torch.autograd.grad(loss.sum(), logits) if sum_loss else torch.autograd.grad(loss, logits, grad_outputs=dy)
    returned = runner.backward_result[0]
    assert len(runner.backward_result) == 8 and all(v is None for v in runner.backward_result[1:])
    assert torch.equal(runner.saved[2], labels_before)
    assert runner.saved[2].storage_offset() == 0 and runner.saved[2].is_contiguous()
    if labels.numel():
        assert runner.saved[2].untyped_storage().data_ptr() != labels.untyped_storage().data_ptr()
    if inplace:
        assert returned is runner.saved[3], "in-place return must be the original normalization result"
        if logits.stride(-1) == 1:
            assert returned is logits, "working normalization must not change caller in-place identity"
    else:
        assert returned is not logits
        first_gradient_allocation = runner.allocations[runner.backward_allocations_start]
        assert returned is first_gradient_allocation
    if not exceptional:
        torch.testing.assert_close(loss.detach().double(), expected[0], atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(grad.detach().double(), expected[1], atol=5e-5, rtol=5e-5)
    else:
        selected = labels_before == ignored
        assert torch.equal(loss[selected], torch.zeros_like(loss[selected]))
        assert torch.equal(grad[selected], torch.zeros_like(grad[selected]))
        assert torch.isnan(loss[~selected]).all() and torch.isnan(grad[~selected]).all()
    allowed = footprint(returned) if inplace else set()
    for base, (tensor, saved) in before.items():
        after = storage_bytes(tensor)
        changed = torch.nonzero(after != saved).flatten().tolist()
        assert all(base + offset in allowed for offset in changed), (name, changed)
    backward_args = runner.launch_tensors[-1]
    assert backward_args["dloss_ptr"].is_contiguous() and backward_args["dloss_ptr"].storage_offset() == 0
    if dy.numel():
        assert backward_args["dloss_ptr"].untyped_storage().data_ptr() != runner.actual_upstream.untyped_storage().data_ptr()
    return {"case": name, "logits": info(logits), "labels": info(labels), "upstream": info(dy),
            "inplace": inplace, "inplace_identity": not inplace or returned is runner.saved[3],
            "logical_reference_passed": not exceptional, "exceptional_ignored_zero": exceptional,
            "full_backing_frame_passed": True, "actual_autograd": True, "launches": runner.launches}


def partial_float_storage(tail):
    storage = torch.UntypedStorage(24 + tail)
    raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (24+tail,), (1,))
    raw.fill_(173)
    result = torch.empty(0, dtype=torch.float32).set_(storage, 0, (2,3), (3,1))
    result.copy_(torch.tensor([[0.,1.,2.],[2.,0.,1.]]))
    return result


def main():
    global ACTIVE_OUTPUT
    output = E / (sys.argv[1] if len(sys.argv) > 1 else "cpu-controls001")
    assert not output.exists()
    output.mkdir(parents=True)
    ACTIVE_OUTPUT = output
    shutil.copyfile(SOURCE, output / SOURCE.name)
    shutil.copyfile(__file__, output / "probe.py")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    cases = []
    def case(*args, **kwargs):
        item = run_case(*args, **kwargs)
        cases.append(item)
        (output/"partial-cases.json").write_text(json.dumps(cases,indent=2,sort_keys=True)+"\n")
        print(item["case"], "passed", flush=True)
    dense = lambda: torch.tensor([[0.,1.,2.],[2.,0.,1.]])
    case("dense", dense(), torch.tensor([0,1]))
    case("smoothing", dense(), torch.tensor([0,1]), smoothing=.1, sum_loss=True)
    case("z_loss", dense(), torch.tensor([0,1]), scale=.2)
    case("smoothing_z", dense(), torch.tensor([0,1]), smoothing=.2, scale=.1, dy=torch.tensor([1.,2.]))
    case("strided_labels", dense(), torch.tensor([0,2,1])[::2])
    case("offset_strided_labels", dense(), torch.tensor([9,0,2,1])[1::2])
    case("expanded_labels", dense(), torch.tensor([0]).expand(2))
    case("mixed_ignored", dense(), torch.tensor([-100,1]), smoothing=.1, scale=.2)
    case("ignored_strided_labels", dense(), torch.tensor([-100,0,-100])[::2])
    case("column_stride", torch.arange(12.).reshape(2,6)[:,::2], torch.tensor([0,1]), inplace=True)
    case("transpose", torch.arange(6.).reshape(3,2).t(), torch.tensor([0,1]), inplace=True)
    case("row_stride_offset", torch.arange(21.).reshape(7,3)[1:5:2], torch.tensor([0,1]), inplace=True)
    case("contiguous_offset", torch.arange(12.).reshape(4,3)[1:3], torch.tensor([0,1]), inplace=True)
    case("expanded_rows", torch.tensor([[0.,1.,2.]]).expand(2,3), torch.tensor([0,1]), dy=torch.tensor([1.,2.]))
    case("overlapping_rows", torch.arange(4.).as_strided((2,3),(1,1)), torch.tensor([0,1]))
    case("lazy_negative_overlapping_rows", torch._neg_view(torch.arange(4.).as_strided((2,3),(1,1))), torch.tensor([0,1]), smoothing=.1, scale=.2)
    case("singleton_strided_column", torch.arange(12.).as_strided((2,1),(3,7),2), torch.tensor([0,0]))
    case("singleton_row_negative_inplace", torch._neg_view(torch.arange(6.).as_strided((1,3),(0,1),2)), torch.tensor([1]), inplace=True)
    case("lazy_negative", torch._neg_view(dense()), torch.tensor([0,1]))
    case("lazy_negative_inplace", torch._neg_view(dense()), torch.tensor([0,1]), inplace=True)
    case("lazy_negative_labels", dense(), torch._neg_view(torch.tensor([0,-1])))
    case("strided_upstream", dense(), torch.tensor([0,1]), dy=torch.tensor([1.,9.,2.])[::2])
    case("lazy_negative_upstream", dense(), torch.tensor([0,1]), dy=torch._neg_view(torch.tensor([1.,2.])))
    x = dense()
    case("inplace_upstream_alias", x, torch.tensor([0,1]), dy=x[:,0], inplace=True)
    labels = torch.tensor([0,1], dtype=torch.int64)
    case("inplace_label_storage_alias", labels.view(torch.float32).reshape(2,2), labels, inplace=True)
    case("single_class", torch.tensor([[2.],[1.]]), torch.tensor([0,0]), scale=.2)
    case("empty_batch", torch.empty(0,3), torch.empty(0,dtype=torch.int64), dy=torch.empty(0))
    for tail in [1,2,3]:
        case("partial_backing_"+str(24+tail), partial_float_storage(tail), torch.tensor([0,1]), inplace=tail==2)
    case("ignored_nan_inf", torch.tensor([[float('nan'),1.],[float('inf'),0.],[0.,float('-inf')],[float('nan'),1.]]),
         torch.tensor([-100,-100,-100,0]), exceptional=True)
    case("backward_tail_4097", torch.zeros(1,4097), torch.tensor([4096]), smoothing=.1, scale=.2)
    case("forward_tail_65537", torch.zeros(1,65537), torch.tensor([65536]), smoothing=.1, scale=.2)
    distributed = []
    for width in [3,65537]:
        coordinator = Collectives(2)
        shards = [torch.linspace(-.5,.5,width).reshape(1,width), torch.linspace(.2,.8,width).reshape(1,width)]
        labels = torch.tensor([width+1])
        full = torch.cat(shards,dim=1)
        expected = reference(full,labels,.1,.2,-100,torch.ones(1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_case,"distributed_"+str(width)+"_rank"+str(rank), shards[rank], labels.clone(),
                       smoothing=.1,scale=.2,group=coordinator.group(rank),
                       expected=(expected[0],expected[1][:,rank*width:(rank+1)*width])) for rank in range(2)]
            results = [future.result(timeout=50) for future in futures]
        cases.extend(results)
        distributed.append({"local_width":width,"world_size":2,"events":coordinator.events})
        print("distributed",width,"passed",flush=True)
    receipt = {"schema":1,"candidate_sha256":sha(SOURCE),"script_sha256":sha(__file__),
        "mode":"actual Triton CPU interpreter and actual Torch autograd; CUDA context adapted; bounded CPU collective semantics",
        "torch":torch.__version__,"triton":triton.__version__,"candidate_positive_count":len(cases),
        "gpu_execution":False,"native_numerical_conformance":False,"cases":cases,"distributed":distributed}
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"receipt":str(output/"receipt.json"),"sha256":sha(output/"receipt.json"),"candidate_positive_count":len(cases)}))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            main()
        except Exception:
            if ACTIVE_OUTPUT is not None:
                (ACTIVE_OUTPUT/"failure.txt").write_text(traceback.format_exc())
            raise
