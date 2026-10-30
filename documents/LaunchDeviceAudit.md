# Focused launch-device binding audit — 2026-10-19

This focused audit is complete for the fourteen source revisions below. It was
prompted by the owning-device correction to `fused_recurrent_retention` and
covered accepted wrappers receiving caller tensors whose direct Triton launches
had no effective owning-device context or executed current-device precondition.
The source search was refined through delegated wrappers and existing guards;
an absent context in the public function alone was not treated as a defect.
This record does not certify every repository launcher or future source revision.

## Runtime evidence and correction

Installed Triton **3.7.1**, `triton/runtime/jit.py`,
`JITFunction.run` lines 713–720 and 761–762, reads the ambient device before
argument binding and uses that device for the stream, cache, target and backend:

```python
device = driver.active.get_current_device()
stream = driver.active.get_current_stream(device)
# ...
kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
# ...
kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, ...)
```

Observed installed file: `/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py`.
Its SHA-256 was `5fe46fbfac257a2b8645b7a2cc709232273d42dd5e1d55de9770b412a9476d77`.
Equal input devices and explicit-device output allocation do not themselves
select the ambient launch device. Each correction puts
`with torch.cuda.device(owner.device):` around the actual source submission.
The caller device is restored on normal return and submission exception.
`spinning_lock_reduction` uses one context around both phases, retaining their
same-stream order and asynchronous return. Delegating public wrappers remain
delegating wrappers; the context is at the effective launch site.

## Exact corrected source revisions

Paths are relative to the repository root. Old hashes identify the accepted
candidate immediately before this focused correction; current hashes identify
the corrected source snapshot. These pairs preserve correction evidence without
rewriting historical acceptance or claiming that the old source selected the
owning device.

| Task / source | Old SHA-256 | Current SHA-256 |
|---|---|---|
| [fused_recurrent_delta](../bench/repairs/tritonbench_g/fused_recurrent_delta/fused_recurrent_delta_repaired.py) | `469d63b2788ba099221e8bedfe9fd9ddac83d673c5262b938acbf9852e1ce682` | `41d896ade9b5f5d6a0a3fdc3843f36124c6417d426f590ecb35c024f104f8b53` |
| [multinomial_sampling](../bench/repairs/tritonbench_g/multinomial_sampling/multinomial_sampling_repaired.py) | `fe754f34e7ae2fb4dc797a8b6b5fb0dc7b49275961fc0888044adaca5ebaf11b` | `378ef3b1bc26f30110386fc1781e3b16bf1bc3c70b6a9ba71dc5d0758885820a` |
| [spinning_lock_reduction](../bench/repairs/tritonbench_g/spinning_lock_reduction/spinning_lock_reduction_phased.py) | `8f07a1b4e3f9ac7b9193faaa45a182d9e928e0cab2071037a2773843f7e7e58e` | `b6a41096433e567770b7aeb5285329128768478b601eeaa36b8ba7ac2c5befb5` |
| [attention_forward_triton](../bench/repairs/tritonbench_g/attention_forward_triton/attention_forward_triton_repaired.py) | `d38157443f0931243ca07de105681a46e8210d01ba16600adfc538b3437eefee` | `545872e280f84d19725d001872cee31bf0714f8aa520e1b5a17dbd09fec165b7` |
| [attention_fwd_triton1](../bench/repairs/tritonbench_g/attention_fwd_triton1/Whole001/attention_fwd_triton1_repaired.py) | `921575e2b476c9033d5042927038075a7c058b7ae174fc0beccabe28440abeef` | `a87c37e5194ed1d3415ca59d17719be62e5b4a529f3307d6d07cabbaed7d6c54` |
| [attention_fwd_triton2](../bench/repairs/tritonbench_g/attention_fwd_triton2/Whole001/attention_fwd_triton2_repaired.py) | `eedab60247bb8128eb6bfe357474f1f18d8a3ac9a55a92e6d8a6769c739af322` | `eb7192d0aaab5db92b98eba026b05eb728774263b5f3afecbd7270d9d84bc005` |
| [attention_llama](../bench/repairs/tritonbench_g/attention_llama/attention_llama_repaired.py) | `4b63f558e0880e2a16709c82ca130ee9309ea4c75212dd56eae202c8b5027e45` | `b209df6f59ed377ecf7480660ef82a19d0068b0ab708dc1ad345100a997945bd` |
| [attn_fwd_triton](../bench/repairs/tritonbench_g/attn_fwd_triton/attn_fwd_triton_repaired.py) | `f9af84483b6ec2cc7463c8f3b25577ec3570e338bd65830516765c9e3a56a760` | `497d3df38ba737060b72f34fc9a6c2dfc5566a263a011ed51a02baa28e5c4086` |
| [fp4_to_bf16](../bench/repairs/tritonbench_g/fp4_to_bf16/fp4_to_bf16_repaired.py) | `dbf52f619ca6e418ef690741e8b8a5f509de2cea4aeee3c1abe71f29179f3930` | `70f10f7ab112fd50f29511e24c8dac26374d51e3ea8bb6dc769e899bdc6d22be` |
| [layer_norm_liger](../bench/repairs/tritonbench_g/layer_norm_liger/layer_norm_liger_repaired.py) | `81e64aaea77ee7b45073440fc67d51b2e6dae041356e7285098399771d8e47d8` | `6ba82f8c92986e02bca37469e5c17f2e1471137f939191d7fef4b48bd7266a28` |
| [matmul_kernel](../bench/repairs/tritonbench_g/matmul_kernel/matmul_kernel_repaired.py) | `596b295a94f32ce1ff9d6d77942dfec3a68b067715f1750688c41a42cb58e3af` | `10892032b7d88b46f0bc43e0bc94176c704c6e7d7b3cb6f92cf7229ca4926a55` |
| [matmul_leakyrelu](../bench/repairs/tritonbench_g/matmul_leakyrelu/matmul_leakyrelu_repaired.py) | `0240070b2db13874a278b632f241768ab223a917b314e3d26e71b86f6487a18d` | `ca9ec0b702afebc59e4fbed320a55e4784005a361856a0bba10d13e60e2db34a` |
| [matmul_persistent_triton](../bench/repairs/tritonbench_g/matmul_persistent_triton/matmul_persistent_triton_repaired.py) | `f6f67cac011b1ec78872fe4a0282e59ddb9e9c05f9889e1aad2657b52b206755` | `ad6eb70f7493aff7f323558870765887e09c208c3a547e2c6271fe519efddd7d` |
| [matmul_triton_autotune](../bench/repairs/tritonbench_g/matmul_triton_autotune/matmul_triton_autotune_view_safe.py) | `4ff075148740f208edd50d868dbe8c6486cf6b53e848d96e880836f88d3b7ed5` | `57ef247c2cd52e4300a270ac83707881b15e11d067dad81430c5d0bde1f75e08` |

## Cases excluded from these fourteen corrections

- Already guarded: `fused_recurrent_retention` has owning-device contexts around
  both submissions from the preceding correction. `fifth_order_sph_harmonics`,
  `isfinite_kernel`, `pow_scalar_tensor` and `relu_strided_buffer` already use
  `torch.cuda._DeviceGuard`. `fused_layernorm_triton` and `layer_norm_welfold`
  use `_DeviceGuard(0)` plus `set_device(0)` under their existing device-0
  contracts. Other preexisting `torch.cuda.device` callers are not enumerated here.
- Delegated context: `cross_entropy_ops` reaches a context-guarded source
  launcher. An additional context at the outer wrapper was unnecessary.
- Executed ambient-device precondition: `chunk_gate_recurrence`, accepted after
  the scan, explicitly rejects a caller current device different from the input
  device. Its existing checked contract was retained.
- Internal allocation only: `batched_vecmat_mult`, `matrix_reduction`,
  `matrix_transpose`, `nested_loops_processing` and `matmul_tma` create their
  kernel inputs internally and did not meet the external-owning-input mismatch
  trigger. This exclusion is not a general runtime-conformance result for them.

## Validation and remaining boundary

The fourteen targeted CPU suites passed after the corrections. They observe the
device at the actual submission, use logical cuda:1 inputs with ambient cuda:0,
and verify caller restoration on success and injected submission exceptions.
Existing no-launch branches, source interpreter checks, input rejections and
original defect controls remain covered. In the two CPU-interpreted matmul
wrappers, logical CUDA tensor metadata is unwrapped to the existing CPU storage
only at the interpreter boundary. These are mocked CUDA context controls, not
multi-GPU tests.

Each changed host helper and public `Correct.lean` / `FPEquiv.lean` entry
compiled. Formal edits clarify that device/stream observations refer to the
effective owning context; they do not add a proof of the CUDA context primitive
or a Python/runtime refinement theorem. Kernel ASTs and numerical graphs are
unchanged. The final implementation audit matched all 56 recorded file hashes,
parsed the changed Python, and passed scoped `git diff --check`.

No native GPU execution, device/stream runtime conformance, native numerical validation,
backend-wide audit or historical acceptance-count change is claimed. The
shared signed-int `gridX * gridY * gridZ` launcher compatibility issue remains
deferred in [bench/TODO.md](../bench/TODO.md), along with its existing runtime
applicability limits. This device-context correction does not resolve that issue.
