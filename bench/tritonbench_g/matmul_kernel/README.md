# matmul_kernel

`FPCorrect.lean` adds two full forward contracts while preserving both original public mathematical statements in `MatmulKernel.lean` byte-for-byte.

- `FPCorrect.half_backend_correctness`: successful execution, independently defined output values, exact memory effect and frame for arbitrary floating backend primitives.
- `FPCorrect.half_software_correctness`: the same complete stream instantiated with binary16 storage, sequential binary32 fused multiply-adds seeded by the live accumulator, and round-to-nearest-even binary16 output conversion. Its output is independent of every fallback callback.

The accepted scope is the original arbitrary `BM`, `BN`, `BK`, `numKBlocks` and two-dimensional program coordinates, with only the original `BN ≤ 4096` output-layout premise. Input rows/columns wrap modulo 4096, pointers advance by the original fixed strides, all K-block lanes load without masks, and every output lane stores without a mask. Zero trips and empty tiles remain covered. No finite-input, alignment, divisibility, non-aliasing or global identity-rounding assumption was added. The backend contract also preserves arbitrary initial undefined values; these unmasked operations do not read padding fallbacks.

`Proofs/FPReference.lean` defines the full recurrence from initial typed memory and primitive operations, independently of executing the source. `Proofs/FPExecution.lean` and `FPComplete.lean` derive it from the actual full loop. `Proofs/FPShape.lean` establishes the typed source's control/address/operation projection against the unchanged original source; erasure does not prove a floating-value identity. The only added source annotations state the actual half input storage used by Python's tests. The three-argument `tl.dot(a, b, accumulator)` remains a dot-with-live-accumulator operation.

`Proofs/FPWitness.lean` executes two K blocks and proves that the second fp32 fused multiply-add rounds `1 + 2^-24` back to `1`. This distinguishes the software profile from exact Real arithmetic. It also checks the concrete half output and an untouched output-region address.

The Python device kernel hard-codes `M=N=K=4096` and row-major strides 4096/1. The original generalized proof supplies its K-loop trip count explicitly; binding an actual launch uses `numKBlocks = cdiv(4096, BK)`. The two-block witness belongs to that original generalized proof domain. No change to the host wrapper or to the old statement scope is implied.

Native correspondence remains separate: typed loads/stores and allocation extents must match the source interface; the compiler's three-argument half dot must implement the chosen dotAcc primitive, including its accumulation schedule, fusion, subnormal/FTZ and NaN policies; and fp32-to-half conversion must match the selected rounding policy. The closed software profile uses ascending fused fp32 FMAs and gradual underflow, and is not asserted to match an unspecified GPU tensor-core schedule. No GPU execution is reported here.
