# Source and execution scope

The original Python file has SHA-256
`28e2c4e324b0098cac1aec32d11a0883dbebc491564f20674f6f08f1939fe314`.
The selected kernel is `chunk_global_reversed_cumsum_vector_kernel` from
`thunlp/TritonBench@603e28a`. The source proof keeps all its statements and
unspecified boundary padding. Reverse iteration is represented by a signed
mirror index, so the empty time dimension performs zero iterations. Signed
block offsets are projected only after multiplication; the pointer element
output cast retains its independent destination storage type.

The real theorems concern the complete per-program JIT body with its explicit
layout parameters and actual program IDs. Successful execution and the
initial-memory output formula require distinct input/output regions, positive
BT and injective active output addresses. Finite IO additionally requires
bounded active addresses in a disjoint finite allocation and an explicitly
zero unspecified-value oracle. Arbitrary oracle values remain supported by
the region theorem, whose result includes their exact time-tail contribution.

Typed state equivalence retains every source operation and quantifies over
all six independent input/output storage choices, every initial state and
every arithmetic backend. Successful bf16-input/fp16-output runs at every
layout and initial state in a mathematical consistency backend establish
nonemptiness separately, including a concrete two-chunk partial-tail witness.
Neither statement supplies an
IEEE model or a proof that Triton lowering implements that model.

Offline native checks use unmodified original JIT bodies, decorators and
wrappers. The vector portion covers all nine BT/warp configurations at six
matching storage types with T = 0, 1, BT and BT + 1, and all 36 independent
input/output storage combinations at BT = 16, two warps and T = 17. BS = 32,
S = 33 and runtime strides use the i32 ABI in that matrix. Compilation and
IR inspection establish admission for those cases only.

Formal refinement to native fixed-width indices and runtime scalar ABI is
pending. In particular, the unbounded integer/natural interpreter theorem is
not a fixed-width no-overflow theorem. The host wrapper passes the input's
strides to both input and output pointers. Compatibility of `empty_like`
output layout, the relation between batch/head strides, and arbitrary
noncontiguous PyTorch input layouts are not proved. Launch-wide composition,
autotuner selection and IEEE/native numerical correctness are also outside
the current per-program proof.
