# chunk_bwd_dqkg repair findings

The accepted repaired arithmetic kernel is retained exactly. The wrapper uses
direct contiguous FP32 inputs on one CUDA device, allocates the three original
outputs, retains `dg.fill_(-1e9)`, and returns those exact objects. It accepts
disjoint data spans in the same underlying storage; no input copy occurs.

The original source's unspecified block-load tail values could contribute to
active outputs. Its history parent shape also allowed a final key tile to read
the following history chunk. The accepted repair supplies zero padding to all
seven block loads and selects the history chunk in the base while masking
against logical K. The original fixture's history capacity is insufficient for
its two time chunks. The repaired ABI requires one K-by-V history matrix per
chunk. Literal canonical strides prevent the original shared Q/K stride issue.

The public direct Real and native roots are `Direct.chunk_bwd_dqkg_real` and
`DirectNative.chunk_bwd_dqkg_if_local`. They compose executable admission,
ordinary fresh allocation descriptors, actual strides/bases, complete source
execution, tuner trials, all output values and frames. The Real theorem covers
arbitrary Real arrays independently of floating encodings. Native inputs are
decoded directly from physical little-endian bytes. Native output bytes are
serialized only in returned output spans; no caller byte-binding, layout
safety, successful-execution or result-value premise is required.

Native realization retains an explicit selected compiler/profile/layout
mapping boundary. This directory does not contain an actual compiler artifact
checker and does not claim that an external checker was run in this repair.
Historical shape profiles are examples only. Primitive numerical agreements
are scalar instructions and fixed m16n8k8 TF32 coordinates with reached carry.
No whole dot, whole reduction or whole output equality is assumed.

The current source harness executes the retained arithmetic AST over CPU
backing storage. It passes 12 admitted shapes across both warp configurations
and empty/nonempty tuner traces, plus 93 rejection cases. It checks complete
dQ/dK/per-key-tile dG targets, tail bounds, direct argument and returned-object
identities, the original fill, and byte frames. CUDA allocation/launch and
floating arithmetic are mocked; this is not a GPU numerical run. Historical
external artifact and autograd claims are not validation of this current tree.

Original source and acceptance entries are unchanged.
