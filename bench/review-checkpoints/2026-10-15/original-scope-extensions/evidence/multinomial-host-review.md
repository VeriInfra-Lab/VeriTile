# Read-only review of new multinomial host modules

Reviewed FPHost.lean, FPWrapper.lean, HostCorrect.lean against canonical native-original/multinomial_sampling.py. The actual-kernel grid fold, independent initial-memory recurrence, source branch order, token1 Bx1 zeros result, default arange/expand logical values, clone initialization and fresh output hypotheses are consistent with the declared logical memory profile. The full host theorem explicitly limits supplied indices to the original int64-typed port and does not establish arbitrary torch dtypes.

One concrete discrepancy to correct: default `expand_as` is not literally zero-stride in the batch dimension for batch size 1. CPU PyTorch checks: `torch.arange(3).expand(0,3).stride() == (0,1)`, `.expand(1,3).stride() == (3,1)`, `.expand(2,3).stride() == (0,1)`. Current wrapperParameters uses strideIndexBatch=0 for every None case. This is observationally equivalent for B=1 (only row 0 is active), but a source-lowering comment should acknowledge canonicalization or parameterization should preserve actual singleton stride. No proof needs a new value restriction.

Native int64 metadata/ABI and arbitrary unbounded `.int` cell interpretation remain an explicit boundary. Do not count this logical port as proof of every supplied-index dtype. Allocation/provenance and native launch/address specialization also remain outside the verified host lowering. `freshDefault` is a source-realistic precondition even though logical equality is more general; do not remove it merely because the proof body does not need it.

No files changed by this review. Findings sent to the root agent before further development.
