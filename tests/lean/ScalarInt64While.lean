import tests.lean.ScalarInt64WhileReference
import VeriTile.Triton.DSL.ScalarInt64While
import VeriTile.Triton.Float.ScalarInt64While
namespace VeriTile.Tests.ScalarInt64While
open VeriTile.Triton

def signature : CallSignature where
  scalars := [("n_rows",.integer)]

def source : ComputeKernel :=
  with_triton_signature signature pointer_offsets [] in triton {
    n_rows = tl.cast(n_rows, tl.int64)
    row = tl.full((), 0, tl.int64)
    while row < tl.cast(n_rows, tl.int64):
      row += tl.full((), 1, tl.int64)
  }

/-- Whole-program equality preserves initialization, signed bound, source
counter increment, zero iterations and the final counter for every input. -/
theorem complete_program : source=ScalarInt64WhileReference.source := by rfl

/-- error: int64 while counter requires a dominating signed64 binding -/
#guard_msgs (error) in
example : ComputeKernel :=
  with_triton_signature signature pointer_offsets [] in triton {
    row = tl.full((), 0.0, tl.float32)
    while row < tl.cast(n_rows, tl.int64):
      row += tl.full((), 1, tl.int64)
  }

/-- error: int64 while requires an invariant scalar bound or its int64 cast -/
#guard_msgs (error) in
example (X : RegionName) : ComputeKernel :=
  with_triton_signature signature pointer_offsets [] in triton {
    row = tl.full((), 0, tl.int64)
    while row < tl.load(X):
      row += tl.full((), 1, tl.int64)
  }

/-- error: int64 while counter may only be written by the final increment -/
#guard_msgs (error) in
example : ComputeKernel :=
  with_triton_signature signature pointer_offsets [] in triton {
    row = tl.full((), 0, tl.int64)
    while row < tl.cast(n_rows, tl.int64):
      row += tl.full((), 1, tl.int64)
      row += tl.full((), 1, tl.int64)
  }

/-- error: int64 while bound must remain unchanged -/
#guard_msgs (error) in
example : ComputeKernel :=
  with_triton_signature signature pointer_offsets [] in triton {
    row = tl.full((), 0, tl.int64)
    while row < tl.cast(n_rows, tl.int64):
      n_rows += tl.full((), 1, tl.int64)
      row += tl.full((), 1, tl.int64)
  }

/-- error: int64 while requires counter += tl.full((), 1, tl.int64) -/
#guard_msgs (error) in
example : ComputeKernel :=
  with_triton_signature signature pointer_offsets [] in triton {
    row = tl.full((), 0, tl.int64)
    while row < tl.cast(n_rows, tl.int64):
      row += tl.full((), 2, tl.int64)
  }

/-- error: int64 while bare bound requires known integer type -/
#guard_msgs (error) in
example : ComputeKernel :=
  with_triton_signature signature pointer_offsets [] in triton {
    bound = tl.full((), 2.5, tl.float32)
    row = tl.full((), 0, tl.int64)
    while row < bound:
      row += tl.full((), 1, tl.int64)
  }

#axiomsClean complete_program
end VeriTile.Tests.ScalarInt64While
