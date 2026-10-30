import bench.tritonbench_g.spinning_lock_reduction.Proofs.Host

/-! Original source entry. The generated typed AST retains both while loops,
nine rounds, persistent accumulator and locks, unmasked P load/store, masked C
store and CAS(1,1). P/C and the accumulator are FP32; locks/control are signed
int32. `originalPlan` is checked against the whole source tree by the restricted
compiler. The wrapper grid is exactly num_sms and its Python return is None. -/
namespace VeriTile.Bench.TritonBenchG.SpinningLock.Kernels
open Frontend

abbrev originalSource := originalTypedSource
abbrev originalKernel := originalPlan

structure WrapperResult (V : Type) where
  events : List Event
  final : State V
  returned : Option Unit

/-- Host launch boundary, including in-place state and Python's implicit None.
This models finite scheduler prefixes, not the GPU runtime's liveness policy. -/
def launch (b : Backend V) (h : HostArgs) (schedule : List Nat) (mem : Memory V) :
    Option (WrapperResult V) :=
  (compileAndRun originalSource b h.config schedule mem).map fun result =>
    ⟨result.1, result.2, wrapperReturn⟩

/-- C readback uses the actual host element strides. -/
def readC (h : HostArgs) (s : State V) (row col : Nat) : V :=
  s.mem.output (row * h.strideM + col * h.strideN : Nat)

 theorem launch_execution (b : Backend V) (h : HostArgs) (schedule : List Nat)
    (mem : Memory V) :
    launch b h schedule mem = some
      ⟨(run b h.config schedule (initial b mem)).1,
        (run b h.config schedule (initial b mem)).2, none⟩ := by
  simp only [launch, originalSource, original_source_execution, Option.map_some, wrapperReturn]

#axiomsClean launch_execution
end VeriTile.Bench.TritonBenchG.SpinningLock.Kernels
