import bench.tritonbench_g.kldiv_ops.Proofs.FlatIO
import bench.tritonbench_g.kldiv_ops.Proofs.SourceDomain

namespace VeriTile.Bench.TritonBenchG.KldivOps.Contracts
open VeriTile.Triton

/-- Actual source access safety, successful full flattened execution, every
independent pointwise loss, and preservation of every other raw memory cell. -/
def ForwardElementIO (kernel : ComputeKernel) (A : FlatAlloc) (s : BlockState)
    (Y T O : RegionName) (sy st so N : Nat) (eps : ℝ) (flag : Bool) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafeR .triv A.extent s ∧
    ∃ t, exec (A.flattenKernel kernel.eraseDType.toAlgKernel)
      (A.flattenState s) = some t ∧
    (∀ k : Fin N,
      t.readMem A.flat (A.addr O (s.pids 0 * so + k.val)) =
        FullMath.loss flag eps (s.readMem Y (s.pids 0 * sy + k.val))
          (s.readMem T (s.pids 0 * st + k.val))) ∧
    (∀ r off, (r ≠ A.flat ∨ ∀ k : Fin N,
      off ≠ A.addr O (s.pids 0 * so + k.val)) →
        t.mem r off = (A.flattenState s).mem r off)

/-- Actual source access safety and complete flattened execution accumulate
the independent whole-row loss before one scalar store, with a full raw frame. -/
def ForwardReductionIO (kernel : ComputeKernel) (A : FlatAlloc) (s : BlockState)
    (Y T O : RegionName) (sy st so N : Nat) (eps : ℝ) (flag : Bool) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafeR .triv A.extent s ∧
    ∃ t, exec (A.flattenKernel kernel.eraseDType.toAlgKernel)
      (A.flattenState s) = some t ∧
    t.readMem A.flat (A.addr O (s.pids 0 * so)) =
      FullMath.rowLoss flag eps (fun k ↦ s.readMem Y (s.pids 0 * sy + k))
        (fun k ↦ s.readMem T (s.pids 0 * st + k)) N ∧
    (∀ r off, (r ≠ A.flat ∨ off ≠ A.addr O (s.pids 0 * so)) →
      t.mem r off = (A.flattenState s).mem r off)

/-- Actual source access safety, successful full flattened backward execution,
every independent scalar gradient, and preservation of every other raw cell. -/
def BackwardIO (kernel : ComputeKernel) (A : FlatAlloc) (s : BlockState)
    (T O : RegionName) (st so N : Nat) (flag : Bool) : Prop :=
  kernel.eraseDType.toAlgKernel.TraceSafeR .triv A.extent s ∧
    ∃ t, exec (A.flattenKernel kernel.eraseDType.toAlgKernel)
      (A.flattenState s) = some t ∧
    (∀ k : Fin N, t.readMem A.flat (A.addr O (s.pids 0 * so + k.val)) =
      FullBackward.gradient flag (s.readMem T (s.pids 0 * st + k.val))) ∧
    (∀ r off, (r ≠ A.flat ∨ ∀ k : Fin N,
      off ≠ A.addr O (s.pids 0 * so + k.val)) →
        t.mem r off = (A.flattenState s).mem r off)

end VeriTile.Bench.TritonBenchG.KldivOps.Contracts

#guard_msgs (drop info) in
#auditModuleAxioms
