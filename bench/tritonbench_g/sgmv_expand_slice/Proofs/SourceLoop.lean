import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceBody

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceBody
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

def accumulated {α : Type} [Inhabited α] (M : Algebra α) (dot : DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (coordinates : Coordinates) (s : State α)
    (start : Nat) (initial : Values α .real [32,32]) : Values α .real [32,32] :=
  foldRange (update M dot abi p coordinates s) start ((p.inner+15)/16) 1 initial

theorem loop_run {α : Type} [Inhabited α] (M : Algebra α) (dot : DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (coordinates : Coordinates) (s t : State α)
    (start : Nat) (initial : Values α .real [32,32])
    (h : Ready p coordinates s t initial) :
    ∃ final, rangeInt (install M dot) "k" (start : Int) (((p.inner+15)/16 : Nat) : Int) 1
      (SourceParts.body abi p) t = some final ∧
      Ready p coordinates s final (accumulated M dot abi p coordinates s start initial) := by
  apply RangeReuse.range_values_int (install M dot) "k" (SourceParts.body abi p)
    (update M dot abi p coordinates s)
    (fun _ acc u => Ready p coordinates s u acc) (fun acc u => Ready p coordinates s u acc)
    start ((p.inner+15)/16) 1 initial t
  · intro n acc u hu _
    exact body_run M dot abi p coordinates s u n acc hu
  · intro _ _ _ hu
    exact hu
  · exact h

theorem signed_block_count (inner : Nat) :
    ((inner : Int)+16-1)/16 = (((inner+15)/16 : Nat) : Int) := by omega

#axiomsClean loop_run
#axiomsClean signed_block_count
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceBody
