import bench.tritonbench_g.triton_argmax.Proofs.Contract

namespace VeriTile.Bench.Repairs.TritonArgmax.Checked
open Source

theorem check_iff (o : Observation) : check o = Bool.true ↔ InputContract o := by
  unfold check InputContract
  cases o.dim with
  | none => simp
  | some d => cases hn : Host.normalizeDim o.input.shape.length d <;> simp [hn]

theorem check_arguments_iff (a : Arguments) :
    checkArguments a = Bool.true ↔ AdmittedArguments a := by
  cases a <;> simp [checkArguments, AdmittedArguments, check_iff]

theorem dimension_dtype_ignored (o : Observation) (d : Int) (branch : o.dim = some d)
    (dtype : DTypeArgument) :
    check {o with dtype := dtype} = check o := by
  simp [check, branch, BaseContract,DimensionContract,outer,extent,inner]

theorem normalized_valid (rank : Nat) (dim : Int) (ax : Nat)
    (normalized : Host.normalizeDim rank dim = some ax) : ax < rank := by
  unfold Host.normalizeDim at normalized
  split at normalized
  · rename_i bounds
    have positive : (0 : Int) < Int.ofNat rank := by omega
    have upper := Int.emod_lt_of_pos dim positive
    have lower := Int.emod_nonneg dim (by omega : Int.ofNat rank ≠ 0)
    simp only [Option.some.injEq] at normalized
    rw [←normalized]
    exact (Int.toNat_lt lower).2 upper
  · cases normalized

theorem global_coverage (o : Observation) (h : GlobalContract o)
    (first : Nat → Reduction.Tree (block o)) (second : Reduction.Tree (midBlock o))
    (completeFirst : ∀ pid, pid < blocks o → (first pid).Complete)
    (completeSecond : second.Complete) :
    Math.GlobalCoverage ⟨o.input.numel,block o,blocks o,midBlock o,first,second⟩ :=
  Host.global_coverage _ h.2.2.1 h.2.2.2.2.1 rfl
    h.2.2.2.2.2.2.2.1 completeFirst completeSecond

#axiomsClean check_iff
#axiomsClean check_arguments_iff
#axiomsClean dimension_dtype_ignored
#axiomsClean normalized_valid
#axiomsClean global_coverage
end VeriTile.Bench.Repairs.TritonArgmax.Checked
