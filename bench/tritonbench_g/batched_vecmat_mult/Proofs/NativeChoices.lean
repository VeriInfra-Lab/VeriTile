import bench.tritonbench_g.batched_vecmat_mult.Proofs.NativePlans

/-! All distinct source trees recovered from the competing shared-store
values in the bounded isolated SM90 traces: eight BK8 alternatives and sixteen
BK16 alternatives. Each is a valid reduction; none is declared FP-equal to
another. The selector retains that choice separately for tagged source inputs.
This is a finite capture profile, not a universal compiler/GPU certificate. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
open VeriTile.Triton FP.Equational

def sm90Shared8Choice0 : ReductionPlan 8 where
  tree := .add (.add (.add (.input ⟨0, by decide⟩) (.input ⟨4, by decide⟩)) (.add (.input ⟨2, by decide⟩) (.input ⟨6, by decide⟩))) (.add (.add (.input ⟨1, by decide⟩) (.input ⟨5, by decide⟩)) (.add (.input ⟨3, by decide⟩) (.input ⟨7, by decide⟩)))
  padding := 0
  valid := by decide +kernel

def sm90Shared8Choice1 : ReductionPlan 8 where
  tree := .add (.add (.add (.input ⟨1, by decide⟩) (.input ⟨5, by decide⟩)) (.add (.input ⟨3, by decide⟩) (.input ⟨7, by decide⟩))) (.add (.add (.input ⟨0, by decide⟩) (.input ⟨4, by decide⟩)) (.add (.input ⟨2, by decide⟩) (.input ⟨6, by decide⟩)))
  padding := 0
  valid := by decide +kernel

def sm90Shared8Choice2 : ReductionPlan 8 where
  tree := .add (.add (.add (.input ⟨2, by decide⟩) (.input ⟨6, by decide⟩)) (.add (.input ⟨0, by decide⟩) (.input ⟨4, by decide⟩))) (.add (.add (.input ⟨3, by decide⟩) (.input ⟨7, by decide⟩)) (.add (.input ⟨1, by decide⟩) (.input ⟨5, by decide⟩)))
  padding := 0
  valid := by decide +kernel

def sm90Shared8Choice3 : ReductionPlan 8 where
  tree := .add (.add (.add (.input ⟨3, by decide⟩) (.input ⟨7, by decide⟩)) (.add (.input ⟨1, by decide⟩) (.input ⟨5, by decide⟩))) (.add (.add (.input ⟨2, by decide⟩) (.input ⟨6, by decide⟩)) (.add (.input ⟨0, by decide⟩) (.input ⟨4, by decide⟩)))
  padding := 0
  valid := by decide +kernel

def sm90Shared8Choice4 : ReductionPlan 8 where
  tree := .add (.add (.add (.input ⟨4, by decide⟩) (.input ⟨0, by decide⟩)) (.add (.input ⟨6, by decide⟩) (.input ⟨2, by decide⟩))) (.add (.add (.input ⟨5, by decide⟩) (.input ⟨1, by decide⟩)) (.add (.input ⟨7, by decide⟩) (.input ⟨3, by decide⟩)))
  padding := 0
  valid := by decide +kernel

def sm90Shared8Choice5 : ReductionPlan 8 where
  tree := .add (.add (.add (.input ⟨5, by decide⟩) (.input ⟨1, by decide⟩)) (.add (.input ⟨7, by decide⟩) (.input ⟨3, by decide⟩))) (.add (.add (.input ⟨4, by decide⟩) (.input ⟨0, by decide⟩)) (.add (.input ⟨6, by decide⟩) (.input ⟨2, by decide⟩)))
  padding := 0
  valid := by decide +kernel

def sm90Shared8Choice6 : ReductionPlan 8 where
  tree := .add (.add (.add (.input ⟨6, by decide⟩) (.input ⟨2, by decide⟩)) (.add (.input ⟨4, by decide⟩) (.input ⟨0, by decide⟩))) (.add (.add (.input ⟨7, by decide⟩) (.input ⟨3, by decide⟩)) (.add (.input ⟨5, by decide⟩) (.input ⟨1, by decide⟩)))
  padding := 0
  valid := by decide +kernel

def sm90Shared8Choice7 : ReductionPlan 8 where
  tree := .add (.add (.add (.input ⟨7, by decide⟩) (.input ⟨3, by decide⟩)) (.add (.input ⟨5, by decide⟩) (.input ⟨1, by decide⟩))) (.add (.add (.input ⟨6, by decide⟩) (.input ⟨2, by decide⟩)) (.add (.input ⟨4, by decide⟩) (.input ⟨0, by decide⟩)))
  padding := 0
  valid := by decide +kernel

def sm90Shared8Choices : List (ReductionPlan 8) :=
  [sm90Shared8Choice0, sm90Shared8Choice1, sm90Shared8Choice2, sm90Shared8Choice3, sm90Shared8Choice4, sm90Shared8Choice5, sm90Shared8Choice6, sm90Shared8Choice7]

def sm90Shared16Choice0 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨0, by decide⟩) (.input ⟨8, by decide⟩)) (.add (.input ⟨4, by decide⟩) (.input ⟨12, by decide⟩))) (.add (.add (.input ⟨2, by decide⟩) (.input ⟨10, by decide⟩)) (.add (.input ⟨6, by decide⟩) (.input ⟨14, by decide⟩)))) (.add (.add (.add (.input ⟨1, by decide⟩) (.input ⟨9, by decide⟩)) (.add (.input ⟨5, by decide⟩) (.input ⟨13, by decide⟩))) (.add (.add (.input ⟨3, by decide⟩) (.input ⟨11, by decide⟩)) (.add (.input ⟨7, by decide⟩) (.input ⟨15, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice1 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨1, by decide⟩) (.input ⟨9, by decide⟩)) (.add (.input ⟨5, by decide⟩) (.input ⟨13, by decide⟩))) (.add (.add (.input ⟨3, by decide⟩) (.input ⟨11, by decide⟩)) (.add (.input ⟨7, by decide⟩) (.input ⟨15, by decide⟩)))) (.add (.add (.add (.input ⟨0, by decide⟩) (.input ⟨8, by decide⟩)) (.add (.input ⟨4, by decide⟩) (.input ⟨12, by decide⟩))) (.add (.add (.input ⟨2, by decide⟩) (.input ⟨10, by decide⟩)) (.add (.input ⟨6, by decide⟩) (.input ⟨14, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice2 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨10, by decide⟩) (.input ⟨2, by decide⟩)) (.add (.input ⟨14, by decide⟩) (.input ⟨6, by decide⟩))) (.add (.add (.input ⟨8, by decide⟩) (.input ⟨0, by decide⟩)) (.add (.input ⟨12, by decide⟩) (.input ⟨4, by decide⟩)))) (.add (.add (.add (.input ⟨11, by decide⟩) (.input ⟨3, by decide⟩)) (.add (.input ⟨15, by decide⟩) (.input ⟨7, by decide⟩))) (.add (.add (.input ⟨9, by decide⟩) (.input ⟨1, by decide⟩)) (.add (.input ⟨13, by decide⟩) (.input ⟨5, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice3 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨11, by decide⟩) (.input ⟨3, by decide⟩)) (.add (.input ⟨15, by decide⟩) (.input ⟨7, by decide⟩))) (.add (.add (.input ⟨9, by decide⟩) (.input ⟨1, by decide⟩)) (.add (.input ⟨13, by decide⟩) (.input ⟨5, by decide⟩)))) (.add (.add (.add (.input ⟨10, by decide⟩) (.input ⟨2, by decide⟩)) (.add (.input ⟨14, by decide⟩) (.input ⟨6, by decide⟩))) (.add (.add (.input ⟨8, by decide⟩) (.input ⟨0, by decide⟩)) (.add (.input ⟨12, by decide⟩) (.input ⟨4, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice4 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨12, by decide⟩) (.input ⟨4, by decide⟩)) (.add (.input ⟨8, by decide⟩) (.input ⟨0, by decide⟩))) (.add (.add (.input ⟨14, by decide⟩) (.input ⟨6, by decide⟩)) (.add (.input ⟨10, by decide⟩) (.input ⟨2, by decide⟩)))) (.add (.add (.add (.input ⟨13, by decide⟩) (.input ⟨5, by decide⟩)) (.add (.input ⟨9, by decide⟩) (.input ⟨1, by decide⟩))) (.add (.add (.input ⟨15, by decide⟩) (.input ⟨7, by decide⟩)) (.add (.input ⟨11, by decide⟩) (.input ⟨3, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice5 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨13, by decide⟩) (.input ⟨5, by decide⟩)) (.add (.input ⟨9, by decide⟩) (.input ⟨1, by decide⟩))) (.add (.add (.input ⟨15, by decide⟩) (.input ⟨7, by decide⟩)) (.add (.input ⟨11, by decide⟩) (.input ⟨3, by decide⟩)))) (.add (.add (.add (.input ⟨12, by decide⟩) (.input ⟨4, by decide⟩)) (.add (.input ⟨8, by decide⟩) (.input ⟨0, by decide⟩))) (.add (.add (.input ⟨14, by decide⟩) (.input ⟨6, by decide⟩)) (.add (.input ⟨10, by decide⟩) (.input ⟨2, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice6 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨14, by decide⟩) (.input ⟨6, by decide⟩)) (.add (.input ⟨10, by decide⟩) (.input ⟨2, by decide⟩))) (.add (.add (.input ⟨12, by decide⟩) (.input ⟨4, by decide⟩)) (.add (.input ⟨8, by decide⟩) (.input ⟨0, by decide⟩)))) (.add (.add (.add (.input ⟨15, by decide⟩) (.input ⟨7, by decide⟩)) (.add (.input ⟨11, by decide⟩) (.input ⟨3, by decide⟩))) (.add (.add (.input ⟨13, by decide⟩) (.input ⟨5, by decide⟩)) (.add (.input ⟨9, by decide⟩) (.input ⟨1, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice7 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨15, by decide⟩) (.input ⟨7, by decide⟩)) (.add (.input ⟨11, by decide⟩) (.input ⟨3, by decide⟩))) (.add (.add (.input ⟨13, by decide⟩) (.input ⟨5, by decide⟩)) (.add (.input ⟨9, by decide⟩) (.input ⟨1, by decide⟩)))) (.add (.add (.add (.input ⟨14, by decide⟩) (.input ⟨6, by decide⟩)) (.add (.input ⟨10, by decide⟩) (.input ⟨2, by decide⟩))) (.add (.add (.input ⟨12, by decide⟩) (.input ⟨4, by decide⟩)) (.add (.input ⟨8, by decide⟩) (.input ⟨0, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice8 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨2, by decide⟩) (.input ⟨10, by decide⟩)) (.add (.input ⟨6, by decide⟩) (.input ⟨14, by decide⟩))) (.add (.add (.input ⟨0, by decide⟩) (.input ⟨8, by decide⟩)) (.add (.input ⟨4, by decide⟩) (.input ⟨12, by decide⟩)))) (.add (.add (.add (.input ⟨3, by decide⟩) (.input ⟨11, by decide⟩)) (.add (.input ⟨7, by decide⟩) (.input ⟨15, by decide⟩))) (.add (.add (.input ⟨1, by decide⟩) (.input ⟨9, by decide⟩)) (.add (.input ⟨5, by decide⟩) (.input ⟨13, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice9 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨3, by decide⟩) (.input ⟨11, by decide⟩)) (.add (.input ⟨7, by decide⟩) (.input ⟨15, by decide⟩))) (.add (.add (.input ⟨1, by decide⟩) (.input ⟨9, by decide⟩)) (.add (.input ⟨5, by decide⟩) (.input ⟨13, by decide⟩)))) (.add (.add (.add (.input ⟨2, by decide⟩) (.input ⟨10, by decide⟩)) (.add (.input ⟨6, by decide⟩) (.input ⟨14, by decide⟩))) (.add (.add (.input ⟨0, by decide⟩) (.input ⟨8, by decide⟩)) (.add (.input ⟨4, by decide⟩) (.input ⟨12, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice10 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨4, by decide⟩) (.input ⟨12, by decide⟩)) (.add (.input ⟨0, by decide⟩) (.input ⟨8, by decide⟩))) (.add (.add (.input ⟨6, by decide⟩) (.input ⟨14, by decide⟩)) (.add (.input ⟨2, by decide⟩) (.input ⟨10, by decide⟩)))) (.add (.add (.add (.input ⟨5, by decide⟩) (.input ⟨13, by decide⟩)) (.add (.input ⟨1, by decide⟩) (.input ⟨9, by decide⟩))) (.add (.add (.input ⟨7, by decide⟩) (.input ⟨15, by decide⟩)) (.add (.input ⟨3, by decide⟩) (.input ⟨11, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice11 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨5, by decide⟩) (.input ⟨13, by decide⟩)) (.add (.input ⟨1, by decide⟩) (.input ⟨9, by decide⟩))) (.add (.add (.input ⟨7, by decide⟩) (.input ⟨15, by decide⟩)) (.add (.input ⟨3, by decide⟩) (.input ⟨11, by decide⟩)))) (.add (.add (.add (.input ⟨4, by decide⟩) (.input ⟨12, by decide⟩)) (.add (.input ⟨0, by decide⟩) (.input ⟨8, by decide⟩))) (.add (.add (.input ⟨6, by decide⟩) (.input ⟨14, by decide⟩)) (.add (.input ⟨2, by decide⟩) (.input ⟨10, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice12 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨6, by decide⟩) (.input ⟨14, by decide⟩)) (.add (.input ⟨2, by decide⟩) (.input ⟨10, by decide⟩))) (.add (.add (.input ⟨4, by decide⟩) (.input ⟨12, by decide⟩)) (.add (.input ⟨0, by decide⟩) (.input ⟨8, by decide⟩)))) (.add (.add (.add (.input ⟨7, by decide⟩) (.input ⟨15, by decide⟩)) (.add (.input ⟨3, by decide⟩) (.input ⟨11, by decide⟩))) (.add (.add (.input ⟨5, by decide⟩) (.input ⟨13, by decide⟩)) (.add (.input ⟨1, by decide⟩) (.input ⟨9, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice13 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨7, by decide⟩) (.input ⟨15, by decide⟩)) (.add (.input ⟨3, by decide⟩) (.input ⟨11, by decide⟩))) (.add (.add (.input ⟨5, by decide⟩) (.input ⟨13, by decide⟩)) (.add (.input ⟨1, by decide⟩) (.input ⟨9, by decide⟩)))) (.add (.add (.add (.input ⟨6, by decide⟩) (.input ⟨14, by decide⟩)) (.add (.input ⟨2, by decide⟩) (.input ⟨10, by decide⟩))) (.add (.add (.input ⟨4, by decide⟩) (.input ⟨12, by decide⟩)) (.add (.input ⟨0, by decide⟩) (.input ⟨8, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice14 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨8, by decide⟩) (.input ⟨0, by decide⟩)) (.add (.input ⟨12, by decide⟩) (.input ⟨4, by decide⟩))) (.add (.add (.input ⟨10, by decide⟩) (.input ⟨2, by decide⟩)) (.add (.input ⟨14, by decide⟩) (.input ⟨6, by decide⟩)))) (.add (.add (.add (.input ⟨9, by decide⟩) (.input ⟨1, by decide⟩)) (.add (.input ⟨13, by decide⟩) (.input ⟨5, by decide⟩))) (.add (.add (.input ⟨11, by decide⟩) (.input ⟨3, by decide⟩)) (.add (.input ⟨15, by decide⟩) (.input ⟨7, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choice15 : ReductionPlan 16 where
  tree := .add (.add (.add (.add (.input ⟨9, by decide⟩) (.input ⟨1, by decide⟩)) (.add (.input ⟨13, by decide⟩) (.input ⟨5, by decide⟩))) (.add (.add (.input ⟨11, by decide⟩) (.input ⟨3, by decide⟩)) (.add (.input ⟨15, by decide⟩) (.input ⟨7, by decide⟩)))) (.add (.add (.add (.input ⟨8, by decide⟩) (.input ⟨0, by decide⟩)) (.add (.input ⟨12, by decide⟩) (.input ⟨4, by decide⟩))) (.add (.add (.input ⟨10, by decide⟩) (.input ⟨2, by decide⟩)) (.add (.input ⟨14, by decide⟩) (.input ⟨6, by decide⟩))))
  padding := 0
  valid := by decide +kernel

def sm90Shared16Choices : List (ReductionPlan 16) :=
  [sm90Shared16Choice0, sm90Shared16Choice1, sm90Shared16Choice2, sm90Shared16Choice3, sm90Shared16Choice4, sm90Shared16Choice5, sm90Shared16Choice6, sm90Shared16Choice7, sm90Shared16Choice8, sm90Shared16Choice9, sm90Shared16Choice10, sm90Shared16Choice11, sm90Shared16Choice12, sm90Shared16Choice13, sm90Shared16Choice14, sm90Shared16Choice15]

/-- Source-layout choice is explicit configuration, not a numerical law. The
selector may differ at each tagged [m,n,K-block] input footprint. -/
def sm90SelectedPlans (choose : ∀ n, (Fin n → Expr) → Nat) : PlanSelector := fun n xs =>
  if h : n = 8 then h.symm ▸ (sm90Shared8Choices.getD (choose n xs) sm90Plan8) else
  if h : n = 16 then h.symm ▸ (sm90Shared16Choices.getD (choose n xs) sm90Plan16) else
  sm90Plans n

#axiomsClean sm90SelectedPlans
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
