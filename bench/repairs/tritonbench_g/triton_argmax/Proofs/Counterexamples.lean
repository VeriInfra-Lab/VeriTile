import bench.repairs.tritonbench_g.triton_argmax.Proofs.ScalarBinding
import VeriTile.Meta.StatementAudit

/-! Target-first diagnosis. The dimension kernel cannot return the NaN at
index one of [1, NaN], regardless of the local reduction tree: selecting that
NaN makes the strict running-maximum update false. No numerical assumption or
new input guard is introduced. This is not a correctness-completion entry. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Counterexamples
open Source
open VeriTile.Bench.TritonBenchG.MatrixReduction.MachineMax

def fp32 : MidFormat := ⟨"fp32",4,by decide⟩

def scalars : Scalars fp32 where
  gt32 := ScalarBinding.gt32
  eq32 := ScalarBinding.eq32
  negInf32 := 0xff800000
  storeCast := id
  gtMid := ScalarBinding.gt32
  eqMid := ScalarBinding.eq32
  negInfMid := 0xff800000

def input (i : Nat) : Word := if i = 1 then 0x7fc00000 else 0x3f800000

/-- The NaN branch of the independent argmax target, without any reference to
the source reduction tree or update loop. -/
def IsFirstNaN (xs : Nat → Word) (n i : Nat) : Prop :=
  i < n ∧ isNaN .fp32 (xs i) = Bool.true ∧
    ∀ j, j < i → isNaN .fp32 (xs j) = Bool.false

theorem input_first_nan : IsFirstNaN input 2 1 := by
  refine ⟨by decide,by decide,?_⟩
  intro j before
  have atZero : j = 0 := by omega
  subst j
  decide

def dimension (tree : Reduction.Tree blockN) : DimConfig :=
  ⟨1,2,1,fun _ _ _ => tree⟩

theorem nan_comparison_false : ScalarBinding.gt32 0x7fc00000 0xff800000 = Bool.false := by
  decide

/-- Selecting the NaN can never commit its index through the source's strict
update. This fact does not depend on a reduction ordering or a NaN policy. -/
theorem strict_update_rejects_nan (value : Word) (index : Nat)
    (coupled : index = 1 → value = 0x7fc00000) :
    (if ScalarBinding.gt32 value 0xff800000 then index else 0) ≠ 1 := by
  by_cases h : index = 1
  · rw [coupled h,nan_comparison_false]
    simp
  · split <;> omega

/-- For every coupled reduction tree, the actual source graph's one-tile
dimension result differs from the first-NaN index. The Python contract admits
this nonempty contiguous FP32 input; adding a nonNaN guard would change it. -/
theorem dimension_nan_counterexample (tree : Reduction.Tree blockN) :
    dimIndex scalars (dimension tree) input 0 0 ≠ 1 := by
  have tiles : dimTiles (dimension tree) = [0] := by
    change List.range ((2+128-1)/128) = [0]
    decide
  rw [dimIndex,tiles,List.foldl_cons,List.foldl_nil,dimStep]
  simp only [apply_ite]
  change (if ScalarBinding.gt32 (dimLocal scalars (dimension tree) input 0 0 0).1 0xff800000
    then (dimLocal scalars (dimension tree) input 0 0 0).2 else 0) ≠ 1
  apply strict_update_rejects_nan
  intro index
  let selected := tree.pick ScalarBinding.gt32 ScalarBinding.eq32
    ((dimension tree).loaded input 0xff800000 0 0 0)
  have value : selected.val = 1 := by
    simpa only [dimLocal,dimension,scalars,Nat.zero_mul,Nat.zero_add] using index
  change ((dimension tree).loaded input 0xff800000 0 0 0 selected) = 0x7fc00000
  simp [DimConfig.loaded,dimension,value,dimInputOffset,input]

#axiomsClean input_first_nan
#axiomsClean nan_comparison_false
#axiomsClean strict_update_rejects_nan
#axiomsClean dimension_nan_counterexample
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.TritonArgmax.Counterexamples
