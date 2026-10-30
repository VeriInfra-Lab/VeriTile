import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.IO
import VeriTile.Meta.StatementAudit

/-! Historical ordered references had a cast at the final FP32 store that was
absent from the Python source. Freeze those independent formulas here and prove
the previous Real/native mathematical results unchanged. No identity law is
asserted for an arbitrary Algebra. Memory views, input domains and schedules
are unchanged; only the generic ordered reference now matches actual source. -/
namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.PrecisionProfile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open Interface
noncomputable section

def previousOrderedRow (M : Algebra A) (op : Operation) (K r : Nat) {D : Nat}
    (x y : Fin D → A) (i : Fin D) : A :=
  match op with
  | .forward log causal mode =>
    let a := selectedValue M M.negInf K r causal x
    let score := fun j => if mode.present then M.binary (some .fp32) .real .add (a j)
      (maskedValue M M.negInf K r causal y j) else a j
    let z := fun j => M.binary (some .fp32) .real .sub (score j) (rowMax M score)
    let num := fun j => M.unary (some .fp32) .exp (z j)
    let den := rowSum M num
    M.cast (some .fp32) .real .real
      (if log then M.binary (some .fp32) .real .sub (z i) (M.unary (some .fp32) .log den)
       else M.binary (some .fp32) .real .div (num i) den)
  | .backward log causal =>
    let zero := M.literal (some .fp32) .real 0
    let g := selectedValue M zero K r causal x
    let o := selectedValue M zero K r causal y
    let v := if log then M.binary (some .fp32) .real .sub (g i)
        (M.binary (some .fp32) .real .mul (M.unary (some .fp32) .exp (o i)) (rowSum M g))
      else M.binary (some .fp32) .real .mul (o i)
        (M.binary (some .fp32) .real .sub (g i)
          (rowSum M (fun j => M.binary (some .fp32) .real .mul (g j) (o j))))
    M.cast (some .fp32) .real .real (if causal then if i.val ≤ r then v else zero else v)

def previousOrderedTarget (M : Algebra A) (op : Operation) (B Q K D : Nat)
    (encode : RegionName → Nat → V → A) (observe : A → V)
    (v : Inputs V B Q D) : Outputs V B Q D :=
  result fun b r i => observe (previousOrderedRow M op K r
    (fun j => encode "left" (Native.flat Q K b r j.val) (left v b r j))
    (fun j => encode "right" (rightAddress op Q K b r j.val) (right v b r j)) i)


theorem native_row_unchanged (rest : Algebra Graph.Expr) (profile : Graph.Profile)
    (op : Operation) (K r : Nat) {D : Nat} (x y : Fin D → Graph.Expr) (i : Fin D) :
    previousOrderedRow (Graph.algebra rest profile) op K r x y i =
      orderedRow (Graph.algebra rest profile) op K r x y i := by cases op <;> rfl

theorem real_row_unchanged (op : Operation) (K r : Nat) {D : Nat}
    (x y : Fin D → WithBot ℝ) (i : Fin D) :
    previousOrderedRow Forward.TypedExecution.model op K r x y i =
      orderedRow Forward.TypedExecution.model op K r x y i := by cases op <;> rfl

theorem native_target_unchanged (rest : Algebra Graph.Expr) (profile : Graph.Profile)
    (op : Operation) (B Q K D : Nat)
    (encode : RegionName → Nat → V → Graph.Expr) (observe : Graph.Expr → V) :
    previousOrderedTarget (Graph.algebra rest profile) op B Q K D encode observe =
      orderedTarget (Graph.algebra rest profile) op B Q K D encode observe := by cases op <;> rfl

theorem real_target_unchanged (op : Operation) (B Q K D : Nat)
    (encode : RegionName → Nat → V → WithBot ℝ) (observe : WithBot ℝ → V) :
    previousOrderedTarget Forward.TypedExecution.model op B Q K D encode observe =
      orderedTarget Forward.TypedExecution.model op B Q K D encode observe := by cases op <;> rfl

theorem native_formula_unchanged (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (op : Operation) (B Q K D : Nat) (dst leftPtr rightPtr : BitVec 64)
    (code : RawKernel.Selected platform op Q K D dst leftPtr rightPtr) :
    previousOrderedTarget (Graph.algebra rest code.lowering.profile) op B Q K D
        nativeEncode (Graph.eval (platform.instructions code.object.primitives)) =
      nativeFormula rest platform op B Q K D dst leftPtr rightPtr code :=
  native_target_unchanged rest code.lowering.profile op B Q K D _ _

#axiomsClean native_row_unchanged
#axiomsClean real_row_unchanged
#axiomsClean native_target_unchanged
#axiomsClean real_target_unchanged
#axiomsClean native_formula_unchanged
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.PrecisionProfile
