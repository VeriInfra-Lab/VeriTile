import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeIndex
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.PublicCheck
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.LayerNormLiger.Wrapper
open VeriTile.Triton FP.Structural VeriTile.Torch
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section

/-- Actual regions returned by the original contiguous/view preparation.
These inputs are read only and may alias one another. -/
structure Inputs where
  (x weight bias dy : RegionName)

/-- Source allocation order: Y, Mean, RSTD, DX, partial DW, partial DB,
returned DW, returned DB. The final two are the real Torch sum/cast results. -/
structure Allocations where
  region : Fin 8 → RegionName

structure Fresh (r : Inputs) (a : Allocations) : Prop where
  distinct : Function.Injective a.region
  input : ∀ i q, (q ∈ [r.x,r.weight,r.bias] ∨ (3 ≤ i.val ∧ q = r.dy)) → q ≠ a.region i

theorem Fresh.ne (r : Inputs) (a : Allocations) (f : Fresh r a) (i j : Fin 8) (h : i ≠ j) :
    a.region i ≠ a.region j := fun eq => h (f.distinct eq)

def forward (r : Inputs) (a : Allocations) (x : TensorMetadata) (eps : Contract.Epsilon) : Forward.Parameters :=
  ⟨a.region 0,r.x,r.weight,r.bias,a.region 1,a.region 2,
    Contract.width x,Contract.width x,1,1,1,1,Contract.width x,Settings.block (Contract.width x),eps.value⟩

def backward (r : Inputs) (a : Allocations) (x : TensorMetadata) (programs : Nat) : Backward.Parameters :=
  ⟨r.x,r.weight,a.region 1,a.region 2,a.region 3,a.region 4,a.region 5,r.dy,
    Contract.width x,Contract.width x,Contract.width x,Contract.width x,Contract.width x,
    Contract.rows x,Contract.width x,Launch.rowsPerProgram (Contract.rows x) programs,Settings.block (Contract.width x)⟩

theorem backward_fresh (r : Inputs) (a : Allocations) (x : TensorMetadata) (p : Nat) (f : Fresh r a) :
    Returned.Fresh (backward r a x p) := by
  have inputs (i : Fin 8) (hi : 3 ≤ i.val) (q : RegionName) (h : q = r.x ∨ q = r.weight ∨ q = r.dy) : q ≠ a.region i := by
    apply f.input i q
    rcases h with rfl | rfl | rfl <;> simp [hi]
  constructor <;> intro q hq <;> simp only [Logical.Reads,backward] at hq ⊢
  all_goals rcases hq with h | h | h | rfl | rfl
  all_goals first
    | exact inputs _ (by decide) _ (.inl h)
    | exact inputs _ (by decide) _ (.inr (.inl h))
    | exact inputs _ (by decide) _ (.inr (.inr h))
    | exact Fresh.ne r a f _ _ (by decide)

theorem link (r : Inputs) (a : Allocations) (x : TensorMetadata) (eps : Contract.Epsilon)
    (p : Nat) (f : Fresh r a) : Public.Link (forward r a x eps) (backward r a x p) (Contract.rows x) := by
  refine ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,le_rfl,?_,?_,?_,?_⟩
  · intro q hq
    have input : q ∈ [r.x,r.weight,r.bias] := by
      simp only [ForwardSaved.Reads,forward] at hq
      rcases hq with rfl | rfl | rfl <;> simp
    exact ⟨f.input 1 q (.inl input),f.input 2 q (.inl input),f.input 0 q (.inl input)⟩
  all_goals exact Fresh.ne r a f _ _ (by decide)

theorem layout (r : Inputs) (a : Allocations) (x w b dy : TensorMetadata) (eps : Contract.Epsilon)
    (p : Nat) (hp : 0 < p) (h : Contract.InputContract x w b eps)
    (g : Contract.GradientContract dy x) (f : Fresh r a) :
    Public.Layout (backward r a x p) p (a.region 6) (a.region 7) dy.shape := by
  refine ⟨hp,Contract.rows_positive x w b eps h,h.width_pos,h.width_bound,
    Settings.nextPower_covers _,rfl,rfl,rfl,rfl,g.elements.trans (Contract.shape_elements x h.rank),
    backward_fresh r a x p f,?_,?_,?_,?_,?_,?_,?_⟩
  all_goals exact Fresh.ne r a f _ _ (by decide)

/-- Successful source allocation and input preparation expose ordinary byte
spans. No new size guard is assumed: each capacity follows from storage extent. -/
structure Storage (rows columns programs : Nat) where
  inputBase : Nat
  partialBase : Nat
  inputSpan : inputBase + 4 * (rows * columns) ≤ 2^64
  partialSpan : partialBase + 4 * (programs * columns) ≤ 2^64

theorem capacity (rows columns p : Nat) (storage : Storage rows columns p) :
    NativeIndex.Capacity rows columns p := by
  have hi := storage.inputSpan
  have hp := storage.partialSpan
  constructor <;> omega

def checked {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α) (scalar : Host.Scalar α)
    (x w b dy : TensorMetadata) (eps : Contract.Epsilon) (r : Inputs) (a : Allocations)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs) (s : State α) : Option (Host.Return α) :=
  match Contract.checkInputs x w b eps, Contract.checkGradient dy x with
  | .ok _, .ok _ => if programs = 0 then none else
      Host.forwardBackward M R scalar (forward r a x eps) (backward r a x programs)
        (Contract.rows x) programs pw pb (a.region 6) (a.region 7) dy.shape s
  | _, _ => none

theorem checked_eq {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α) (scalar : Host.Scalar α)
    (x w b dy : TensorMetadata) (eps : Contract.Epsilon) (r : Inputs) (a : Allocations)
    (p : Nat) (pw pb : ReductionPlan.Plan p) (s : State α)
    (h : Contract.InputContract x w b eps) (g : Contract.GradientContract dy x) (hp : 0 < p) :
    checked M R scalar x w b dy eps r a p pw pb s =
      Host.forwardBackward M R scalar (forward r a x eps) (backward r a x p)
        (Contract.rows x) p pw pb (a.region 6) (a.region 7) dy.shape s := by
  simp only [checked,(Contract.checkInputs_iff x w b eps).mpr h,(Contract.checkGradient_iff dy x).mpr g]
  simp [Nat.ne_of_gt hp]

theorem native (x w b : TensorMetadata) (eps : Contract.Epsilon) (p : Nat)
    (h : Contract.InputContract x w b eps) (hp : 0 < p)
    (storage : Storage (Contract.rows x) (Contract.width x) p) :
    NativeIndex.Safe (Contract.rows x) (Contract.width x) p (Settings.block (Contract.width x)) :=
  NativeIndex.safe _ _ _ _ (Contract.rows_positive x w b eps h) h.width_pos hp
    (Settings.nextPower_bound _ h.width_bound) (capacity _ _ _ storage)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.Wrapper
