import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativePhysical
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Numerical

namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativeInstructions
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word MMAWidth ScalarOps dotWord)
noncomputable section

/-- The actual SM90 source profile has QK N=128 and PV N=D. This bounded
selector contains no floating result, plan callback or arbitrary-size dot. -/
def matrixWidth (n : Nat) : MMAWidth :=
  if n≤16 then .n16 else if n≤32 then .n32 else if n≤64 then .n64 else .n128

/-- Each arbitrary source K axis becomes an explicit chain of fixed k8
WGMMA instructions, retaining its zero or live FP32 accumulator. -/
def dotAt (ops : ScalarOps) : (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Word) → (TileIndex (batch++[k,n]) → Word) →
    Bool → Word → TileIndex (batch++[m,n]) → Word
  | [],_,k,n,a,b,accumulate,seed,i => dotWord ops (matrixWidth n) i.1.val i.2.1.val k
      (fun j => a ⟨i.1,j,()⟩) (fun j => b ⟨j,i.2.1,()⟩) accumulate seed
  | _::ds,_,_,_,a,b,accumulate,seed,i => dotAt ops ds (fun j => a ⟨i.1,j⟩)
      (fun j => b ⟨i.1,j⟩) accumulate seed i.2

/-- IEEE ordered equality on the actual raw words; both zero signs compare
as equal and every NaN comparison is false. There is no Real equality oracle. -/
def equalWord (x y : Word) : Bool :=
  let ax := x.toNat%2^31
  let ay := y.toNat%2^31
  decide (ax≤0x7f800000 ∧ ay≤0x7f800000 ∧ (x=y ∨ (ax=0 ∧ ay=0)))

/-- Adapter to the preserved historical graph API. The broad historical
dotAcc field is implemented here by fixed instruction chains, so it is not
a deferred numerical primitive of the public selected FP32 theorem. -/
def primitives (ops : ScalarOps) : NativeGraph.Primitives Word where
  leaf := fun _ word => word
  literal := fun _ _ x => ops.literal32 x
  fromNat := fun _ n => ops.literal32 n
  fromInt := fun _ n => ops.literal32 n
  cast := fun _ _ _ x => x
  unary := fun _ op x => if op=.exp2 then ops.exp2 x else 0
  binary := fun _ _ op => match op with
    | .add => ops.add | .sub => ops.sub | .mul => ops.mul
    | .div => ops.div | .max => ops.max | _ => fun _ _ => 0
  fma := fun _ _ => ops.fmaRN
  eq := fun _ _ => equalWord
  dotAcc := fun _ _ _ _ _ batch _ _ _ a b seed i => dotAt ops batch a b Bool.true seed i

/-- Source syntax distinguishes cleared QK from a live PV accumulator,
even when that live accumulator happens to contain a zero word. -/
def accumulate (acc : NativePhysical.Expr) : Bool :=
  match acc with
  | .literal _ _ value => decide (value≠0)
  | _ => Bool.true

theorem qk_cleared (precision : Option ComputeDType) :
    accumulate (.literal precision .fp32 0)=Bool.false := by simp [accumulate]

theorem pv_live (precision : Option ComputeDType) (dtype : FloatDType)
    (x y : NativePhysical.Expr) : accumulate (.product precision dtype x y)=Bool.true := rfl

#axiomsClean qk_cleared
#axiomsClean pv_live

open NativePhysical (Expr Plan Owner evalTree)

/-- Selected native evaluation retains the instruction's explicit scale-d
(clear/live) selector. Historical broad dotAcc APIs are not numerical ports. -/
def evaluate (ops : ScalarOps) (plan : Plan) : Expr → Owner → Word
  | .leaf d w,_ => (primitives ops).leaf d w
  | .literal p d x,_ => (primitives ops).literal p d x
  | .fromNat p x,_ => (primitives ops).fromNat p x
  | .fromInt p x,_ => (primitives ops).fromInt p x
  | e@(.cast p source target x),owner =>
      (primitives ops).cast p source target (evaluate ops plan x (plan.route e .unary owner))
  | e@(.unary p op x),owner => (primitives ops).unary p op (evaluate ops plan x (plan.route e .unary owner))
  | e@(.binary p d op x y),owner => (primitives ops).binary p d op
      (evaluate ops plan x (plan.route e .left owner)) (evaluate ops plan y (plan.route e .right owner))
  | e@(.product p d x y),owner =>
      let a := evaluate ops plan x (plan.route e .left owner)
      let b := evaluate ops plan y (plan.route e .right owner)
      if plan.reverseMul p d x y owner then (primitives ops).binary p d .mul b a else (primitives ops).binary p d .mul a b
  | e@(.fma p d x y acc),owner =>
      let a := evaluate ops plan x (plan.route e .factorLeft owner)
      let b := evaluate ops plan y (plan.route e .factorRight owner)
      let c := evaluate ops plan acc (plan.route e .accumulator owner)
      if plan.reverseFMA p d x y owner then (primitives ops).fma p d b a c else (primitives ops).fma p d a b c
  | e@(.guard p d value test replacement),owner =>
      let x := evaluate ops plan value (plan.route e .guardValue owner)
      if (primitives ops).eq p d x (evaluate ops plan test (plan.route e .guardTest owner)) then
        evaluate ops plan replacement (plan.route e .guardReplacement owner) else x
  | e@(.reduction p d op n trees inputs),owner =>
      evalTree ((primitives ops).binary p d op) ((primitives ops).leaf d)
        (fun i => evaluate ops plan (inputs i) (plan.route e (.reductionInput i.val) owner)) (trees owner).tree
  | e@(.pendingDot p l r out opts batch m k n a b i),owner =>
      dotAt ops batch
        (fun j => evaluate ops plan (a j) (plan.route e (.dotLeft (TileShape.linearIndex _ j)) owner))
        (fun j => evaluate ops plan (b j) (plan.route e (.dotRight (TileShape.linearIndex _ j)) owner))
        Bool.false ((primitives ops).literal p out 0) i
  | e@(.dotAcc p l r out opts batch m k n a b acc i),owner =>
      dotAt ops batch
        (fun j => evaluate ops plan (a j) (plan.route e (.dotLeft (TileShape.linearIndex _ j)) owner))
        (fun j => evaluate ops plan (b j) (plan.route e (.dotRight (TileShape.linearIndex _ j)) owner))
        (accumulate acc) (evaluate ops plan acc (plan.route e .accumulator owner)) i

abbrev ScalarLaws := VeriTile.Bench.Repairs.ContextAttnFwd.Numerical.ScalarLaws

theorem primitives_eq (actual reference : ScalarOps) (laws : ScalarLaws actual reference) :
    primitives actual=primitives reference := by
  rw [laws.operations_eq actual reference]

theorem feature_coverage (k d : Nat) (hd : d<k) :
    d/8 ∈ List.range ((k+7)/8) ∧ d%8<8 ∧ d/8*8+d%8=d :=
  VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.feature_coverage k d hd

theorem feature_unique (d block lane : Nat) (hl : lane<8) (he : block*8+lane=d) :
    block=d/8 ∧ lane=d%8 :=
  VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.feature_unique d block lane hl he

#axiomsClean primitives_eq
#axiomsClean feature_coverage
#axiomsClean feature_unique
#specNonCircular primitives avoiding [FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativeInstructions
