import bench.tritonbench_g.triton_attention.Proofs.NativeTarget
import bench.tritonbench_g.triton_attention.Proofs.PreprocessReduction
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000

def PreReady (c : Shape) (s : State Expr) (out grad : Nat → Nat → Graph) (den : Nat → Graph) : Prop :=
  (∀ i : PreLane (preParameters c),(s.mem "Out" (preOffset (preParameters c) s i)).read .fp32=
    fun _=>Stage.ready (out (preRow (preParameters c) s (i.1,())) i.2.1.val)) ∧
  (∀ i : PreLane (preParameters c),(s.mem "DO" (preOffset (preParameters c) s i)).read .fp32=
    fun _=>Stage.ready (grad (preRow (preParameters c) s (i.1,())) i.2.1.val)) ∧
  (∀ i : PreRow (preParameters c),(s.mem "L" (preRow (preParameters c) s i)).read .fp32=
    fun _=>Stage.ready (den (preRow (preParameters c) s i)))

theorem preQuotient_native (c : Shape) (one : Bool) (s : State Expr)
    (out grad : Nat → Nat → Graph) (den : Nat → Graph) (h : PreReady c s out grad den) (i : PreLane (preParameters c)) :
    preQuotient (modelScalar one .preprocess) (preParameters c) s i=
      fun _=>Stage.ready (.div (grad (preRow (preParameters c) s (i.1,())) i.2.1.val) (den (preRow (preParameters c) s (i.1,())))) := by
  unfold preQuotient preInput preDenominator
  change (fun u=>Stage.ready (.div
    (lowerScalar one .preprocess (((s.mem "DO" (preOffset (preParameters c) s i)).read .fp32) u))
    (lowerScalar one .preprocess (((s.mem "L" (preRow (preParameters c) s (i.1,()))).read .fp32) u))))=_
  rw [h.2.1 i,h.2.2 (i.1,())]
  rfl

theorem preProducts_native (c : Shape) (one : Bool) (s : State Expr)
    (out grad : Nat → Nat → Graph) (den : Nat → Graph) (h : PreReady c s out grad den) (i : PreLane (preParameters c)) :
    preProducts (modelScalar one .preprocess) (preParameters c) s i=
      fun _=>Stage.mul (.ready (out (preRow (preParameters c) s (i.1,())) i.2.1.val))
        (.ready (.div (grad (preRow (preParameters c) s (i.1,())) i.2.1.val) (den (preRow (preParameters c) s (i.1,()))))) := by
  unfold preProducts
  rw [preQuotient_native c one s out grad den h]
  change (fun u=>Stage.mul (((s.mem "Out" (preOffset (preParameters c) s i)).read .fp32) u) _) =_
  rw [h.1 i]

/-- Congruence follows the exact shuffle tree and only its reached leaves. -/
theorem shuffle_congr_leaves (masks : List Nat) (f g : Nat → Graph) (leaves : Nat → List Nat) (P : Nat → Prop)
    (hfg : ∀ o,(∀ j∈leaves o,P j) → f o=g o) (owner : Nat)
    (h : ∀ j∈TritonAttention.Whole001.Native.shuffleLeaves masks leaves owner,P j) :
    TritonAttention.Whole001.Native.shuffleAdd ops masks f owner=
      TritonAttention.Whole001.Native.shuffleAdd ops masks g owner := by
  induction masks generalizing f g leaves owner with
  | nil=>exact hfg owner h
  | cons mask masks ih=>
    apply ih (fun o=>.add (f o) (f (o ^^^ mask))) (fun o=>.add (g o) (g (o ^^^ mask)))
      (fun o=>leaves o++leaves (o ^^^ mask))
    · intro o ho
      rw [hfg o (by intro j hj;exact ho j (List.mem_append_left _ hj)),
        hfg (o ^^^ mask) (by intro j hj;exact ho j (List.mem_append_right _ hj))]
    · exact h

theorem preDelta_native (c : Shape) (one : Bool) (s : State Expr)
    (out grad : Nat → Nat → Graph) (den : Nat → Graph) (h : PreReady c s out grad den)
    (i : PreRow (preParameters c)) (u : Nat) :
    lowerScalar one .preprocess (preDelta (modelScalar one .preprocess) (preParameters c) s i u)=
      TritonAttention.Whole001.Native.shuffleAdd ops (TritonAttention.Whole001.Native.preMasks c.dim)
        (TritonAttention.Whole001.Native.preLocal ops (out (preRow (preParameters c) s i))
          (grad (preRow (preParameters c) s i)) (den (preRow (preParameters c) s i))) (u%(c.dim/4)) := by
  let row:=preRow (preParameters c) s i
  let values : Nat → Stage:=fun j=>if hj : j<c.dim then
    preProducts (modelScalar one .preprocess) (preParameters c) s (i.1,⟨j,hj⟩,()) 0 else .zero
  change TritonAttention.Whole001.Native.shuffleAdd ops (TritonAttention.Whole001.Native.preMasks c.dim)
    (preLocal values) (u%(c.dim/4))=_
  apply shuffle_congr_leaves _ _ _ TritonAttention.Whole001.Native.preLocalLeaves (fun j=>j<c.dim)
  · intro o ho
    have hv (j : Nat) (hj : j<c.dim) : values j=Stage.mul (.ready (out row j)) (.ready (.div (grad row j) (den row))) := by
      simp only [values,dif_pos hj,preProducts_native c one s out grad den h]
      rfl
    have h0 : o*4<c.dim:=ho _ (by simp [TritonAttention.Whole001.Native.preLocalLeaves])
    have h1 : o*4+1<c.dim:=ho _ (by simp [TritonAttention.Whole001.Native.preLocalLeaves])
    have h2 : o*4+2<c.dim:=ho _ (by simp [TritonAttention.Whole001.Native.preLocalLeaves])
    have h3 : o*4+3<c.dim:=ho _ (by simp [TritonAttention.Whole001.Native.preLocalLeaves])
    simp only [preLocal,hv _ h0,hv _ h1,hv _ h2,hv _ h3]
    rfl
  · intro j hj
    have cov:=TritonAttention.Whole001.Native.preprocess_coverage c u
    exact List.mem_range.mp (cov.mem_iff.mp hj)

#axiomsClean preDelta_native
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
