import bench.tritonbench_g.triton_attention.Proofs.NativeTarget
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem lower_scale_mul (one : Bool) (kind : Kind) (scale : ℝ) (a : Stage) :
    lowerScalar one kind (.mul a (literal scale))=Target.scaleMul one scale (lowerScalar one kind a) := by
  by_cases h0 : scale=0
  · subst scale
    simp [literal,lowerScalar,Target.scaleMul]
  by_cases h1 : scale=1
  · subst scale
    simp [literal,lowerScalar,Target.scaleMul]
  · simp [literal,lowerScalar,Target.scaleMul,h0,h1]

theorem lower_scale_sub (one : Bool) (scale : ℝ) (a m : Stage) :
    lowerScalar one .backward (.sub (.mul a (literal scale)) m)=
      Target.scaleSubtract one scale (lowerScalar one .backward a) (lowerScalar one .backward m) := by
  by_cases h0 : scale=0
  · subst scale
    simp [literal,lowerScalar,Target.scaleSubtract,Target.scaleWord]
  by_cases h1 : scale=1
  · subst scale
    simp [literal,lowerScalar,Target.scaleSubtract,Target.scaleWord]
  · simp [literal,lowerScalar,Target.scaleSubtract,Target.scaleWord,h0,h1]

def ReadyQ (c : Shape) (scale : ℝ) (s : State Expr) (x : Target.Inputs) (head : Nat) : Prop :=
  ∀ i : FQuery (forwardParameters c scale),forwardQ (forwardParameters c scale) s i=
    fun _=>Stage.ready (x.q head (forwardQueryRow (forwardParameters c scale) s i.1) i.2.1.val)
def ReadyKV (c : Shape) (scale : ℝ) (s : State Expr) (x : Target.Inputs) (head block : Nat) : Prop :=
  (∀ i : FKey (forwardParameters c scale),forwardKV (forwardParameters c scale) s "K" block i=
    fun _=>Stage.ready (x.k head (block*128+i.1.val) i.2.1.val)) ∧
  (∀ i : FKey (forwardParameters c scale),forwardKV (forwardParameters c scale) s "V" block i=
    fun _=>Stage.ready (x.v head (block*128+i.1.val) i.2.1.val))

theorem forwardDot_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.Inputs) (head block : Nat) (hq : ReadyQ c scale s x head) (hkv : ReadyKV c scale s x head block)
    (i : FScore (forwardParameters c scale)) (u : Nat) :
    forwardDot (modelScalar one .forward) (forwardParameters c scale) s block i u=
      .dot .n128 c.dim (Target.pad c.dim (x.q head (forwardQueryRow (forwardParameters c scale) s i.1)))
        (Target.pad c.dim (x.k head (block*128+i.2.1.val))) := by
  unfold forwardDot
  change Stage.dot .n128 c.dim
    (fun d=>if hd : d<c.dim then lowerScalar one .forward
      (forwardQ (forwardParameters c scale) s (i.1,⟨d,hd⟩,()) (Reduction.owner d).val) else .word 0)
    (fun d=>if hd : d<c.dim then lowerScalar one .forward
      (forwardKV (forwardParameters c scale) s "K" block (i.2.1,⟨d,hd⟩,()) (Reduction.owner i.2.1.val).val) else .word 0)=_
  congr 1
  · funext d
    by_cases hd : d<c.dim
    · simp only [Target.pad,dif_pos hd,if_pos hd]
      rw [hq (i.1,⟨d,hd⟩,())]
      rfl
    · simp only [Target.pad,dif_neg hd,if_neg hd]
  · funext d
    by_cases hd : d<c.dim
    · simp only [Target.pad,dif_pos hd,if_pos hd]
      rw [hkv.1 (i.2.1,⟨d,hd⟩,())]
      rfl
    · simp only [Target.pad,dif_neg hd,if_neg hd]

theorem forwardScore_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.Inputs) (head block : Nat) (hq : ReadyQ c scale s x head) (hkv : ReadyKV c scale s x head block)
    (i : FScore (forwardParameters c scale)) (u : Nat) :
    lowerScalar one .forward (forwardScore (modelScalar one .forward) (forwardParameters c scale) s block i u)=
      Target.score c one scale x head (forwardQueryRow (forwardParameters c scale) s i.1) block i.2.1.val := by
  unfold forwardScore Target.score
  simp only [show (forwardParameters c scale).blockN=128 from rfl]
  by_cases hc : block*128+i.2.1.val≤forwardQueryRow (forwardParameters c scale) s i.1
  · rw [if_pos hc,if_pos hc]
    change lowerScalar one .forward (.mul (.add (literal 0)
      (forwardDot (modelScalar one .forward) (forwardParameters c scale) s block i u)) (literal scale))=_
    rw [show literal 0=Stage.zero by simp [literal],forwardDot_native c scale one s x head block hq hkv i u,lower_scale_mul]
    rfl
  · rw [if_neg hc,if_neg hc]
    rfl

#axiomsClean lower_scale_mul
#axiomsClean lower_scale_sub
#axiomsClean forwardScore_native
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
