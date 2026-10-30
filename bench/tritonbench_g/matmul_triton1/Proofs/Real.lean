import bench.tritonbench_g.matmul_triton1.Proofs.RealSum
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

def extend (a : Fin n → ℝ) (k : Nat) : ℝ := if h:k<n then a ⟨k,h⟩ else 0
def realPrimitives : Primitives ℝ where
  zero := 0
  dot := fun {k} a b z => z+RealSum.sum (fun i => extend a i*extend b i) k

/-- The mathematical matrix product uses only the two original logical inputs
and K. It mentions neither block padding, launch order nor source execution. -/
def realProduct (p : Parameters) (s : State ℝ) (r c k : Nat) : ℝ :=
  (s.mem p.input (r*p.inputRow+k*p.inputInner)).read .real *
  (s.mem p.weight (k*p.weightInner+c*p.weightColumn)).read .real
def realMatmul (p : Parameters) (s : State ℝ) (r c : Nat) : ℝ :=
  RealSum.sum (realProduct p s r c) (inner p)

theorem real_step (p : Parameters) (s : State ℝ) (r c block : Nat) (z : ℝ) :
    realPrimitives.dot (logicalA realPrimitives p s r block)
      (logicalB realPrimitives p s c block) z = z+
        RealSum.sum (fun k => if block*p.blockInner+k< inner p then
          realProduct p s r c (block*p.blockInner+k) else 0) p.blockInner := by
  dsimp only [realPrimitives]
  congr 1
  apply RealSum.sum_congr
  intro k hk
  simp only [extend,dif_pos hk,logicalA,logicalB]
  have hc : k< inner p-block*p.blockInner ↔ block*p.blockInner+k< inner p := by omega
  simp only [hc]
  split_ifs <;> simp [realProduct]

theorem real_accumulation (p : Parameters) (s : State ℝ) (r c n : Nat) :
    logicalAccum realPrimitives p s r c n =
      RealSum.sum (fun k => if k< inner p then realProduct p s r c k else 0) (n*p.blockInner) := by
  induction n with
  | zero => simp [logicalAccum,realPrimitives,RealSum.sum]
  | succ n ih =>
    simp only [logicalAccum,real_step,ih,Nat.succ_mul,RealSum.sum_add]

theorem logical_realMatmul (p : Parameters) (s : State ℝ) (r c : Nat)
    (hBK : 0<p.blockInner) : logical realPrimitives p s r c=realMatmul p s r c := by
  unfold logical
  rw [real_accumulation,RealSum.sum_padding]
  have h := RealSum.ceil_cover (inner p) p.blockInner hBK
  change inner p≤steps p*p.blockInner at h
  rw [Nat.min_eq_left h]
  rfl

#specNonCircular realMatmul avoiding [source,FP.Structural.exec,logical,logicalAccum]
#axiomsClean real_step
#axiomsClean real_accumulation
#axiomsClean logical_realMatmul
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
