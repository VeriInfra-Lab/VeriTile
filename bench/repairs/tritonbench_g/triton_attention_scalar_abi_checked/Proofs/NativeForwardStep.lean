import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativeForwardValues
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000

def replica (u : Nat) : Fin 4 := ⟨u%4,Nat.mod_lt _ (by decide)⟩
@[simp] theorem replica_val (u : Fin 4) : replica u.val=u := Fin.ext (Nat.mod_eq_of_lt u.isLt)

theorem lower_ready_sub (one : Bool) (g : Graph) (a : Stage) :
    lowerScalar one .forward (.sub (.ready g) a)=.sub g (lowerScalar one .forward a) := rfl

def AccAt (c : Shape) (scale : ℝ) (one : Bool) (st : ForwardAccumulator Expr (forwardParameters c scale))
    (row : Fin 128) (col : Fin c.dim) (a : Target.Acc) : Prop :=
  (∀ u,lowerScalar one .forward (st.maximum (row,()) u)=a.maximum (replica u)) ∧
  (∀ u,lowerScalar one .forward (st.denominator (row,()) u)=a.denominator (replica u)) ∧
  (∀ u,lowerScalar one .forward (st.numerator (row,col,()) u)=a.output (replica u)) ∧
  (∀ u b,lowerScalar one .forward (.sub (st.maximum (row,()) u) b)=
    .sub (lowerScalar one .forward (st.maximum (row,()) u)) (lowerScalar one .forward b))

theorem forwardMax_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.Inputs) (head block : Nat) (hq : ReadyQ c scale s x head) (hkv : ReadyKV c scale s x head block)
    (st : ForwardAccumulator Expr (forwardParameters c scale)) (row : Fin 128) (col : Fin c.dim)
    (a : Target.Acc) (ha : AccAt c scale one st row col a) (u : Nat) :
    lowerScalar one .forward (forwardMax (modelScalar one .forward) (forwardParameters c scale) s block st (row,()) u)=
    Target.maximum c one scale x head (forwardQueryRow (forwardParameters c scale) s row) block a (replica u) := by
  change Graph.max ((Reduction.plan (replica u)).eval Graph.max
    (fun j=>if hj : j<128 then lowerScalar one .forward
      (forwardScore (modelScalar one .forward) (forwardParameters c scale) s block (row,⟨j,hj⟩,()) (Reduction.owner j).val) else .word 0))
    (lowerScalar one .forward (st.maximum (row,()) u))=_
  rw [ha.1]
  unfold Target.maximum
  congr 2
  funext j
  by_cases hj : j<128
  · simp only [Target.pad,dif_pos hj,if_pos hj]
    exact forwardScore_native c scale one s x head block hq hkv (row,⟨j,hj⟩,()) _
  · simp only [Target.pad,dif_neg hj,if_neg hj]

theorem forwardScore_sub (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (block : Nat) (i : FScore (forwardParameters c scale)) (u : Nat) (b : Stage) :
    lowerScalar one .forward (.sub (forwardScore (modelScalar one .forward) (forwardParameters c scale) s block i u) b)=
      .sub (lowerScalar one .forward (forwardScore (modelScalar one .forward) (forwardParameters c scale) s block i u))
        (lowerScalar one .forward b) := by
  unfold forwardScore
  split_ifs
  · let value : Stage:=.add (literal 0)
      (forwardDot (modelScalar one .forward) (forwardParameters c scale) s block i u)
    change lowerScalar one .forward (.sub (.mul value (literal scale)) b)=
      .sub (lowerScalar one .forward (.mul value (literal scale))) (lowerScalar one .forward b)
    by_cases h0 : scale=0
    · subst scale
      simp [literal,lowerScalar]
    by_cases h1 : scale=1
    · subst scale
      simp [literal,lowerScalar]
      cases one <;> rfl
    · simp [literal,lowerScalar,h0,h1]
  · rfl

theorem forwardWeights_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.Inputs) (head block : Nat) (hq : ReadyQ c scale s x head) (hkv : ReadyKV c scale s x head block)
    (st : ForwardAccumulator Expr (forwardParameters c scale)) (row : Fin 128) (col : Fin c.dim)
    (a : Target.Acc) (ha : AccAt c scale one st row col a) (j : Fin 128) (u : Nat) :
    lowerScalar one .forward (forwardWeights (modelScalar one .forward) (forwardParameters c scale) s block st (row,j,()) u)=
      Target.exp (.sub (Target.score c one scale x head (forwardQueryRow (forwardParameters c scale) s row) block j.val)
        (Target.maximum c one scale x head (forwardQueryRow (forwardParameters c scale) s row) block a (replica u))) := by
  change Target.exp (lowerScalar one .forward (.sub
    (forwardScore (modelScalar one .forward) (forwardParameters c scale) s block (row,j,()) u)
    (forwardMax (modelScalar one .forward) (forwardParameters c scale) s block st (row,()) u)))=_
  rw [forwardScore_sub,forwardScore_native c scale one s x head block hq hkv,
    forwardMax_native c scale one s x head block hq hkv st row col a ha]

theorem forwardDenominator_native (c : Shape) (scale : ℝ) (one : Bool) (s : State Expr)
    (x : Target.Inputs) (head block : Nat) (hq : ReadyQ c scale s x head) (hkv : ReadyKV c scale s x head block)
    (st : ForwardAccumulator Expr (forwardParameters c scale)) (row : Fin 128) (col : Fin c.dim)
    (a : Target.Acc) (ha : AccAt c scale one st row col a) (u : Nat) :
    lowerScalar one .forward (forwardDenominator (modelScalar one .forward) (forwardParameters c scale) s block st (row,()) u)=
      Target.denominator c one scale x head (forwardQueryRow (forwardParameters c scale) s row) block a (replica u) := by
  change Graph.fma (lowerScalar one .forward (st.denominator (row,()) u))
    (Target.exp (lowerScalar one .forward (.sub (st.maximum (row,()) u)
      (forwardMax (modelScalar one .forward) (forwardParameters c scale) s block st (row,()) u))))
    ((Reduction.plan (replica u)).eval Graph.add (fun j=>if hj : j<128 then lowerScalar one .forward
      (forwardWeights (modelScalar one .forward) (forwardParameters c scale) s block st (row,⟨j,hj⟩,()) (Reduction.owner j).val) else .word 0))=_
  rw [ha.2.1,ha.2.2.2,ha.1,forwardMax_native c scale one s x head block hq hkv st row col a ha]
  unfold Target.denominator
  congr 2
  funext j
  by_cases hj : j<128
  · simp only [Target.pad,dif_pos hj,if_pos hj]
    rw [forwardWeights_native c scale one s x head block hq hkv st row col a ha,replica_val]
    rfl
  · simp only [Target.pad,dif_neg hj,if_neg hj]

#axiomsClean forwardMax_native
#axiomsClean forwardWeights_native
#axiomsClean forwardDenominator_native
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
