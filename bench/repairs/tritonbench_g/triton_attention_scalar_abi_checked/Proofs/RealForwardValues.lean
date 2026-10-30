import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealBackend
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealMath
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- An interpretation of input loads only. No source result occurs here. -/
def ForwardReads (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ) : Prop :=
  ∀ r ∈ [p.q,p.k,p.v], ∀ off,
    toFloat p.dtype ((s.mem r off).read p.dtype.toTileDType)=some (μ r off)
def qAt (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ) (i d : Nat) : ℝ :=
  μ p.q ((forwardHeadRow p s+i)*p.head+d)
def kAt (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ) (j d : Nat) : ℝ :=
  μ p.k ((forwardHeadRow p s+j)*p.head+d)
def vAt (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ) (j d : Nat) : ℝ :=
  μ p.v ((forwardHeadRow p s+j)*p.head+d)
def logicalScore (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (i : Fin p.blockM) (j : Nat) : ℝ :=
  Mathematical.score p.head p.scale (qAt p s μ) (kAt p s μ) (forwardQueryRow p s i) j

theorem query_real (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (i : FQuery p) :
    toFloat p.dtype (forwardQ p s i)=some (qAt p s μ (forwardQueryRow p s i.1) i.2.1.val) := by
  exact h p.q (by simp) _
theorem kv_real (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (r : RegionName) (hr : r ∈ [p.q,p.k,p.v])
    (c : Nat) (i : FKey p) (hb : forwardHeadRow p s+c*p.blockN+i.1.val<p.rows) :
    toFloat p.dtype (forwardKV p s r c i)=some (μ r (forwardKVOffset p s c i)) := by
  simp only [forwardKV,forwardKVPointer,TileShape.blockPtr_inBounds_2d_offsets_index,
    Nat.zero_add,show forwardHeadRow p s+c*p.blockN+i.1.val<p.rows ∧ i.2.1.val<p.head from ⟨hb,i.2.1.isLt⟩,
    decide_true,ite_true]
  exact h r hr _
theorem dot_real (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (c : Nat) (i : FScore p)
    (hb : forwardHeadRow p s+c*p.blockN+i.2.1.val<p.rows) :
    forwardDot model p s c i=some (∑ d ∈ Finset.range p.head,
      qAt p s μ (forwardQueryRow p s i.1) d*kAt p s μ (c*p.blockN+i.2.1.val) d) := by
  change sum p.head (fun d=>WithBot.realMul (toFloat p.dtype (forwardQ p s (i.1,d,())))
    (toFloat p.dtype (forwardKV p s p.k c (i.2.1,d,()))))=_
  have he : (fun d : Fin p.head=>WithBot.realMul (toFloat p.dtype (forwardQ p s (i.1,d,())))
      (toFloat p.dtype (forwardKV p s p.k c (i.2.1,d,())))) =
      (fun d=>some (qAt p s μ (forwardQueryRow p s i.1) d.val*kAt p s μ (c*p.blockN+i.2.1.val) d.val)) := by
    funext d
    rw [query_real p s μ h,kv_real p s μ h p.k (by simp) c (i.2.1,d,()) hb]
    simp [kAt,forwardKVOffset,Nat.add_assoc,WithBot.realMul]
  rw [he,sum_some]
  congr 1
  exact Fin.sum_univ_eq_sum_range (fun d : Nat=>qAt p s μ (forwardQueryRow p s i.1) d*kAt p s μ (c*p.blockN+i.2.1.val) d) p.head
theorem score_real (p : ForwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (h : ForwardReads p s μ) (c : Nat) (i : FScore p)
    (hb : forwardHeadRow p s+c*p.blockN+i.2.1.val<p.rows) :
    forwardScore model p s c i = if c*p.blockN+i.2.1.val ≤ forwardQueryRow p s i.1
      then some (logicalScore p s μ i.1 (c*p.blockN+i.2.1.val)) else none := by
  by_cases hv : c*p.blockN+i.2.1.val ≤ forwardQueryRow p s i.1
  · simp only [forwardScore,if_pos hv]
    rw [dot_real p s μ h c i hb]
    simp [forwardZero,model,algebra,logicalScore,Mathematical.score,WithBot.realAdd,WithBot.realMul]
  · simp [forwardScore,hv,model,algebra]

#axiomsClean score_real
#specNonCircular ForwardReads avoiding [FP.Structural.exec,forwardRounds,forwardWrite]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
