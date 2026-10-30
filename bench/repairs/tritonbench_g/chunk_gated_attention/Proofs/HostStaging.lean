import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Host
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Address

/-! Ordinary coordinate-preserving tensor-library materialization contracts,
then a proof that the staged physical closed form equals independent logical
math. No premise supplies execution, kernel output or a contraction equation. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostStaging
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

structure Staged (p : Layout) (l : Launch) (a : Inputs) (x : Host.InnerInputs)
    (T K V : Nat) (gateK : Bool) : Prop where
  key_le : K ≤ p.K
  value_le : V ≤ p.V
  time_extent : p.T = p.NT*p.BT
  positive_time : 0 < p.BT
  key_stride : p.keyTime = p.K*p.keyFeature
  value_stride : p.valueTime = p.V*p.valueFeature
  key : ∀ t k, t < p.T → k < K →
    a.key (keyAddress p l k t) = if t < T then x.key t k else 0
  value : ∀ t v, t < p.T → v < V →
    a.value (valueAddress p l t v) = if t < T then x.value t v else 0
  gate_key : gateK = Bool.true → ∀ t k, t < p.T → k < K →
    a.gate (keyAddress p l k t) = Host.repeatGate x.cumulativeGate T t k
  gate_value : gateK = Bool.false → ∀ t v, t < p.T → v < V →
    a.gate (valueAddress p l t v) = Host.repeatGate x.cumulativeGate T t v
  seed : ∀ k v, k < K → v < V → a.seed (stateAddress p l k v) = x.seed k v

theorem end_lt (p : Layout) (l : Launch) (a : Inputs) (x : Host.InnerInputs)
    (T K V : Nat) (gateK : Bool) (h : Staged p l a x T K V gateK)
    (c : Nat) (hc : c < p.NT) : c*p.BT+p.BT-1 < p.T := by
  have hb := Nat.mul_le_mul_right p.BT (Nat.succ_le_iff.mpr hc)
  have hp := h.positive_time
  rw [h.time_extent]
  simp only [Nat.succ_mul] at hb
  omega

theorem endpoint_eq (p : Layout) (l : Launch) (a : Inputs) (x : Host.InnerInputs)
    (T K V : Nat) (gateK : Bool) (h : Staged p l a x T K V gateK)
    (c k v : Nat) (hc : c < p.NT) (hk : k < K) (hv : v < V) :
    endpoint p l a gateK c k v =
      x.cumulativeGate (Host.endpoint T p.BT c) (if gateK then k else v) := by
  have he := end_lt p l a x T K V gateK h c hc
  have hkp := lt_of_lt_of_le hk h.key_le
  have hvp := lt_of_lt_of_le hv h.value_le
  cases gateK
  · have hn := Address.flatten2_bounded he hvp
    simp only [endpoint,endpointIndex,Bool.false_eq_true,↓reduceIte,hn,if_true]
    rw [show l.head*p.valueHead+((c*p.BT+p.BT-1)*p.V+v)*p.valueFeature =
        valueAddress p l (c*p.BT+p.BT-1) v by
          simp only [valueAddress,h.value_stride]; ring]
    rw [h.gate_value rfl _ _ he hv,Host.repeated_endpoint_is_clipped]
  · have hn := Address.flatten2_bounded he hkp
    simp only [endpoint,endpointIndex,↓reduceIte,hn,if_true]
    rw [show l.head*p.keyHead+((c*p.BT+p.BT-1)*p.K+k)*p.keyFeature =
        keyAddress p l k (c*p.BT+p.BT-1) by
          simp only [keyAddress,h.key_stride]; ring]
    rw [h.gate_key rfl _ _ he hk,Host.repeated_endpoint_is_clipped]

theorem contribution_eq (p : Layout) (l : Launch) (a : Inputs) (x : Host.InnerInputs)
    (T K V : Nat) (gateK : Bool) (h : Staged p l a x T K V gateK)
    (c k v : Nat) (hc : c < p.NT) (hk : k < K) (hv : v < V) :
    contribution p l a gateK c k v = Host.innerContribution x T p.BT gateK c k v := by
  have hkp := lt_of_lt_of_le hk h.key_le
  have hvp := lt_of_lt_of_le hv h.value_le
  unfold contribution Host.innerContribution
  apply Finset.sum_congr rfl
  intro lane _
  have ht : timeIndex p c lane.val < p.T := by
    simpa [timeIndex,h.time_extent] using Address.flatten2_bounded hc lane.isLt
  rw [endpoint_eq p l a x T K V gateK h c k v hc hk hv]
  simp only [keyValue,valueValue,hkp,hvp,ht,and_self,if_true,h.key _ _ ht hk,h.value _ _ ht hv]
  simp only [timeIndex] at ht ⊢
  by_cases hvalid : c*p.BT+lane.val < T
  · cases gateK
    · simp [gateValue,h.gate_value rfl _ _ ht hv,hvp,ht,Host.repeatGate,hvalid]
    · simp [gateValue,h.gate_key rfl _ _ ht hk,hkp,ht,Host.repeatGate,hvalid]
  · simp [hvalid]

theorem factor_eq (p : Layout) (l : Launch) (a : Inputs) (x : Host.InnerInputs)
    (T K V : Nat) (gateK : Bool) (h : Staged p l a x T K V gateK)
    (c k v : Nat) (hc : c < p.NT) (hk : k < K) (hv : v < V) :
    gateFactor p l a gateK c k v = Host.innerWeight x T p.BT gateK c k v := by
  unfold gateFactor Host.innerWeight
  rw [endpoint_eq p l a x T K V gateK h c k v hc hk hv]

theorem closed_eq (p : Layout) (l : Launch) (a : Inputs) (x : Host.InnerInputs)
    (T K V : Nat) (gateK hasSeed : Bool) (h : Staged p l a x T K V gateK)
    (c k v : Nat) (hc : c ≤ p.NT) (hk : k < K) (hv : v < V) :
    closed p l a gateK hasSeed c k v = Host.innerState x T p.BT gateK hasSeed c k v := by
  have hkp := lt_of_lt_of_le hk h.key_le
  have hvp := lt_of_lt_of_le hv h.value_le
  have hs : seed p l a hasSeed k v = if hasSeed then x.seed k v else 0 := by
    cases hasSeed <;> simp [seed,hkp,hvp,h.seed k v hk hv]
  unfold closed Host.innerState
  rw [hs]
  congr 1
  · congr 1
    apply Finset.prod_congr rfl
    intro j hj
    exact factor_eq p l a x T K V gateK h j k v (lt_of_lt_of_le (Finset.mem_range.mp hj) hc) hk hv
  · apply Finset.sum_congr rfl
    intro j hj
    rw [contribution_eq p l a x T K V gateK h j k v (lt_of_lt_of_le (Finset.mem_range.mp hj) hc) hk hv]
    congr 1
    apply Finset.prod_congr rfl
    intro z hz
    exact factor_eq p l a x T K V gateK h z k v (lt_of_lt_of_le (Finset.mem_Ico.mp hz).2 hc) hk hv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HostStaging
