import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.StagedLayout

/-! Ordinary coordinate-preserving tensor copies, zero fill and repeated gate
tails construct the staging laws. Logical inputs are snapshots, so the proof
also applies when the caller's final-state view aliases an input allocation. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.Materialize
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def dense (times features : Nat) (f : Nat → Nat → Nat → ℝ) (off : Nat) : ℝ :=
  f (off/(features*times)) ((off/features)%times) (off%features)

theorem dense_read (times features head time feature : Nat) (f : Nat → Nat → Nat → ℝ)
    (ht : time < times) (hf : feature < features) :
    dense times features f (head*(times*features)+time*features+feature) = f head time feature := by
  have h := Grid.decode_encode (z := head) hf ht
  have he : head*(times*features)+time*features+feature = Grid.encode features times feature time head := by
    unfold Grid.encode
    ring
  simp only [dense,he]
  simp only [Grid.decodeX,Grid.decodeY,Grid.decodeZ] at h
  rw [h.2.2,h.2.1,h.1]

def inputs (p : Layout) (T K V : Nat) (gateK : Bool) (x : Nat → Host.InnerInputs) : Inputs where
  key := dense p.T p.K (fun h => Host.zeroPad (x h).key T K)
  value := dense p.T p.V (fun h => Host.zeroPad (x h).value T V)
  gate := dense p.T (if gateK then p.K else p.V)
    (fun h t f => if f < (if gateK then K else V) then Host.repeatGate (x h).cumulativeGate T t f else 0)
  seed := dense p.K p.V (fun h k v => if k < K ∧ v < V then (x h).seed k v else 0)

theorem staged (p : Layout) (l : Launch) (T K V : Nat) (gateK : Bool)
    (x : Nat → Host.InnerInputs) (hp : StagedLayout.Canonical p) (hk : K ≤ p.K) (hv : V ≤ p.V) :
    HostStaging.Staged p l (inputs p T K V gateK x) (x l.head) T K V gateK := by
  refine ⟨hk,hv,hp.time,hp.bt,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [hp.keyTime,hp.keyFeature]
  · simp [hp.valueTime,hp.valueFeature]
  · intro t k ht hkl
    have he : keyAddress p l k t = l.head*(p.T*p.K)+t*p.K+k := by
      simp only [keyAddress,hp.keyHead,hp.keyTime,hp.keyFeature]
      ring
    simp only [inputs,he,dense_read _ _ _ _ _ _ ht (lt_of_lt_of_le hkl hk),Host.zeroPad,hkl,and_true]
  · intro t v ht hvl
    simp only [inputs,valueAddress,hp.valueHead,hp.valueTime,hp.valueFeature,Nat.mul_one,
      dense_read _ _ _ _ _ _ ht (lt_of_lt_of_le hvl hv),Host.zeroPad,hvl,and_true]
  · intro h t k ht hkl
    subst gateK
    have he : keyAddress p l k t = l.head*(p.T*p.K)+t*p.K+k := by
      simp only [keyAddress,hp.keyHead,hp.keyTime,hp.keyFeature]
      ring
    simp only [inputs,↓reduceIte,he,dense_read _ _ _ _ _ _ ht (lt_of_lt_of_le hkl hk),hkl,if_true]
  · intro h t v ht hvl
    subst gateK
    simp only [inputs,Bool.false_eq_true,↓reduceIte,valueAddress,hp.valueHead,hp.valueTime,
      hp.valueFeature,Nat.mul_one,dense_read _ _ _ _ _ _ ht (lt_of_lt_of_le hvl hv),hvl,if_true]
  · intro k v hkl hvl
    rw [show stateAddress p l k v = l.head*(p.K*p.V)+k*p.V+v by simp [stateAddress,Nat.mul_assoc]]
    simp only [inputs,dense_read _ _ _ _ _ _ (lt_of_lt_of_le hkl hk) (lt_of_lt_of_le hvl hv),hkl,hvl,and_self,if_true]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.Materialize
