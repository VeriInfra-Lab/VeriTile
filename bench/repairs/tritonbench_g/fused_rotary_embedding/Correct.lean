import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Contract
import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.HistoricalCorrect

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding.Correct
open Checked
noncomputable section

structure Result where
  value : Option Unit
  tensors : Region → Tensor
  deviceRestored : Nat
  memory : Memory ℝ

def invoke (c : Call) (callerDevice : Nat) (metadata : Physical.Bytes) (initial : Memory ℝ) : Option Result :=
  if check c metadata then
    let result := Host.mathematical (host c metadata) callerDevice initial
    some ⟨result.value,c.tensor,result.deviceRestored,result.memory⟩
  else none

/-- Independent full split-half rotation and K/V scatter. Both cache layouts,
all Q/K heads, the None return, original Tensor descriptors, caller device,
and every untouched cell are included. This is the exact-real projection. -/
theorem fused_rotary_real (c : Call) (callerDevice : Nat) (metadata : Physical.Bytes)
    (initial : Memory ℝ) (admitted : check c metadata=true) :
    ∃ result,invoke c callerDevice metadata initial=some result ∧ result.value=none ∧
      result.tensors=c.tensor ∧ result.deviceRestored=callerDevice ∧
      result.memory=actualSource (host c metadata).config (fun _=>realOps) initial ∧
      (∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width →
        result.memory kind.output (logicalOutput (host c metadata).config kind t h d)=
          formula c.shape initial kind t h d) ∧
      (∀ r a,¬Footprint (host c metadata).config r a → result.memory r a=initial r a) ∧
      (c.shape.tokens=0 → result.memory=initial) := by
  have h := whole_real (host c metadata) (host_admitted c metadata admitted) callerDevice initial
  refine ⟨⟨none,c.tensor,callerDevice,(Host.mathematical (host c metadata) callerDevice initial).memory⟩,
    by simp [invoke,admitted,Host.mathematical],rfl,rfl,rfl,h.source,h.values,h.frame,?_⟩
  intro empty
  simp [Host.mathematical,Host.empty,host,Physical.configuration,empty]

#axiomsClean fused_rotary_real
#specNonCircular formula avoiding [invoke,Host.mathematical,actualSource]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRotaryEmbedding.Correct
