import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Contents

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Correct
open VeriTile.Triton FP.Structural Strided.Grid Checked
noncomputable section

/-- Independent finite-segment softmax on the original input pointer view. -/
def realTarget (c : TensorCall) (m : Memory Strided.Real.Scalar) (b h n : Nat) :=
  Strided.Host.target (host c.observation) (view c m) b h n

/-- The complete guarded call mutates the supplied Tensor, returns None,
computes every active segment, and preserves all other physical storage cells. -/
theorem tensor_correct (c : TensorCall)
    (m : Memory Strided.Real.Scalar)
    (raw : RegionName → Nat → BitVec 32) (metadata : MetadataPin c raw m)
    (h : InputContract c.observation (observedStart c raw) (observedLength c raw))
    (payload : PayloadCells c m)
    (numeric : Strided.Host.NumericData (host c.observation) (view c m))
    (work : List PID)
    (complete : (fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work) :
    ∃ result, invokeChecked Strided.Real.algebra c work raw m=some result ∧ result.returned=none ∧
      result.output=c.output.tensor ∧
      (contentActive c.observation → ∀ b  <  rows c.start.observation, ∀ h  <  rows c.input.observation,
        ∀ n  <  Strided.Host.seq (host c.observation) (view c m) b,
        WithBot.unbotD 0 ((result.memory (storageRegion c.output.tensor.storage)
          (outputIndex c m b h n)).read .fp32)=realTarget c m b h n) ∧
      (∀ r i,exterior c m r i → result.memory r i=m r i) := by
  simp only [invoke_checked _ c work raw _ h]
  by_cases launch : active c.observation
  · have data := checked_data c raw m h ⟨launch.1,launch.2.1,launch.2.2.1⟩ metadata payload
    obtain ⟨result,run,ret,out,values,frame,safe⟩ := Strided.Host.host_real (host c.observation)
      work (view c m) (checked_layout _ h.metadata launch) data numeric (checked_storage c h.metadata launch) complete
    refine ⟨⟨result.returned,c.output.tensor,physical c m result.memory⟩,?_,ret,rfl,?_,?_⟩
    · simp [invoke,launch,run]
    · intro _ b hb head hh n hn
      simp only [outputIndex,physical_read]
      exact values b hb head hh n hn
    · exact physical_frame c m result.memory frame
  · refine ⟨⟨none,c.output.tensor,m⟩,by simp [invoke,launch],rfl,rfl,?_,fun _ _ _ => rfl⟩
    intro content b hb head hh n hn
    have data := checked_data c raw m h content metadata payload
    exact False.elim (launch (coordinate_active c m data b head n hb hh hn))

#axiomsClean tensor_correct
#specNonCircular realTarget avoiding [invoke,Strided.Grid.run,Strided.outputValues]
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Correct
