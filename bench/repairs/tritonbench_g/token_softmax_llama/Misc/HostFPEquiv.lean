import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Contents

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.FPEquiv
open VeriTile.Triton FP.Structural Strided.Grid Checked
noncomputable section

def nativeTarget (I : Strided.Native.Instructions) (V : Nat) (c : TensorCall)
    (m : Memory Strided.Native.Word) (b h n : Nat) :=
  Strided.Native.target I V (Strided.Host.params (host c.observation)) (view c m) (b,h) n

theorem tensor_fp32 (I : Strided.Native.Instructions) (V : Nat)
    (c : TensorCall) (m : Memory Strided.Native.Word)
    (raw : RegionName → Nat → BitVec 32) (metadata : MetadataPin c raw m)
    (h : InputContract c.observation (observedStart c raw) (observedLength c raw))
    (payload : PayloadCells c m)
    (work : List PID)
    (complete : (fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work) :
    ∃ result, invokeChecked (SoftmaxTriton1.NativeGraph.algebra I V) c work raw (Strided.Native.entry m)=some result ∧
      result.returned=none ∧ result.output=c.output.tensor ∧
      (contentActive c.observation → ∀ b  <  rows c.start.observation, ∀ h  <  rows c.input.observation,
        ∀ n  <  Strided.Host.seq (host c.observation) (view c m) b,
        ((result.memory (storageRegion c.output.tensor.storage) (outputIndex c m b h n)).read .fp32)
          (SoftmaxTriton1.NativeGraph.owner V (Strided.Host.width (host c.observation)) n)=
          nativeTarget I V c m b h n) ∧
      (∀ r i,exterior c (Strided.Native.entry m) r i → result.memory r i=Strided.Native.entry m r i) := by
  simp only [invoke_checked _ c work raw _ h]
  by_cases launch : active c.observation
  · have data := checked_data c raw m h ⟨launch.1,launch.2.1,launch.2.2.1⟩ metadata payload
    obtain ⟨result,run,ret,out,values,frame,safe⟩ := Strided.Host.host_native I V (host c.observation)
      work (view c m) (checked_layout _ h.metadata launch) data (checked_storage c h.metadata launch) complete
    refine ⟨⟨result.returned,c.output.tensor,physical c (Strided.Native.entry m) result.memory⟩,?_,ret,rfl,?_,?_⟩
    · simp [invoke,launch,view_entry,run]
    · intro _ b hb head hh n hn
      simp only [outputIndex,physical_read]
      exact values b hb head hh n hn
    · apply physical_frame c (Strided.Native.entry m) result.memory
      simpa only [view_entry] using frame
  · refine ⟨⟨none,c.output.tensor,Strided.Native.entry m⟩,by simp [invoke,launch],rfl,rfl,?_,fun _ _ _ => rfl⟩
    intro content b hb head hh n hn
    have data := checked_data c raw m h content metadata payload
    exact False.elim (launch (coordinate_active c m data b head n hb hh hn))

/-- Only the six scalar raw-word instruction families cross the numerical
boundary. No whole-row, reduction, source-result or final-memory oracle occurs. -/
theorem local_refinement (actual reference : Strided.Native.Instructions)
    (ports : SoftmaxTriton1.NativeGraph.InstructionAgreement actual reference)
    (V : Nat) (c : TensorCall) (m : Memory Strided.Native.Word) (b h n : Nat) :
    nativeTarget actual V c m b h n=nativeTarget reference V c m b h n :=
  SoftmaxTriton1.NativeGraph.row_agreement actual reference ports V _ _ _ _

#axiomsClean tensor_fp32
#axiomsClean local_refinement
#specNonCircular nativeTarget avoiding [invoke,Strided.Grid.run,Strided.outputValues]
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.FPEquiv
