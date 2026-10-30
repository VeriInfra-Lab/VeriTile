import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.AttnFwdCausal.RawNative
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- Complete native shared-source execution at raw typed memory, independent
fixed-instruction output words, and every exterior memory cell. -/
theorem kernel_native (instructions : Native.Instructions) (p : Parameters)
    (apart : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (initial : State Native.Word) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms (fun _=>Native.algebra) (fun _=>Native.dotModel) p (grid p) work
        (Native.injectState initial)=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
        (Native.projectState instructions p initial final).mem p.Out (logicalAddress p b h n d)=
          .mk .fp16 (NativeReference.target instructions p initial b h n d hd)) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off →
        (Native.projectState instructions p initial final).mem r off=initial.mem r off) := by
  obtain ⟨final,run,values,frame⟩ := Kernel.grid_result (fun _=>Native.algebra) (fun _=>Native.dotModel)
    p (RawKernel.storage p apart) (Native.injectState initial) work legal
  refine ⟨final,run,?_,?_⟩
  · intro b hb h hh n hn d hd
    rw [←NativeReference.target_binding instructions p initial b h n d hh hn hd]
    change Native.projectCell instructions (Native.ownerAt p p.Out (logicalAddress p b h n d))
      (final.mem p.Out (logicalAddress p b h n d))=_
    rw [values b hb h hh n hn d hd]
    simp only [Native.ownerAt,if_pos rfl,
      Native.featureAt_output p (RawKernel.storage p apart) b h n d hb hh hn hd]
    rfl
  · intro r off outside
    change Native.projectCell instructions (Native.ownerAt p r off) (final.mem r off)=_
    rw [frame r off outside]
    exact Native.injectState_project instructions _ initial r off

#axiomsClean kernel_native
#specNonCircular NativeReference.target avoiding [Whole.answer,Source.accumulated,Source.update,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttnFwdCausal.RawNative
