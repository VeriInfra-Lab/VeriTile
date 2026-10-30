import bench.tritonbench_g.chunk_retention.Proofs.IOLayout

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelPacket
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 1000000

/-- Every active original observation is a typed stored cell. This augments the
old read-value result by reusing its exact source-memory and store-readback facts. -/
theorem output_tag (M : Algebra α) (D : DotPrimitive α) (a : Call) (s t : State α)
    (ran : FP.Structural.exec (engine (withDot M D)) a.kernel s=some t)
    (o : Observation a.shape) (active : Active a s o) :
    ∃ value,t.mem o.region (o.offset s)=Cell.mk .real value := by
  cases stage : a.stage with
  | forwardH =>
    obtain ⟨u,hu,hm⟩ := ForwardHSource.source_memory M D a.shape a.seed a.final s
    have he : u=t := Option.some.inj (hu.symm.trans (by simpa only [Call.kernel,stage] using ran))
    subst u
    rw [hm]
    cases o <;> simp only [Active,stage,reduceCtorEq,false_and,true_and] at active
    · exact ⟨_,ForwardHSource.history_readback M D a.shape a.seed a.final s _ _ active.1 active.2⟩
    · rcases active with ⟨hf,hi⟩
      simp only [Observation.region,Observation.offset,hf]
      exact ⟨_,ForwardHSource.final_readback M D a.shape a.seed s _ hi⟩
    all_goals contradiction
  | forwardO =>
    obtain ⟨u,hu,hm⟩ := ForwardOSource.source_memory M D a.shape a.scale s
    have he : u=t := Option.some.inj (hu.symm.trans (by simpa only [Call.kernel,stage] using ran))
    subst u
    rw [hm]
    cases o <;> simp only [Active,stage,reduceCtorEq,false_and,true_and] at active
    all_goals first | contradiction | exact ⟨_,ForwardOSource.readback M D a.shape a.scale s _ active (ForwardOSource.address_injective a.shape s)⟩
  | backwardH =>
    obtain ⟨u,hu,hm⟩ := BackwardHSource.source_memory M D a.shape a.scale a.seed s
    have he : u=t := Option.some.inj (hu.symm.trans (by simpa only [Call.kernel,stage] using ran))
    subst u
    rw [hm]
    cases o <;> simp only [Active,stage,reduceCtorEq,false_and,true_and] at active
    all_goals first | contradiction | exact ⟨_,BackwardHSource.history_readback M D a.shape s a.scale a.seed _ _ active.1 active.2⟩
  | backwardQKV =>
    obtain ⟨u,hu,hm⟩ := BackwardQKVSource.source_memory M D a.shape a.scale s
    have he : u=t := Option.some.inj (hu.symm.trans (by simpa only [Call.kernel,stage] using ran))
    subst u
    rw [hm]
    cases o <;> simp only [Active,stage,reduceCtorEq,false_and,true_and] at active
    · exact ⟨_,BackwardQKVSource.query_readback M D a.shape s a.scale _ active⟩
    · exact ⟨_,BackwardQKVSource.key_readback M D a.shape s a.scale _ active⟩
    · exact ⟨_,BackwardQKVSource.partial_readback M D a.shape s a.scale _ _ active.1 active.2⟩
    all_goals contradiction

#axiomsClean output_tag
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
