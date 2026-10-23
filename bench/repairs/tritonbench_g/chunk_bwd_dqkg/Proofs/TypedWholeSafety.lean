import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedSourceSafety
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedBodySafety
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedPostSafety

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedWholeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole Typed
open FP.Structural.Safety AddressBounds
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 1000000
variable {α : Type} [Inhabited α]

theorem loop_safe (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    stepSafe (engine (withDot M dot)) bounds
      (.forRangeDyn "i_v" (parts c).start (parts c).stop (parts c).stride (parts c).body)
      (entry M dot c (Typed.prepare c s a)).val := by
  rw [stepSafe_unfold]
  refine ⟨?_,?_,?_,?_⟩
  · simp [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,
      Option.map,Option.getD,safeOp_unfold]
  · simp [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,
      Option.map,Option.getD,safeOp_unfold]
  · simp [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,
      Option.map,Option.getD,safeOp_unfold]
  · change rangeSafe (engine (withDot M dot)) bounds "i_v" 0 (valueBlocks c) 1 (parts c).body
      (entry M dot c (Typed.prepare c s a)).val
    simpa only [Nat.zero_add] using
      TypedBodySafety.range_safe M dot c s a (valueBlocks c) 0
        ⟨(entry M dot c (Typed.prepare c s a)).val,entry_ready M dot c (Typed.prepare c s a)⟩
        (entry_context M dot c (Typed.prepare c s a)) bounds cap

/-- Safety is checked on the same entry, loop successors, and final stores as
`source_runs`. No bound/success premise about the program is assumed. -/
theorem source_safe (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    TraceSafe (engine (withDot M dot)) bounds (kernel c) (Typed.prepare c s a) := by
  change traceSafeList _ _ (kernel c).surfaceBody _
  rw [source_parts,List.append_assoc,traceSafeList_append]
  refine ⟨TypedSourceSafety.entry_safe M dot c s a bounds cap,?_⟩
  intro t ht
  have eq:=Option.some.inj (ht.symm.trans (entry M dot c (Typed.prepare c s a)).property)
  subst t
  change traceSafeList _ _ (_::(parts c).after) _
  exact traceSafeList_cons_of_step (loop_safe M dot c s a bounds cap)
    (loop_runs M dot c (Typed.prepare c s a))
    (TypedPostSafety.after_safe M dot c s _ a _
      (loop_context M dot c (Typed.prepare c s a)) bounds cap)

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedWholeSafety
