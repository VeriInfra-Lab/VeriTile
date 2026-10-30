import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPFinish

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPOutput
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open FPDefault FPReference
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def ActiveInjective (initial : State α) (BM BN : Nat) (p : RealProgram.Parameters) : Prop :=
  ∀ a b, active initial BM BN p a → active initial BM BN p b →
    offset initial BM BN p p.oBatch p.oRow p.oHead p.oColumn a =
      offset initial BM BN p p.oBatch p.oRow p.oHead p.oColumn b → a = b

theorem readback (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) (i : TileIndex [BM,BN])
    (hi : active initial BM BN p i) (hinj : ActiveInjective initial BM BN p) :
    (FPFinish.write M D BM BN BCS p initial).mem p.Db
      (offset initial BM BN p p.oBatch p.oRow p.oHead p.oColumn i) =
        .mk .fp32 (expected M D BM BN BCS p initial i) := by
  obtain ⟨before,after,hsplit⟩ := List.append_of_mem (TileShape.mem_allIndices [BM,BN] i)
  have hn := TileShape.allIndices_nodup [BM,BN]
  rw [hsplit,List.nodup_append,List.nodup_cons] at hn
  unfold FPFinish.write
  rw [hsplit,List.foldl_append,List.foldl_cons,if_pos hi]
  rw [State.masked_scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro k hk ha he
    have hki := hinj i k hi ha he
    subst k
    exact hn.2.1.1 hk

theorem frame (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) (r : RegionName) (a : Nat)
    (ha : r ≠ p.Db ∨ ∀ i, active initial BM BN p i →
      a ≠ offset initial BM BN p p.oBatch p.oRow p.oHead p.oColumn i) :
    (FPFinish.write M D BM BN BCS p initial).mem r a = initial.mem r a := by
  apply State.masked_scatter_frame
  exact ha.imp_right (fun h i _ hi => h i hi)

/-- Complete actual source with its independent ordered FP32 recurrence.
No successful-run, carry, store, or final-output assumption appears. -/
theorem source_memory (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) :
    ∃ final, FP.Structural.exec (engine (backend M D)) (source BM BN BCS p) initial = some final ∧
      final.mem = (FPFinish.write M D BM BN BCS p initial).mem := by
  let stop := (length initial p+BCS-1)/BCS
  obtain ⟨mid,hmid,hinv⟩ := FPCarry.loop M D BM BN BCS p initial stop
  have loopRun : step (engine (backend M D))
      (.forRangeDyn "cs" (parts BM BN BCS p).start (parts BM BN BCS p).stop
        (parts BM BN BCS p).stride (parts BM BN BCS p).body)
      (setup (backend M D) BM BN BCS p initial).val = some mid := by
    have hb : evalOp (engine (backend M D)) none (parts BM BN BCS p).stop
        (setup (backend M D) BM BN BCS p initial).val = some (fun _ => stop) := by
      by_cases hc : p.chunk < p.seqlen-initial.pids 2/p.groups*p.chunk
      · simp [parts,source,RealProgram.source,TypedSource.specialize,ComputeKernel.surfaceBody,
          Source.dynamicRange?,evalOp_unfold,setup,numeric,integralFloorDiv,bop,stop,length,chunk,
          hc,Nat.min_eq_left (Nat.le_of_lt hc)]
        rfl
      · simp [parts,source,RealProgram.source,TypedSource.specialize,ComputeKernel.surfaceBody,
          Source.dynamicRange?,evalOp_unfold,setup,numeric,integralFloorDiv,bop,stop,length,chunk,
          hc,Nat.min_eq_right (Nat.le_of_not_gt hc)]
        rfl
    simp only [step]
    rw [hb]
    change range (engine (backend M D)) "cs" 0 stop 1 (parts BM BN BCS p).body
      (setup (backend M D) BM BN BCS p initial).val = some mid
    exact hmid
  obtain ⟨final,hfinal,hmem⟩ := FPFinish.finish M D BM BN BCS p initial mid hinv
  refine ⟨final,?_,hmem⟩
  change run (engine (backend M D)) (source BM BN BCS p).surfaceBody initial = _
  rw [source_split,List.append_assoc,run_append,(setup (backend M D) BM BN BCS p initial).property]
  simp only [Option.bind_some,List.singleton_append,run,loopRun,bind,Option.bind_some]
  exact hfinal

theorem source_output (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) (hinj : ActiveInjective initial BM BN p) :
    OutputGoal M D BM BN BCS p initial := by
  obtain ⟨final,he,hm⟩ := source_memory M D BM BN BCS p initial
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact readback M D BM BN BCS p initial i hi hinj
  · intro r a ha
    rw [hm]
    exact frame M D BM BN BCS p initial r a ha

#axiomsClean source_memory
#axiomsClean source_output
#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPOutput
