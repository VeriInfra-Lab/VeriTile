import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceGridSchedule

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceGrid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 20000

/-- Actual ordered jobs, including overlapping batches and empty/sentinel/
zero-column skips. The tile value law is a proved adapter at each concrete
backend, never an additional public execution or numerical assumption. -/
theorem run_segments {α β : Type} [Inhabited α] (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (X : Nat → Nat → α) (W : Nat → Nat → Nat → α) (encode : β → α)
    (update : Segment → Nat → Nat → β → β)
    (values : ∀ batch segment, segment ∈ segments → segment.active → 0 < shape.columns →
      ValueLaw M dot abi {p with batch := batch} shape segment X W encode (update segment))
    (separation : PrivateRegions p) (fits : ∀ segment ∈ segments, SegmentFits shape segment)
    (slice : p.slice = shape.slice) (capacity : shape.slice+shape.columns ≤ shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride)
    (first : Nat) (s : State α) (previous : Nat → Nat → β)
    (memory : SourceMemory.Inputs abi p shape s X W)
    (output : SourceSuffix.OutputMemory abi p shape s (fun r c => encode (previous r c)))
    (metadata : MetadataList p first segments s) :
    ∃ final, runSourceJobs (SourceBody.install M dot) abi p (jobsFrom shape.columns first segments) s = some final ∧
      SourceSuffix.OutputMemory abi p shape final
        (fun r c => encode ((segments.foldl (fun y segment => formulaStep shape update segment y) previous) r c)) ∧
      ReadFrame p s final := by
  cases segments with
  | nil => exact ⟨s,rfl,output,fun _ _ _ => rfl⟩
  | cons segment tail =>
      have tfits : ∀ seg ∈ tail, SegmentFits shape seg := fun seg hs => fits seg (List.mem_cons_of_mem _ hs)
      have tvalues : ∀ batch seg, seg ∈ tail → seg.active → 0 < shape.columns →
          ValueLaw M dot abi {p with batch := batch} shape seg X W encode (update seg) :=
        fun batch seg hs => values batch seg (List.mem_cons_of_mem _ hs)
      by_cases active : segment.active ∧ 0 < shape.columns
      · obtain ⟨next,he,ho,hf⟩ := batch M dot abi {p with batch := first} shape segment X W encode (update segment)
          (values first segment (by simp) active.1 active.2) (separation_batch p first separation)
          (fits segment (by simp)) active.1 slice active.2 capacity layout s previous
          (input_batch abi p first shape s X W memory) output (metadata.1 active.1)
        have hn := inputs_frame abi p shape s next X W memory separation hf
        have hm := metadataList_frame p (first+1) tail s next separation metadata.2 hf
        obtain ⟨final,hr,hy,frame⟩ := run_segments M dot abi p shape tail X W encode update tvalues separation tfits
          slice capacity layout (first+1) next (formulaStep shape update segment previous) hn ho hm
        refine ⟨final,?_,hy,?_⟩
        · rw [jobsFrom,source_jobs_append]
          rw [runner_batch] at he
          rw [he,Option.bind_some]
          exact hr
        · intro region hregion address
          exact (frame region hregion address).trans (hf region hregion address)
      · obtain ⟨final,hr,hy,frame⟩ := run_segments M dot abi p shape tail X W encode update tvalues separation tfits
          slice capacity layout (first+1) s previous memory output metadata.2
        refine ⟨final,?_,?_,frame⟩
        · simpa only [jobsFrom,batchJobs,if_neg active,List.nil_append] using hr
        · simpa only [List.foldl_cons,inactive_formula shape update segment previous active] using hy
termination_by segments.length
decreasing_by
  all_goals
    subst segments
    simp

#axiomsClean run_segments
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceGrid
