import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceGridTile
import bench.tritonbench_g.sgmv_expand_slice.Proofs.Schedule

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceGrid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1400000
set_option maxRecDepth 20000

def pidState {α : Type} (s : State α) (pid count : Nat) : State α :=
  {s with
    pids := fun ax => if ax = 0 then pid else 0
    numPids := fun ax => if ax = 0 then count else 1}

theorem run_pids {α β : Type} [Inhabited α] (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment)
    (X : Nat → Nat → α) (W : Nat → Nat → Nat → α) (encode : β → α)
    (update : Nat → Nat → β → β) (values : ValueLaw M dot abi p shape segment X W encode update)
    (separation : PrivateRegions p) (fits : SegmentFits shape segment) (live : segment.active)
    (slice : p.slice = shape.slice) (positive : 0 < shape.columns)
    (capacity : shape.slice+shape.columns ≤ shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride)
    (pids : List Nat) (bound : ∀ pid ∈ pids, pid < tiles segment.length*tiles shape.columns)
    (s : State α) (previous : Nat → Nat → β)
    (memory : SourceMemory.Inputs abi p shape s X W)
    (output : SourceSuffix.OutputMemory abi p shape s (fun r c => encode (previous r c)))
    (metadata : Metadata p segment s) :
    ∃ final, runSourceJobs (SourceBody.install M dot) abi p
        (pids.map (fun pid => ⟨p.batch,pid,tiles segment.length*tiles shape.columns⟩)) s = some final ∧
      SourceSuffix.OutputMemory abi p shape final
        (fun r c => encode ((pids.foldl (fun y pid => GridFold.step shape segment update pid y) previous) r c)) ∧
      ReadFrame p s final := by
  induction pids generalizing s previous with
  | nil => exact ⟨s,rfl,output,fun _ _ _ => rfl⟩
  | cons pid tail ih =>
      let one := pidState s pid (tiles segment.length*tiles shape.columns)
      have mo : SourceMemory.Inputs abi p shape one X W :=
        ⟨memory.inner,memory.columns,memory.x,memory.weight⟩
      have hmeta : Metadata p segment one := ⟨metadata.start,metadata.length,metadata.adapter⟩
      obtain ⟨next,he,ho,hf⟩ := tile M dot abi p shape segment X W encode update values one previous
        mo output hmeta fits live slice positive capacity layout (bound pid (by simp))
      have hn := inputs_frame abi p shape one next X W mo separation hf
      have hm := metadata_frame p segment one next hmeta separation hf
      obtain ⟨final,hr,hy,frame⟩ := ih (by intro k hk; exact bound k (List.mem_cons_of_mem _ hk))
        next (GridFold.step shape segment update pid previous) hn ho hm
      refine ⟨final,?_,hy,?_⟩
      · change (exec (SourceBody.install M dot) (kernel abi p) one).bind
          (runSourceJobs (SourceBody.install M dot) abi p
            (tail.map (fun pid => ⟨p.batch,pid,tiles segment.length*tiles shape.columns⟩))) = some final
        rw [he,Option.bind_some]
        exact hr
      · intro region hregion address
        exact (frame region hregion address).trans (hf region hregion address)

theorem batch {α β : Type} [Inhabited α] (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment)
    (X : Nat → Nat → α) (W : Nat → Nat → Nat → α) (encode : β → α)
    (update : Nat → Nat → β → β) (values : ValueLaw M dot abi p shape segment X W encode update)
    (separation : PrivateRegions p) (fits : SegmentFits shape segment) (live : segment.active)
    (slice : p.slice = shape.slice) (positive : 0 < shape.columns)
    (capacity : shape.slice+shape.columns ≤ shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride)
    (s : State α) (previous : Nat → Nat → β)
    (memory : SourceMemory.Inputs abi p shape s X W)
    (output : SourceSuffix.OutputMemory abi p shape s (fun r c => encode (previous r c)))
    (metadata : Metadata p segment s) :
    ∃ final, runSourceJobs (SourceBody.install M dot) abi p
        (batchJobs shape.columns p.batch segment) s = some final ∧
      SourceSuffix.OutputMemory abi p shape final
        (fun r c => encode (if ActiveCell shape segment r c then update r c (previous r c) else previous r c)) ∧
      ReadFrame p s final := by
  obtain ⟨final,he,ho,hf⟩ := run_pids M dot abi p shape segment X W encode update values separation fits live
    slice positive capacity layout (List.range (tiles segment.length*tiles shape.columns))
    (by intro pid hp; exact List.mem_range.mp hp) s previous memory output metadata
  rw [GridFold.full_grid] at ho
  exact ⟨final,by simpa only [batchJobs,if_pos (And.intro live positive)] using he,ho,hf⟩

#axiomsClean run_pids
#axiomsClean batch
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceGrid
