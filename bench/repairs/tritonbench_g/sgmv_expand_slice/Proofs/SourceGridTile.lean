import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceTile
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceMemory
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.OutputValues
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.GridCoordinates

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceGrid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1400000
set_option maxRecDepth 20000

def ReadFrame {α : Type} (p : KernelArgs) (s t : State α) : Prop :=
  ∀ region, region ≠ p.output → ∀ address, t.mem region address = s.mem region address

structure Metadata {α : Type} [Inhabited α] (p : KernelArgs) (segment : Segment) (s : State α) : Prop where
  start : (s.mem p.starts p.batch).read .int = (segment.start : Int)
  length : (s.mem p.lengths p.batch).read .int = (segment.length : Int)
  adapter : (s.mem p.adapters p.batch).read .int = (segment.adapter.getD 0 : Int)

/-- The role separation concerns six private clone regions; no caller
allocation separation is required. -/
structure PrivateRegions (p : KernelArgs) : Prop where
  input : p.input ≠ p.output
  weight : p.weight ≠ p.output
  starts : p.starts ≠ p.output
  lengths : p.lengths ≠ p.output
  adapters : p.adapters ≠ p.output

theorem inputs_frame {α : Type} [Inhabited α] (abi : SourceABI) (p : KernelArgs) (shape : Shape)
    (s t : State α) (X : Nat → Nat → α) (W : Nat → Nat → Nat → α)
    (memory : SourceMemory.Inputs abi p shape s X W) (separation : PrivateRegions p) (frame : ReadFrame p s t) :
    SourceMemory.Inputs abi p shape t X W := by
  refine ⟨memory.inner,memory.columns,?_,?_⟩
  · intro r k hr hk
    rw [frame p.input separation.input]
    exact memory.x r k hr hk
  · intro a c k ha hc hk
    rw [frame p.weight separation.weight]
    exact memory.weight a c k ha hc hk

theorem metadata_frame {α : Type} [Inhabited α] (p : KernelArgs) (segment : Segment)
    (s t : State α) (metadata : Metadata p segment s) (separation : PrivateRegions p)
    (frame : ReadFrame p s t) : Metadata p segment t := by
  constructor
  · rw [frame p.starts separation.starts]; exact metadata.start
  · rw [frame p.lengths separation.lengths]; exact metadata.length
  · rw [frame p.adapters separation.adapters]; exact metadata.adapter

def ValueLaw {α β : Type} [Inhabited α] (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment)
    (X : Nat → Nat → α) (W : Nat → Nat → Nat → α)
    (encode : β → α) (update : Nat → Nat → β → β) : Prop :=
  ∀ (s : State α) (previous : Nat → Nat → β), SourceMemory.Inputs abi p shape s X W →
    SourceSuffix.OutputMemory abi p shape s (fun r c => encode (previous r c)) →
    ∀ i, SourceSuffix.active p (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i →
      SourceSuffix.outputValues M abi p
        (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) s
        (SourceExecution.accumulated M dot abi p s segment.start (segment.adapter.getD 0) segment.length) i =
      encode (update
        (SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i)
        (SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i)
        (previous
          (SourceSuffix.row (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i)
          (SourceSuffix.column p (SourceExecution.outputCoordinates p segment.start segment.length (s.pids 0)) i)))

theorem tile {α β : Type} [Inhabited α] (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment)
    (X : Nat → Nat → α) (W : Nat → Nat → Nat → α) (encode : β → α)
    (update : Nat → Nat → β → β) (values : ValueLaw M dot abi p shape segment X W encode update)
    (s : State α) (previous : Nat → Nat → β)
    (memory : SourceMemory.Inputs abi p shape s X W)
    (output : SourceSuffix.OutputMemory abi p shape s (fun r c => encode (previous r c)))
    (metadata : Metadata p segment s) (fits : SegmentFits shape segment) (live : segment.active)
    (slice : p.slice = shape.slice) (positive : 0 < shape.columns)
    (capacity : shape.slice+shape.columns ≤ shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride)
    (hp : s.pids 0 < tiles segment.length*tiles shape.columns) :
    ∃ final, exec (SourceBody.install M dot) (kernel abi p) s = some final ∧
      SourceSuffix.OutputMemory abi p shape final
        (fun r c => encode (GridFold.step shape segment update (s.pids 0) previous r c)) ∧
      ReadFrame p s final := by
  obtain ⟨final,he,read,frame⟩ := SourceTile.execution M dot abi p shape segment s
    metadata.start metadata.length metadata.adapter fits live memory.columns slice positive capacity layout hp
  refine ⟨final,he,?_,?_⟩
  · intro r c hr hc
    by_cases covered : ActiveCell shape segment r c ∧ GridFold.owner shape segment r c = s.pids 0
    · obtain ⟨i,hi,hri,hci⟩ := (GridCoordinates.covered_iff p shape segment (s.pids 0) r c
        memory.columns slice live).mpr covered
      have hread := read i hi
      rw [SourceSuffix.active_offset p _ i hi,hri,hci] at hread
      change SourceSuffix.readOutput abi p final r c = _ at hread
      rw [hread,values s previous memory output i hi,hri,hci]
      simp only [GridFold.step,if_pos covered]
    · have hframe := frame p.output (r*p.outputRowStride+c*p.outputColumnStride)
        (Or.inr (GridCoordinates.miss_offset p shape segment (s.pids 0) r c memory.columns slice live
          fits capacity layout hr hc covered))
      simp only [SourceSuffix.readOutput,hframe]
      exact (output r c hr hc).trans (by simp only [GridFold.step,if_neg covered])
  · intro region hregion address
    exact frame region address (Or.inl hregion)

#axiomsClean inputs_frame
#axiomsClean metadata_frame
#axiomsClean tile
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceGrid
