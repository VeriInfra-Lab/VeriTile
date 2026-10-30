import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceGridBatch

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceGrid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 20000

theorem runner_batch {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (batch : Nat) (js : List Job) (s : State α) :
    runSourceJobs M abi {p with batch := batch} js s = runSourceJobs M abi p js s := by
  induction js generalizing s with
  | nil => rfl
  | cons j tail ih =>
      simp only [runSourceJobs]
      have hf : runSourceJobs M abi {p with batch := batch} tail = runSourceJobs M abi p tail := by
        funext t; exact ih t
      rw [hf]

def MetadataList {α : Type} [Inhabited α] (p : KernelArgs) : Nat → List Segment → State α → Prop
  | _,[],_ => True
  | batch,segment::tail,s => (segment.active → Metadata {p with batch := batch} segment s) ∧
      MetadataList p (batch+1) tail s

theorem separation_batch (p : KernelArgs) (batch : Nat) (h : PrivateRegions p) :
    PrivateRegions {p with batch := batch} :=
  ⟨h.input,h.weight,h.starts,h.lengths,h.adapters⟩

theorem input_batch {α : Type} [Inhabited α] (abi : SourceABI) (p : KernelArgs) (batch : Nat)
    (shape : Shape) (s : State α) (X : Nat → Nat → α) (W : Nat → Nat → Nat → α)
    (h : SourceMemory.Inputs abi p shape s X W) : SourceMemory.Inputs abi {p with batch := batch} shape s X W :=
  ⟨h.inner,h.columns,h.x,h.weight⟩

theorem output_batch {α : Type} [Inhabited α] (abi : SourceABI) (p : KernelArgs) (batch : Nat)
    (shape : Shape) (s : State α) (Y : Nat → Nat → α)
    (h : SourceSuffix.OutputMemory abi p shape s Y) :
    SourceSuffix.OutputMemory abi {p with batch := batch} shape s Y := h

theorem metadataList_frame {α : Type} [Inhabited α] (p : KernelArgs) (batch : Nat)
    (segments : List Segment) (s t : State α) (separation : PrivateRegions p)
    (metadata : MetadataList p batch segments s) (frame : ReadFrame p s t) :
    MetadataList p batch segments t := by
  induction segments generalizing batch with
  | nil => trivial
  | cons segment tail ih =>
      refine ⟨?_,ih (batch+1) metadata.2⟩
      intro live
      exact metadata_frame {p with batch := batch} segment s t (metadata.1 live) (separation_batch p batch separation) frame

def formulaStep {β : Type} (shape : Shape) (update : Segment → Nat → Nat → β → β)
    (segment : Segment) (previous : Nat → Nat → β) : Nat → Nat → β :=
  fun r c => if ActiveCell shape segment r c then update segment r c (previous r c) else previous r c

theorem inactive_formula {β : Type} (shape : Shape) (update : Segment → Nat → Nat → β → β)
    (segment : Segment) (previous : Nat → Nat → β) (inactive : ¬(segment.active ∧ 0 < shape.columns)) :
    formulaStep shape update segment previous = previous := by
  funext r c
  have h : ¬ ActiveCell shape segment r c := by
    intro active
    have hc := active.2.2.2.2
    have hs := active.2.2.2.1
    exact inactive ⟨active.1,by omega⟩
  simp only [formulaStep,if_neg h]

#axiomsClean runner_batch
#axiomsClean metadataList_frame
#axiomsClean inactive_formula
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceGrid
