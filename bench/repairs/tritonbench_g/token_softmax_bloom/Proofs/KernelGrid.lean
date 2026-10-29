import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelAdmission
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelCore

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelGrid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open KernelTensor
noncomputable section
set_option maxHeartbeats 4000000

@[simp] theorem offset_view {α} [Inhabited α] (c : TensorCall) (read : Nat → α)
    (metadataRead : Nat → Word) (before : NativeCells α) (pid : Grid.PID)
    (i : TileIndex [(params c).width]) :
    Grid.offset (params c) (view c read metadataRead before) pid i=
      pid.2*c.output.rowStride+(start c metadataRead pid.1+i.1.val)*c.output.colStride := by
  simp [Grid.offset,outputOffset,params,startIndex,Grid.state,view,start,observedStart]

def outputAddress (c : TensorCall) (metadataRead : Nat → Word) (b head n : Nat) : Nat :=
  c.output.tensor.dataPtr.toNat+4*(head*c.output.rowStride+(start c metadataRead b+n)*c.output.colStride)

def Exterior (c : TensorCall) (metadataRead : Nat → Word) (address : Nat) : Prop :=
  ∀ b < rows c.start.observation, ∀ head < rows c.input.observation,
    ∀ n < length c metadataRead b, address≠outputAddress c metadataRead b head n

/-- The raw typed execution changes only its actual active output cells. -/
def RawFrame {α} [Inhabited α] (c : TensorCall) (read : Nat → α) (metadataRead : Nat → Word)
    (before : NativeCells α) (work : List Grid.PID) (after : Grid.Memory α) : Prop :=
  ∀ (r : RegionName) (i : Nat),(r≠"output" ∨ ∀ e∈Grid.events (params c) (view c read metadataRead before) work,
    i≠Grid.offset (params c) (view c read metadataRead before) e.1 e.2) →
      after r i=view c read metadataRead before r i

theorem live {α} [Inhabited α] (c : TensorCall) (read : Nat → α) (metadataRead : Nat → Word)
    (before : NativeCells α) (work : List Grid.PID)
    (complete : (Grid.fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work)
    (b head : Nat) (i : TileIndex [(params c).width])
    (hb : b < rows c.start.observation) (hh : head < rows c.input.observation)
    (hn : i.1.val < length c metadataRead b) :
    ((b,head),i)∈Grid.events (params c) (view c read metadataRead before) work := by
  apply (Grid.mem_events _ _ _ _).mpr
  refine ⟨complete.mem_iff.mp ((Grid.mem_fullGrid _ _ _).mpr ⟨hb,hh⟩),?_⟩
  simpa [Grid.active] using hn

/-- Full-grid successful source and candidate runs share the entire typed
memory. Every event's value and every exterior raw Cell are proved from exec. -/
theorem complete {α} [Inhabited α] (M : Algebra α) (c : TensorCall)
    (read : Nat → α) (metadataRead : Nat → Word) (before : NativeCells α)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (work : List Grid.PID)
    (order : (Grid.fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work) :
    ∃ after,
      Grid.run (FP.ProfiledRewrite.engine M) (params c) work (view c read metadataRead before)=some after ∧
      KernelRewrite.candidateRun (FP.ProfiledRewrite.engine M) (params c) work
        (view c read metadataRead before)=some after ∧
      (∀ e∈Grid.events (params c) (view c read metadataRead before) work,
        after "output" (Grid.offset (params c) (view c read metadataRead before) e.1 e.2)=
          .mk .fp32 (Grid.values (FP.ProfiledRewrite.engine M) (params c) (view c read metadataRead before) e)) ∧
      (∀ (r : RegionName) (i : Nat),(r≠"output" ∨ ∀ e∈Grid.events (params c) (view c read metadataRead before) work,
        i≠Grid.offset (params c) (view c read metadataRead before) e.1 e.2) →
          after r i=view c read metadataRead before r i) := by
  have positive : 0 < (params c).width :=
    lt_of_lt_of_le (show 0 < c.maximum from launch.2.2.2)
      (VeriTile.Bench.Repairs.SquareMatrix.block_covers c.maximum)
  exact KernelCore.complete M (params c) positive work (view c read metadataRead before)
    (KernelAdmission.separate c)
    (KernelAdmission.unique c read metadataRead before contract launch work order)

theorem project_frame {α} [Inhabited α] (c : TensorCall) (read : Nat → α) (metadataRead : Nat → Word)
    (before : NativeCells α) (after : Grid.Memory α) (work : List Grid.PID)
    (order : (Grid.fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work)
    (frame : ∀ (r : RegionName) (i : Nat),(r≠"output" ∨ ∀ e∈Grid.events (params c) (view c read metadataRead before) work,
      i≠Grid.offset (params c) (view c read metadataRead before) e.1 e.2) → after r i=view c read metadataRead before r i)
    (address : Nat) (outside : Exterior c metadataRead address) :
    project c before after address=before address := by
  unfold project
  split
  · rename_i aligned
    rw [frame "output" _ (Or.inr (by
      intro e he equal
      obtain ⟨event,active⟩ := (Grid.mem_events _ _ _ _).mp he
      have legal := (Grid.mem_fullGrid _ _ _).mp (order.mem_iff.mpr event)
      have count : e.2.1.val < length c metadataRead e.1.1 := by simpa [Grid.active] using active
      have absent := outside e.1.1 legal.1 e.1.2 legal.2 e.2.1.val count
      rw [offset_view] at equal
      apply absent
      unfold outputAddress
      omega))]
    simp only [view,show ("output" : RegionName)≠"input" by decide,if_false,
      show ("output" : RegionName)≠"start" by decide,
      show ("output" : RegionName)≠"length" by decide,if_true]
    congr 1
    omega
  · rfl

#axiomsClean complete
#axiomsClean project_frame
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelGrid
