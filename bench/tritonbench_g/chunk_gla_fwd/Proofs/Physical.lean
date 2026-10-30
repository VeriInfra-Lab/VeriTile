import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPHost
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPReadBounds
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Physical
open VeriTile Triton FP.Structural
open FPStorage (Memory Allocation project flush)
open Host (Dimensions)
noncomputable section
variable {α : Type} [Inhabited α]

def tracked (r : RegionName) : Prop :=
  r="q" ∨ r="k" ∨ r="g" ∨ r="v" ∨ r="h" ∨ r="attention" ∨ r="scratch" ∨ r="o"
def writable (r : RegionName) : Prop := r="attention" ∨ r="scratch" ∨ r="o"
def size (c : Dimensions) (r : RegionName) : Nat :=
  if r="q" ∨ r="k" ∨ r="g" then c.heads*c.T*c.K
  else if r="v" ∨ r="o" then Host.outputSize c
  else if r="h" then c.heads*c.NT*c.K*c.V
  else if r="attention" then Host.attentionSize c
  else if r="scratch" then Host.scratchSize c else 0

/-- A Tensor/runtime allocation contract only: finite readable FP32 slices.
Freshness is required separately for the allocation written by a call. In
particular a standalone consumer's A may alias any other read-only input. -/
structure ABI (c : Dimensions) where
  alloc : RegionName → Allocation
  capacity : Nat
  lengths : ∀ r, tracked r → (alloc r).length=size c r
  finite : ∀ r, tracked r → (alloc r).base+(alloc r).length ≤ capacity

def Fresh (abi : ABI c) (w : RegionName) : Prop :=
  ∀ r, tracked r → w≠r → (abi.alloc w).Separate (abi.alloc r)
def ProducerFresh (abi : ABI c) : Prop :=
  Fresh abi "attention" ∧ (256<c.K ∧ 0<c.heads*c.T → Fresh abi "scratch")
def ConsumerFresh (abi : ABI c) : Prop := Fresh abi "o"

def commit (abi : ABI c) (region : RegionName) (m : Memory α) (physical : Nat → α) : Nat → α :=
  flush (abi.alloc region) (m region) physical

theorem commit_readback (abi : ABI c) (region : RegionName) (m : Memory α) (physical : Nat → α)
    (i : Nat) (hi : i<(abi.alloc region).length) :
    commit abi region m physical ((abi.alloc region).base+i)=m region i :=
  FPStorage.flush_readback _ _ _ i hi

theorem commit_frame (abi : ABI c) (region : RegionName) (m : Memory α) (physical : Nat → α)
    (p : Nat) (hp : ¬(abi.alloc region).Contains p) : commit abi region m physical p=physical p :=
  FPStorage.flush_frame _ _ _ p hp

theorem commit_preserves (abi : ABI c) (w r : RegionName) (hw : Fresh abi w) (hr : tracked r) (hne : w≠r)
    (m : Memory α) (physical : Nat → α) (i : Nat) (hi : i<(abi.alloc r).length) :
    commit abi w m physical ((abi.alloc r).base+i)=physical ((abi.alloc r).base+i) :=
  FPStorage.flush_preserves_input _ _ (hw r hr hne) _ _ i hi

/-- Finite projection/commit simulation for a source stage. A write only
changes its own fresh allocation; aliased input projections stay coherent. -/
theorem commit_projection (abi : ABI c) (w : RegionName) (hw : Fresh abi w)
    (m : Memory α) (physical : Nat → α)
    (hframe : ∀ r, r≠w → ∀ i, m r i=project abi.alloc physical r i)
    (r : RegionName) (hr : tracked r) (i : Nat) (hi : i<(abi.alloc r).length) :
    project abi.alloc (commit abi w m physical) r i=m r i := by
  by_cases he : r=w
  · subst r
    exact commit_readback abi w m physical i hi
  · change commit abi w m physical ((abi.alloc r).base+i)=m r i
    rw [commit_preserves abi w r hw hr (Ne.symm he) m physical i hi]
    exact (hframe r he i).symm

def stageExecute (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c)
    (stage : FPStages.Stage) (physical : Nat → α) : Option (Nat → α) :=
  (FPStages.execute M D stage (project abi.alloc physical)).map
    (fun m => commit abi stage.region m physical)
def stageReference (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c)
    (stage : FPStages.Stage) (physical : Nat → α) : Nat → α :=
  commit abi stage.region (FPStages.reference M D stage (project abi.alloc physical)) physical
theorem stage_source (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c)
    (stage : FPStages.Stage) (physical : Nat → α) :
    stageExecute M D abi stage physical=some (stageReference M D abi stage physical) := by
  simp [stageExecute,stageReference,FPStages.execute_eq]
theorem stage_projection (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c)
    (stage : FPStages.Stage) (hw : Fresh abi stage.region) (physical : Nat → α)
    (r : RegionName) (hr : tracked r) (i : Nat) (hi : i<(abi.alloc r).length) :
    project abi.alloc (stageReference M D abi stage physical) r i=
      FPStages.reference M D stage (project abi.alloc physical) r i :=
  commit_projection abi stage.region hw _ physical
    (fun rr hrr ii => FPStages.reference_frame M D stage _ rr ii hrr) r hr i hi

def producerCommit (abi : ABI c) (m : Memory α) (physical : Nat → α) : Nat → α :=
  if c.heads*c.T=0 ∨ c.K ≤ 256 then commit abi "attention" m physical
  else commit abi "attention" m (commit abi "scratch" m physical)
def producerExecute (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) : Option (Nat → α) :=
  (Host.producerExecute M D c scale (project abi.alloc physical)).map (fun m => producerCommit abi m physical)
def producerReference (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) : Nat → α :=
  producerCommit abi (Host.producerReference M D c scale (project abi.alloc physical)) physical
theorem producer_source (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) : producerExecute M D abi scale physical=some (producerReference M D abi scale physical) := by
  simp [producerExecute,producerReference,Host.producer_source]
theorem producer_readback (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) (i : Nat) (hi : i<Host.attentionSize c) :
    producerReference M D abi scale physical ((abi.alloc "attention").base+i)=
      Host.producerReference M D c scale (project abi.alloc physical) "attention" i := by
  have hlen : (abi.alloc "attention").length=Host.attentionSize c := by
    simpa [size] using abi.lengths "attention" (by simp [tracked])
  unfold producerReference producerCommit
  split <;> exact commit_readback abi "attention" _ _ i (by simpa [hlen] using hi)
theorem producer_frame (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) (p : Nat)
    (ha : ¬(abi.alloc "attention").Contains p) (hs : ¬(abi.alloc "scratch").Contains p) :
    producerReference M D abi scale physical p=physical p := by
  unfold producerReference producerCommit
  split
  · exact commit_frame abi "attention" _ _ p ha
  · rw [commit_frame abi "attention" _ _ p ha,commit_frame abi "scratch" _ _ p hs]

def consumerExecute (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) : Option (Nat → α) :=
  (Host.consumerExecute M D c scale (project abi.alloc physical)).map (fun m => commit abi "o" m physical)
def consumerReference (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) : Nat → α :=
  commit abi "o" (Host.consumerReference M D c scale (project abi.alloc physical)) physical
theorem consumer_source (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) : consumerExecute M D abi scale physical=some (consumerReference M D abi scale physical) := by
  simp [consumerExecute,consumerReference,Host.consumer_source]
theorem consumer_readback (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) (i : Nat) (hi : i<Host.outputSize c) :
    consumerReference M D abi scale physical ((abi.alloc "o").base+i)=
      Host.consumerReference M D c scale (project abi.alloc physical) "o" i := by
  apply commit_readback
  have hlen : (abi.alloc "o").length=Host.outputSize c := by
    simpa [size] using abi.lengths "o" (by simp [tracked])
  simpa [hlen] using hi
theorem consumer_frame (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) (p : Nat) (hp : ¬(abi.alloc "o").Contains p) :
    consumerReference M D abi scale physical p=physical p := commit_frame abi "o" _ _ p hp

#specNonCircular producerReference avoiding [producerExecute,Host.producerExecute,FP.Structural.exec]
#specNonCircular consumerReference avoiding [consumerExecute,Host.consumerExecute,FP.Structural.exec]
#axiomsClean stage_projection
#axiomsClean producer_source
#axiomsClean producer_readback
#axiomsClean producer_frame
#axiomsClean consumer_source
#axiomsClean consumer_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.Physical
