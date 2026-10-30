import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.HostMetadata
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPBounds
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Host
open VeriTile Triton FP.Structural
open FPStorage (Memory)
open FPStages (Stage)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000

def zeroRange (M : Algebra α) (m : Memory α) (region : RegionName) (size : Nat) : Memory α :=
  fun r off => if r=region ∧ off<size then M.literal (some .fp32) .real 0 else m r off
theorem zeroRange_readback (M : Algebra α) (m : Memory α) (region : RegionName) (size off : Nat)
    (ho : off<size) : zeroRange M m region size region off=M.literal (some .fp32) .real 0 := by
  simp [zeroRange,ho]
theorem zeroRange_frame (M : Algebra α) (m : Memory α) (region : RegionName) (size : Nat)
    (r : RegionName) (off : Nat) (hn : r≠region ∨ size ≤ off) : zeroRange M m region size r off=m r off := by
  simp only [zeroRange]
  split
  · rename_i h
    rcases hn with hr|ho
    · exact (hr h.1).elim
    · omega
  · rfl

def attentionSize (c : Dimensions) := c.heads*c.T*c.BT
def scratchSize (c : Dimensions) := c.NK*c.heads*c.T*16
def outputSize (c : Dimensions) := c.heads*c.T*c.V

/-- Ordered public producer body. Scratch is initialized after inter and
only on the large-K branch, exactly as in the repaired Python. -/
def producerExecute (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) : Option (Memory α) := do
  let a := zeroRange M m "attention" (attentionSize c)
  if c.heads*c.T=0 then return a
  let inter ← FPStages.execute M D (.inter c.producer scale) a
  if c.K ≤ 256 then
    FPStages.execute M D (.intra c.producer c.intraBK scale) inter
  else
    let scratch := zeroRange M inter "scratch" (scratchSize c)
    let split ← FPStages.execute M D (.split c.split scale) scratch
    FPStages.execute M D (.merge c.split) split

def producerReference (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) : Memory α :=
  let a := zeroRange M m "attention" (attentionSize c)
  if c.heads*c.T=0 then a else
    let inter := FPStages.reference M D (.inter c.producer scale) a
    if c.K ≤ 256 then FPStages.reference M D (.intra c.producer c.intraBK scale) inter
    else FPStages.reference M D (.merge c.split)
      (FPStages.reference M D (.split c.split scale) (zeroRange M inter "scratch" (scratchSize c)))

/-- The consumer accepts A and history as inputs; it does not invent a
history recurrence or couple itself to a newly introduced public wrapper. -/
def consumerExecute (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) : Option (Memory α) :=
  if outputSize c=0 then some m else FPStages.execute M D (.output c.consumer scale) m
def consumerReference (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) : Memory α :=
  if outputSize c=0 then m else FPStages.reference M D (.output c.consumer scale) m

theorem producer_source (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) : producerExecute M D c scale m=some (producerReference M D c scale m) := by
  unfold producerExecute producerReference
  simp only [FPStages.execute_eq]
  split
  · rfl
  · dsimp only [bind,Option.bind]
    split <;> rfl
theorem consumer_source (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) : consumerExecute M D c scale m=some (consumerReference M D c scale m) := by
  unfold consumerExecute consumerReference
  split <;> simp only [FPStages.execute_eq]

theorem producer_frame (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) (r : RegionName) (off : Nat)
    (ha : r≠"attention" ∨ attentionSize c ≤ off) (hs : r≠"scratch" ∨ scratchSize c ≤ off) :
    producerReference M D c scale m r off=m r off := by
  unfold producerReference
  split
  · exact zeroRange_frame M m "attention" _ r off ha
  · split
    · rw [FPStages.reference_bounded M D (.intra c.producer c.intraBK scale) _ r off ha,
        FPStages.reference_bounded M D (.inter c.producer scale) _ r off ha,
        zeroRange_frame M m "attention" _ r off ha]
    · rw [FPStages.reference_bounded M D (.merge c.split) _ r off ha,
        FPStages.reference_bounded M D (.split c.split scale) _ r off hs,
        zeroRange_frame M _ "scratch" _ r off hs,
        FPStages.reference_bounded M D (.inter c.producer scale) _ r off ha,
        zeroRange_frame M m "attention" _ r off ha]

theorem consumer_frame (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) (r : RegionName) (off : Nat)
    (ho : r≠"o" ∨ outputSize c ≤ off) : consumerReference M D c scale m r off=m r off := by
  unfold consumerReference
  split
  · rfl
  · exact FPStages.reference_bounded M D (.output c.consumer scale) m r off ho

theorem producer_empty (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) (he : c.heads*c.T=0) :
    producerExecute M D c scale m=some (zeroRange M m "attention" (attentionSize c)) := by
  rw [producer_source]
  simp [producerReference,he]
theorem consumer_empty (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) (he : outputSize c=0) : consumerExecute M D c scale m=some m := by
  simp [consumerExecute,he]

/-- A logical composition of the two existing calls, using the same G and
the supplied history input. This theorem is not a new Python entry point. -/
theorem calls_compose (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : Dimensions)
    (scale : ℝ) (m : Memory α) :
    (producerExecute M D c scale m).bind (consumerExecute M D c scale)=
      some (consumerReference M D c scale (producerReference M D c scale m)) := by
  rw [producer_source]
  exact consumer_source M D c scale _

#specNonCircular producerReference avoiding [producerExecute,FPStages.execute,FPGrid.invoke,FP.Structural.exec]
#specNonCircular consumerReference avoiding [consumerExecute,FPStages.execute,FPGrid.invoke,FP.Structural.exec]
#axiomsClean producer_source
#axiomsClean consumer_source
#axiomsClean producer_frame
#axiomsClean calls_compose
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.Host
