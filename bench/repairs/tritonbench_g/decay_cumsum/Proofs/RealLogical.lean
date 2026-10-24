import bench.repairs.tritonbench_g.decay_cumsum.Proofs.RealStorage
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.LogicalStorage

namespace VeriTile.Bench.Repairs.DecayCumsum.RealLogical
open VeriTile.Triton Geometry Grid PackedStorage Logical LogicalStorage
open VeriTile.Bench.TritonBenchG.DecayCumsum
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 5000000

def dq (read : RegionName → Nat → ℝ) (t : Nat) : ℝ :=
  read "dqi" t+read "dqo" t*Real.exp (read "g" t*Real.log 2)
def dk (span : Nat) (read : RegionName → Nat → ℝ) (t : Nat) : ℝ :=
  read "dki" t+read "dko" t*Real.exp ((read "g" (span-1)-read "g" t)*Real.log 2)

/-- Ordinary independent real sums and exponentials, indexed only by logical
chunk coordinates. The backward sum is local to its chunk. -/
def evaluate (kind : Kind) (span : Nat) (scale : ℝ) (read : RegionName → Nat → ℝ)
    (r : RegionName) (t : Nat) : ℝ :=
  match kind with
  | .forward => 1.44269504*∑ n : Fin (t+1), read "g" n.val
  | .prepare => if r="qg" then read "q" t*Real.exp (read "g" t*Real.log 2)*scale
      else read "k" t*Real.exp ((read "g" (span-1)-read "g" t)*Real.log 2)
  | .backward => if r="dqo" then dq read t else if r="dko" then dk span read t
      else ∑ n : Fin (span-t), (dq read (t+n.val)*read "q" (t+n.val)-dk span read (t+n.val)*read "k" (t+n.val))

def reference (h : Call) (kind : Kind) (scale : ℝ) (data : Data ℝ) (r : RegionName) (c : Coordinate h) : ℝ :=
  evaluate kind (chunkSpan h c.2.1.val) scale
    (fun s n => data s c.1.val (chunkStart h c.2.1.val+n) c.2.2.val) r (c.2.1.val-chunkStart h c.2.1.val)

theorem read_role (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) (i : Fin (RealIO.io kind h p scale).nIn) :
    (RealIO.io kind h p scale).inp i ∈ Roles.inputs kind := by
  cases kind <;> fin_cases i <;> simp [RealIO.io,RealTiles.prepareIO,RealTiles.backwardIO,
    PrepareCorrect.originalIO,BackwardCorrect.originalIO,Roles.inputs]

theorem values_evaluate (kind : Kind) (h : Call) (p : Phase) (positive : 0<p.span) (scale : ℝ)
    (xs : Fin (RealIO.io kind h p scale).nIn → Fin (RealIO.io kind h p scale).T → Fin (RealIO.io kind h p scale).B → ℝ)
    (read : RegionName → Nat → ℝ) (o : Fin (RealIO.io kind h p scale).nOut)
    (t : Fin (RealIO.io kind h p scale).T) (j : Fin (RealIO.io kind h p scale).B)
    (reads : ∀ i n, xs i n j=read ((RealIO.io kind h p scale).inp i) n.val) :
    RealIO.values kind h p scale xs o t j=evaluate kind p.span scale read ((RealIO.io kind h p scale).out o) t.val := by
  cases kind <;> fin_cases o
  all_goals simp [RealIO.values,ForwardCorrect.scan,PrepareCorrect.values,BackwardCorrect.values,
    BackwardCorrect.dq,BackwardCorrect.dk,BackwardCorrect.dg,evaluate,dq,dk,positive,reads,
    RealIO.io,RealTiles.prepareIO,RealTiles.backwardIO,PrepareCorrect.originalIO,BackwardCorrect.originalIO]

theorem expected_reference (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (heap : Heap ℝ) (data : Data ℝ) (snapshot : Snapshot h kind storage heap data)
    (c : Coordinate h) (o : Owner h c) (r : RegionName) (hr : r∈Roles.outputs kind) :
    ∃ i : RealStorage.OutputIndex h kind scale o.work,
      RealStorage.outputAddress h kind storage scale o.work i=address storage r (LogicalStorage.offset h c.1.val c.2.1.val c.2.2.val) ∧
      RealStorage.expected h kind storage scale heap o.work i=reference h kind scale data r c := by
  obtain ⟨channel,outEq⟩ := RealIO.output_covers kind h o.work.1 scale r hr
  let t : Fin (RealIO.io kind h o.work.1 scale).T := ⟨o.step.val,by simpa only [RealIO.timeSize] using o.step.isLt⟩
  let j : Fin (RealIO.io kind h o.work.1 scale).B := ⟨o.lane.val,by simpa only [RealIO.laneSize] using o.lane.isLt⟩
  refine ⟨(channel,t,j),?_,?_⟩
  · simp only [RealStorage.outputAddress,t,j,outEq,owner_offset h c o]
  · obtain ⟨_,spanEq,timeEq⟩ := owner_chunk h c o
    unfold reference
    rw [spanEq,timeEq,←outEq]
    apply values_evaluate kind h o.work.1 (phase_positive h o.work.1 (schedule_valid h o.work o.member)).2.1
    intro i n
    exact owner_read h kind storage heap data snapshot c o _ (read_role kind h o.work.1 scale i)
      n.val (RealStorage.time_bound kind h o.work.1 scale n)

theorem grid_correct (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (heap : Heap ℝ) (data : Data ℝ) (snapshot : Snapshot h kind storage heap data) :
    ∃ final, Run.run (RealStorage.execute h kind storage scale) (schedule h) heap=some final ∧
      (∀ r∈Roles.outputs kind, ∀ i, crop h kind storage final r i=reference h kind scale data r (coordinates h i)) ∧
      (∀ addr, (∀ w∈schedule h, ¬footprint h kind storage w addr) → final addr=heap addr) := by
  obtain ⟨final,run,output,frame⟩ := RealStorage.grid_complete h kind storage scale heap
  refine ⟨final,run,?_,frame⟩
  intro r hr i
  let c := coordinates h i
  let o := owner h c
  obtain ⟨index,addressEq,valueEq⟩ := expected_reference h kind storage scale heap data snapshot c o r hr
  have out := output o.work o.member index
  rw [addressEq,valueEq] at out
  exact out

#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.DecayCumsum.RealLogical
