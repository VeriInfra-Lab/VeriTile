import bench.tritonbench_g.chunk_gla_fwd.Proofs.OutputTail

/-! Repaired per-program execution, values and tagged-memory frames. The existing
source theorems are reused through the proved zero-padding bridge; no execution
or output-value premise is introduced. Full grid composition is separate. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.LocalReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

section Inter
open RealInterMath RealInterProgram RealInterPointers RealInterLoop RealInterFinal
theorem inter_executes (p : Layout) (r : Regions) (scale : ℝ) (s : BlockState)
    (hu : s.undef = fun _ _ => 0) :
    ∃ sf, exec (SourceBridge.interNew p r scale).eraseDType.toAlgKernel s = some sf ∧
      (enabled p (launch s) → ∀ i : TileIndex [p.BC,p.BC], active p (launch s) i.1.val i.2.1.val →
        sf.readMem r.output (outputAddress p (launch s) i.1.val i.2.1.val) =
          expected p (launch s) (inputs r s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint p (launch s) r.output) s sf := by
  rw [SourceBridge.inter_exec p r scale s
    (RealInterSafety.source_flatten_ok p r .fp32 .fp32 .fp32 .fp32 scale Bool.false) hu]
  exact RealInterExecution.source_executes p r .fp32 .fp32 .fp32 .fp32 scale Bool.false s (by intro region off; rw [hu])
end Inter

section Intra
open RealIntraMath RealIntraProgram RealIntraMemory RealIntraLoop
open RealIntraSetup (inputs)
theorem intra_executes (p : Layout) (r : Regions) (scale : ℝ) (s : BlockState)
    (hu : s.undef = fun _ _ => 0)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hcols : columns p (launch s) ≤ p.BT)
    (hK : p.K ≤ p.BK) :
    ∃ sf, exec (SourceBridge.intraNew p r scale).eraseDType.toAlgKernel s = some sf ∧
      (enabled p (launch s) → ∀ j, j < columns p (launch s) → ∀ i : Fin p.BC,
        rowActive p (launch s) i.val → sf.readMem r.output (outputAddress p (launch s) i.val j) =
          expected p (launch s) (inputs r s) scale i.val j) ∧
      BlockState.WriteWithin (RealIntraExecution.footprint p (launch s) r.output) s sf := by
  rw [SourceBridge.intra_exec p r scale s
    (RealIntraSafety.source_flatten_ok p r .fp32 .fp32 .fp32 scale Bool.false) hu]
  exact RealIntraExecution.source_executes p r .fp32 .fp32 .fp32 scale Bool.false s hkey hgate hcols hK
end Intra

section Split
open RealSplitMath RealSplitProgram RealSplitMemory RealSplitLoop
open RealSplitSetup (inputs)
theorem split_executes (p : Layout) (r : Regions) (scale : ℝ) (s : BlockState)
    (hu : s.undef = fun _ _ => 0)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output)
    :
    ∃ sf, exec (SourceBridge.splitNew p r scale).eraseDType.toAlgKernel s = some sf ∧
      (enabled p (launch p s) → ∀ j, j < columns p (launch p s) → ∀ i : Fin p.BC,
        rowActive p (launch p s) i.val → sf.readMem r.output (outputAddress p (launch p s) i.val j) =
          expected p (launch p s) (inputs r s) scale i.val j) ∧
      BlockState.WriteWithin (RealSplitExecution.footprint p (launch p s) r.output) s sf := by
  rw [SourceBridge.split_exec p r scale s
    (RealSplitSafety.source_flatten_ok p r .fp32 .fp32 .fp32 scale Bool.false) hu]
  exact RealSplitExecution.source_executes p r .fp32 .fp32 .fp32 scale Bool.false s hkey hgate
end Split

section Merge
open RealMergeProgram
theorem merge_executes (p : Layout) (A A2 : RegionName) (s : BlockState)
    (hu : s.undef = fun _ _ => 0)
    (hinj : NoCollision p (launch s)) :
    ∃ sf, exec (SourceBridge.mergeNew p A A2).eraseDType.toAlgKernel s = some sf ∧
      (∀ i, active p (launch s) i → sf.readMem A2 (outputAddress p (launch s) i) = expected p (launch s) (s.readMem A) i) ∧
      BlockState.WriteWithin (footprint p (launch s) A2) s sf := by
  rw [SourceBridge.merge_exec p A A2 s
    (RealMergeSafety.source_flatten_ok p A A2 .fp32 Bool.false) hu]
  exact RealMergeExecution.source_executes p A A2 .fp32 Bool.false s hinj
end Merge


section Output
open RealOutputIO

def outputInputs (r : Regions) (s : BlockState) : Inputs :=
  ⟨s.readMem r.query,s.readMem r.value,s.readMem r.gate,s.readMem r.history,s.readMem r.attention⟩
def outputFootprint (p : Layout) (l : Launch) (out : RegionName) : WriteFootprint :=
  fun cell => cell.1=out ∧ ∃ i, active p l i ∧ cell.2=outputAddress p l i

theorem output_executes (p : Layout) (r : Regions) (scale : ℝ) (s : BlockState)
    (hu : s.undef = fun _ _ => 0)
    (hinj : ∀ i j, active p (launch s) i → active p (launch s) j →
      outputAddress p (launch s) i=outputAddress p (launch s) j → i=j) :
    ∃ sf, exec (SourceBridge.outputNew p r scale).eraseDType.toAlgKernel s=some sf ∧
      (∀ i, active p (launch s) i →
        sf.readMem r.output (outputAddress p (launch s) i)=formula p scale (launch s) (outputInputs r s) i) ∧
      BlockState.WriteWithin (outputFootprint p (launch s) r.output) s sf := by
  have hinj' : ∀ i j : TileIndex [p.BT,p.BV],
      (RealOutput.tIndex s p.BT i.1.val < p.T ∧ RealOutput.vIndex s p.BV i.2.1.val < p.V) →
      (RealOutput.tIndex s p.BT j.1.val < p.T ∧ RealOutput.vIndex s p.BV j.2.1.val < p.V) →
      RealOutput.outOffset s p.valueHead p.valueTime p.BT p.BV i =
        RealOutput.outOffset s p.valueHead p.valueTime p.BT p.BV j → i=j := by
    simpa only [active,outputAddress,launch,RealOutput.tIndex,RealOutput.vIndex,
      RealOutput.outOffset,Nat.mul_one] using hinj
  obtain ⟨sf,he,ho,hf⟩ := OutputTail.source_executes r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV s hu hinj'
  refine ⟨sf,?_,?_,?_⟩
  · rw [SourceBridge.output_exec p r scale s
      (RealOutputSafety.source_flatten_ok .fp32 .fp32 .fp32 .fp32 .fp32
        r.query r.value r.gate r.history r.output r.attention
        p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey
        scale p.T p.K p.V p.BT p.BK p.BV) hu]
    unfold SourceBridge.outputOld VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput.original
    rw [exec,RealOutputSource.storage_erasure]
    exact he
  · intro i ha
    simpa only [outputInputs,launch,formula,outputAddress,RealOutput.outOffset,RealOutput.tIndex,RealOutput.vIndex,
      RealOutput.cgfOutput,RealOutput.numKB,RealOutput.qgElem,RealOutput.hGuarded,RealOutput.vGuarded,RealOutput.aMasked,
      RealOutput.qElem,RealOutput.gElem,RealOutput.hElem,RealOutput.vElem,RealOutput.aElem,RealOutput.kIndex,Nat.mul_one]
      using ho i ha
  · intro region off hn
    apply hf region off
    simpa only [outputFootprint,RealOutput.footprint,launch,outputAddress,active,
      RealOutput.outOffset,RealOutput.tIndex,RealOutput.vIndex,Nat.mul_one] using hn
end Output

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.LocalReal
