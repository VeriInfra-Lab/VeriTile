import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.PipelineLocality
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealTensors
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativePipeline

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.FiniteExecution
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 3000000

def inputRegions : List RegionName := ["Q","K","V","DO"]

/-- A proof-only extension of four input buffers beyond the actual tensor.
It leaves every finite source window and all other regions unchanged. -/
def extend {α : Type} (c : Shape) (μ : RegionName→Nat→α) (mem : Grid.Memory α) : Grid.Memory α :=
  fun r o=>if r∈inputRegions ∧ c.elements≤o then .mk .fp32 (μ r o) else mem r o

theorem extend_agree {α : Type} (c : Shape) (μ : RegionName→Nat→α) (mem : Grid.Memory α) :
    MemoryAgreement.Agree c.elements mem (extend c μ mem) := by
  intro r o ho
  simp [extend,show ¬c.elements≤o by omega]

theorem real (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (mem : Grid.Memory Erased.R) (μ : RegionName→Nat→ℝ)
    (inputs : ∀ r∈inputRegions,∀ o,o<c.elements → (mem r o).read .fp32=some (μ r o))
    (zero : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 (some 0)) :
    ∃ final,Grid.wholeRun Erased.algebra Erased.model Erased.model Erased.model c scale fw pw bw mem=some final ∧
      Erased.TensorValues c scale μ final ∧
      (∀ r o,¬Grid.Written c r o → final r o=mem r o) := by
  let expanded := extend c (fun r o=>some (μ r o)) mem
  have global : ∀ r∈inputRegions,∀ o,(expanded r o).read .fp32=some (μ r o) := by
    intro r hr o
    by_cases inside : o<c.elements
    · simpa [expanded,extend,show ¬c.elements≤o by omega] using inputs r hr o inside
    · simp only [expanded,extend,if_pos (show r∈inputRegions ∧ c.elements≤o from ⟨hr,by omega⟩)]
      rfl
  have z : ∀ i : Coord c,expanded "DQ" (offset c i)=.mk .fp32 (some 0) := by
    intro i
    simpa [expanded,extend,inputRegions] using zero i
  obtain ⟨reference,ranReference,values,_⟩ := Erased.whole_execution_tensors c scale fw pw bw hf hp hb expanded μ global z
  have refEq : reference=Grid.wholeReference Erased.model Erased.model Erased.model c scale fw pw bw expanded :=
    Option.some.inj (ranReference.symm.trans (Grid.whole_success Erased.algebra Erased.model Erased.model Erased.model c scale fw pw bw expanded))
  let final := Grid.wholeReference Erased.model Erased.model Erased.model c scale fw pw bw mem
  have agree : MemoryAgreement.Agree c.elements final reference := by
    rw [refEq]
    exact MemoryAgreement.whole Erased.model Erased.model Erased.model c scale fw pw bw
      (Grid.FullSchedule.legal _ _ hf) (Grid.FullSchedule.legal _ _ hp) (Grid.FullSchedule.legal _ _ hb)
      mem expanded (extend_agree c _ mem)
  refine ⟨final,Grid.whole_success Erased.algebra Erased.model Erased.model Erased.model c scale fw pw bw mem,?_,?_⟩
  · constructor
    · intro i;rw [agree "Out" _ (offset_lt c i)];exact values.output i
    · intro i;rw [agree "DQ" _ (offset_lt c i)];exact values.dq i
    · intro i;rw [agree "DK" _ (offset_lt c i)];exact values.dk i
    · intro i;rw [agree "DV" _ (offset_lt c i)];exact values.dv i
  · exact Grid.wholeReference_frame Erased.model Erased.model Erased.model c scale fw pw bw mem
      (Grid.FullSchedule.legal _ _ hf) (Grid.FullSchedule.legal _ _ hp) (Grid.FullSchedule.legal _ _ hb)

theorem native (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (mem : Grid.Memory Native.Graph) (μ : RegionName→Nat→Native.Graph)
    (inputs : ∀ r∈inputRegions,∀ o,o<c.elements → mem r o=.mk .fp32 (μ r o))
    (zero : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 (.word 0)) :
    ∃ final,Native.Selected.run one c scale owners fw pw bw mem=some final ∧
      Native.Selected.Values one c scale owners (Native.Selected.input c μ) final ∧
      (∀ r o,¬Grid.Written c r o → final r o=mem r o) := by
  let expanded := extend c μ mem
  have global : Native.Selected.Inputs expanded μ := by
    intro r hr o
    by_cases inside : o<c.elements
    · simpa [expanded,extend,show ¬c.elements≤o by omega] using inputs r hr o inside
    · simp [expanded,extend,show r∈inputRegions from hr,show c.elements≤o by omega]
  have z : ∀ i : Coord c,expanded "DQ" (offset c i)=.mk .fp32 (.word 0) := by
    intro i
    simpa [expanded,extend,inputRegions] using zero i
  obtain ⟨reference,ranReference,values,_⟩ := Native.Selected.whole_execution_graph one c scale owners fw pw bw hf hp hb expanded μ global z
  have refEq : reference=Native.Selected.stores one c scale owners fw pw bw expanded :=
    Option.some.inj (ranReference.symm.trans (Native.Selected.run_success one c scale owners fw pw bw expanded))
  let final := Native.Selected.stores one c scale owners fw pw bw mem
  have agree : MemoryAgreement.Agree c.elements final reference := by
    rw [refEq]
    exact MemoryAgreement.native_whole one c scale owners fw pw bw
      (Grid.FullSchedule.legal _ _ hf) (Grid.FullSchedule.legal _ _ hp) (Grid.FullSchedule.legal _ _ hb)
      mem expanded (extend_agree c μ mem)
  refine ⟨final,Native.Selected.run_success one c scale owners fw pw bw mem,?_,?_⟩
  · refine ⟨?_,?_,?_,?_⟩
    · intro i;rw [agree "Out" _ (offset_lt c i)];exact values.1 i
    · intro i;rw [agree "DQ" _ (offset_lt c i)];exact values.2.1 i
    · intro i;rw [agree "DK" _ (offset_lt c i)];exact values.2.2.1 i
    · intro i;rw [agree "DV" _ (offset_lt c i)];exact values.2.2.2 i
  · exact Native.Selected.stores_frame one c scale owners fw pw bw mem
      (Grid.FullSchedule.legal _ _ hf) (Grid.FullSchedule.legal _ _ hp) (Grid.FullSchedule.legal _ _ hb)

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.FiniteExecution
