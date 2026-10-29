import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPCharacterizationValues
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Characterization
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem all_active (mode : Bool) (i : CIndex (parameters mode)) : active (parameters mode) initial i := by
  have hr : i.1.val<64 := i.1.isLt
  have hc : i.2.1.val<128 := i.2.1.isLt
  simp [active,rowBlock,columnBlock,firstRow,rowsInGroup,blocksM,blocksN,parameters,initial,hr,hc]

theorem output_offset (mode : Bool) (i : CIndex (parameters mode)) :
    outputOffset (parameters mode) initial i = 128*i.1.val+i.2.1.val := by
  simp [outputOffset,rowBlock,columnBlock,firstRow,rowsInGroup,blocksM,blocksN,parameters,initial]

theorem output_bounded (mode : Bool) (i : CIndex (parameters mode)) :
    outputOffset (parameters mode) initial i<8192 := by
  rw [output_offset]
  have hr : i.1.val<64 := i.1.isLt
  have hc : i.2.1.val<128 := i.2.1.isLt
  omega

theorem output_injective (mode : Bool) : ∀ i j,
    active (parameters mode) initial i → active (parameters mode) initial j →
    outputOffset (parameters mode) initial i=outputOffset (parameters mode) initial j → i=j := by
  intro i j _ _ he
  rw [output_offset,output_offset] at he
  rcases i with ⟨r,c,u⟩;rcases j with ⟨r',c',u'⟩
  have hc : c.val<128 := c.isLt
  have hc' : c'.val<128 := c'.isLt
  have hr : r=r' := Fin.ext (by dsimp at he;omega)
  have hcol : c=c' := Fin.ext (by dsimp at he;omega)
  cases hr;cases hcol;cases u;cases u';rfl

def logicalReference (value : Word) (r : RegionName) (o : Nat) : Cell Word :=
  if r="C" ∧ o<8192 then .mk .fp16 value else initial.mem r o

theorem whole_memory_of_post (mode : Bool) (t : State Word) (value : Word)
    (h : FPTyped.Post (parameters mode) initial t (fun _ => value)) :
    t.mem=logicalReference value := by
  funext r o
  by_cases hw : r="C" ∧ o<8192
  · rcases hw with ⟨hr,ho⟩;subst r
    let i : CIndex (parameters mode) :=
      (⟨o/128,by change o/128<64;omega⟩,⟨o%128,Nat.mod_lt _ (by decide)⟩,PUnit.unit)
    have hoff : outputOffset (parameters mode) initial i=o := by
      rw [output_offset]
      dsimp [i]
      omega
    have hm:=h.1 i (all_active mode i)
    rw [hoff] at hm
    simpa [parameters,logicalReference,ho] using hm
  · have hf : r≠(parameters mode).output ∨ ∀ i, active (parameters mode) initial i →
        o≠outputOffset (parameters mode) initial i := by
      by_cases hr : r="C"
      · right
        intro i hi he
        have hb:=output_bounded mode i
        exact hw ⟨hr,he ▸ hb⟩
      · exact Or.inl hr
    simpa only [logicalReference,if_neg hw] using h.2 r o hf

/-- Word-addressed physical envelope for nonzero logical pointer origins.
The prefix is outside every source region's nonnegative offsets. Byte ABI and
actual GPU pointer equality are separately checked by the capture runner. -/
def physicalView (mem : RegionName → Nat → Cell Word) (r : RegionName) (o : Nat) : Cell Word :=
  if o<8 then initial.mem r 0 else mem r (o-8)

def physicalReference (value : Word) (r : RegionName) (o : Nat) : Cell Word :=
  if r="C" ∧ 8≤o ∧ o<8200 then .mk .fp16 value else physicalView initial.mem r o

theorem physical_reference_bridge (value : Word) :
    physicalView (logicalReference value)=physicalReference value := by
  funext r o
  by_cases hp : o<8
  · simp [physicalView,physicalReference,hp,show ¬8≤o from by omega]
  · have h : o-8<8192 ↔ o<8200 := by omega
    simp [physicalView,physicalReference,logicalReference,hp,show 8≤o from by omega,h]

theorem complete_legal_tile_execution (mode : Bool) : ∃ original separate fused,
    FP.Structural.exec (FPTyped.backend (FPSoftware.scalar SourceBits.unused) FPSoftware.softwareDot)
      (FPTyped.source (parameters mode)) initial=some original ∧
    FP.Structural.exec (FPCompiled.backend (FPSoftware.scalar FPCompiled.unused)
      (FPCompiled.plannedDotAcc FPCompiled.ascendingPlan)) (FPCompiled.source (parameters mode)) initial=some separate ∧
    FPFused.exec (FPSoftware.scalar FPCompiled.unused) (FPCompiled.plannedDotAcc FPCompiled.ascendingPlan)
      (parameters mode) initial=some fused ∧
    original.mem=logicalReference (halfWord (separateOutputBits mode)) ∧
    separate.mem=logicalReference (halfWord (separateOutputBits mode)) ∧
    fused.mem=logicalReference (halfWord (fusedOutputBits mode)) ∧
    physicalView original.mem=physicalReference (halfWord (separateOutputBits mode)) ∧
    physicalView separate.mem=physicalReference (halfWord (separateOutputBits mode)) ∧
    physicalView fused.mem=physicalReference (halfWord (fusedOutputBits mode)) := by
  obtain ⟨a,ha,hpa⟩ := SourceBits.source_readback_correctness SourceBits.unused
    (parameters mode) initial (output_injective mode)
  obtain ⟨b,hb,hpb⟩ := FPCompiled.planned_readback_correctness FPCompiled.ascendingPlan FPCompiled.unused
    (parameters mode) initial (output_injective mode)
  obtain ⟨c,hc,hpc⟩ := FPFused.planned_readback_correctness FPCompiled.ascendingPlan FPCompiled.unused
    (parameters mode) initial (output_injective mode)
  have va : SourceBits.sourceOutput (parameters mode) initial = fun _ => halfWord (separateOutputBits mode) :=
    funext (original_output_all_lanes mode)
  have vb : FPCompiled.plannedOutput FPCompiled.ascendingPlan (parameters mode) initial = fun _ => halfWord (separateOutputBits mode) :=
    funext (separate_output_all_lanes mode)
  have vc : FPFused.plannedOutput FPCompiled.ascendingPlan (parameters mode) initial = fun _ => halfWord (fusedOutputBits mode) :=
    funext (fused_output_all_lanes mode)
  have ma := whole_memory_of_post mode a _ (va ▸ hpa)
  have mb := whole_memory_of_post mode b _ (vb ▸ hpb)
  have mc := whole_memory_of_post mode c _ (vc ▸ hpc)
  exact ⟨a,b,c,ha,hb,hc,ma,mb,mc,ma ▸ physical_reference_bridge _,mb ▸ physical_reference_bridge _,mc ▸ physical_reference_bridge _⟩

#axiomsClean all_active
#axiomsClean output_offset
#axiomsClean output_bounded
#axiomsClean output_injective
#axiomsClean whole_memory_of_post
#axiomsClean physical_reference_bridge
#axiomsClean complete_legal_tile_execution
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Characterization
