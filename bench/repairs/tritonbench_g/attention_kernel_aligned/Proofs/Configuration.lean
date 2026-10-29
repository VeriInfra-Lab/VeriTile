import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Binding
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan
open VeriTile Triton Full Whole
structure Configuration where
  bm : Width
  bn : Width
  dim : Width
  eightWarps : Bool
  stage : Fin 4

def Configuration.numWarps (c : Configuration) : Nat := if c.eightWarps then 8 else 4

def Configuration.Matches (c : Configuration) (p : Parameters) : Prop :=
  c.bm.value=p.blockM ∧ c.bn.value=p.blockN ∧ c.dim.value=p.head ∧
  c.numWarps=p.numWarps ∧ c.stage.val+1=p.numStages

theorem configuration_exists (p : Parameters) (s : Selected p)
    (hw : p.numWarps∈[4,8]) (hs : 1≤p.numStages ∧ p.numStages≤4) :
    ∃ c : Configuration,c.Matches p := by
  let stage : Fin 4:=⟨p.numStages-1,by omega⟩
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hw
  rcases hw with hw|hw
  · exact ⟨⟨widthOf p.blockM,widthOf p.blockN,widthOf p.head,Bool.false,stage⟩,
      widthOf_value _ s.blockM,widthOf_value _ s.blockN,widthOf_value _ s.head,
      hw.symm,by dsimp [stage];omega⟩
  · exact ⟨⟨widthOf p.blockM,widthOf p.blockN,widthOf p.head,Bool.true,stage⟩,
      widthOf_value _ s.blockM,widthOf_value _ s.blockN,widthOf_value _ s.head,
      hw.symm,by dsimp [stage];omega⟩

theorem wrapper_configuration (p : Parameters) (g : Guard p) (abi : InputABI p) :
    ∃ c : Configuration,c.Matches p :=
  configuration_exists p g.selected abi.launchConfig.1 abi.launchConfig.2

#axiomsClean configuration_exists
#axiomsClean wrapper_configuration
#specNonCircular Configuration.Matches avoiding [FP.Structural.exec]
end VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan
