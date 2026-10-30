import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.GridOutput

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.KernelResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section

/-- Complete quoted-source grid execution over raw regions/strides and state,
independent low32 dense signed sums, and the full output-exterior frame. -/
theorem low32 (cast : Word32 → Byte)
    (castLeaf : ∀ word,DefinedCast word → cast word=referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (initial : State Word32) (defined : FP32Domain layout s initial) (bytes : UInt8Domain layout s initial)
    (hout : OutputLayout layout s) (separate : layout.output≠layout.input ∧ layout.output≠layout.weight) :
    GridExec cast rest (parameters layout s c) initial (List.range (s.programs c))
      (flatten initial) (gridMemory layout s c initial) ∧
    (∀ m n,m<s.m → n<s.n →
      gridMemory layout s c initial (layout.output,VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001.outputOffset s m n)=
        int32Cell (matrixWordTarget layout s initial m n)) ∧
    (∀ loc,loc.1≠layout.output ∨ s.m*s.n≤loc.2 →
      gridMemory layout s c initial loc=flatten initial loc) :=
  ⟨grid_low32 cast castLeaf rest layout s c legal initial defined bytes hout separate,
    grid_values layout s c initial hout,grid_frame layout s c initial hout⟩


/-- Complete source grid and primitive-native outputs share one raw backing.
Only the existing FP32-to-signed8, fixed K32 and add.u32 leaves are premises. -/
theorem native (cast : Word32 → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word32 → Word32)
    (add32 : Word32 → Word32 → Word32) (leaves : NumericLeaves cast instruction)
    (addLeaf : Add32Leaf add32) (rest : Algebra Word32)
    (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (initial : State Word32) (defined : FP32Domain layout s initial) (bytes : UInt8Domain layout s initial)
    (hout : OutputLayout layout s) (separate : layout.output≠layout.input ∧ layout.output≠layout.weight) :
    GridExec cast rest (parameters layout s c) initial (List.range (s.programs c))
      (flatten initial) (gridMemory layout s c initial) ∧
    (∀ m n,m<s.m → n<s.n →
      gridMemory layout s c initial (layout.output,VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001.outputOffset s m n)=
        int32Cell (nativeModularAccumulation cast instruction add32 s c
          (inputWords layout initial m) (packedWords layout initial n))) ∧
    (∀ loc,loc.1≠layout.output ∨ s.m*s.n≤loc.2 →
      gridMemory layout s c initial loc=flatten initial loc) := by
  obtain ⟨run,values,frame⟩ := low32 cast leaves.cast rest layout s c legal initial defined bytes hout separate
  refine ⟨run,?_,frame⟩
  intro m n hm hn
  rw [values m n hm hn,nativeAccumulation_low32 cast instruction add32 leaves addLeaf s c legal
    (inputWords layout initial m) (packedWords layout initial n) (defined m hm)]
  rfl


#axiomsClean low32
#axiomsClean native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.KernelResults
