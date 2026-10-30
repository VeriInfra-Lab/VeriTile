import bench.tritonbench_g.f8_conversion_utils.Proofs.RealSource

namespace VeriTile.Bench.Repairs.F8ConversionUtils.RealSource
open VeriTile.Triton VeriTile.Bench.TritonBenchG.F8ConversionUtils
noncomputable section
set_option maxRecDepth 24000

def backedState (iw ow input output pid : Nat) (memory : Cells.Heap) : BlockState where
  mem := fun r i => MemCell.real (if r="x" then memory (Memory.address input iw i)
    else memory (Memory.address output ow i))
  regs := fun _ _ _ => none
  pids := fun _ => pid

def ProgramResult {n iw ow : Nat} (direction : Direction) (input : TypedSource.Input)
    (format : TypedSource.Format) (x : Native.Allocation n iw) (y : Native.Allocation n ow)
    (c pid : Nat) (programBefore : Cells.Heap) : Prop :=
  let initial := backedState iw ow (x.base+Chunking.chunkStart c*iw)
    (y.base+Chunking.chunkStart c*ow) pid programBefore
  ∃ final,exec (kernel direction input format "y" "x" (Chunking.chunkLength n c) 1024).eraseDType.toAlgKernel initial=some final ∧
    (∀ i,final.readMem "y" i=
      Lowering.program (Cells.primitives iw ow x.positive y.positive) direction.doubleStore id
        (x.base+Chunking.chunkStart c*iw) (y.base+Chunking.chunkStart c*ow)
        (Chunking.chunkLength n c) pid programBefore
        (Memory.address y.base ow (Chunking.chunkStart c+i))) ∧
    (∀ (r : RegionName) i,r≠("y" : RegionName) → final.mem r i=initial.mem r i)

/-- The arbitrary-Real source state exists for every actual pre-program heap.
There is no finite-word codec and no assumed successful source execution. -/
theorem programs {n iw ow : Nat} (direction : Direction) (input : TypedSource.Input)
    (format : TypedSource.Format) (x : Native.Allocation n iw) (y : Native.Allocation n ow)
    (c pid : Nat) (launched : pid<NativeIndex.grid (Chunking.chunkLength n c)) (programBefore : Cells.Heap) :
    ProgramResult direction input format x y c pid programBefore := by
  apply program_refines direction input format x y (Chunking.chunkStart c) (Chunking.chunkLength n c) pid
    (Chunking.chunk_small n c) launched (Chunking.chunk_active n c) programBefore "y" "x"
    (backedState iw ow (x.base+Chunking.chunkStart c*iw) (y.base+Chunking.chunkStart c*ow) pid programBefore) rfl
  · intro i
    simp [backedState,BlockState.readMem,Memory.address,Nat.add_mul,Nat.add_assoc]
  · intro i
    simp [backedState,BlockState.readMem,Memory.address,Nat.add_mul,Nat.add_assoc]

#axiomsClean programs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.RealSource
