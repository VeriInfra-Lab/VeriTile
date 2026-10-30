import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.RawGridProof
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Native

namespace VeriTile.Bench.Repairs.Int4Matmul.RawNative
open VeriTile.Triton KernelExecution
noncomputable section

/-- Complete quoted-program execution and native lane-serialization result,
with scalar/fixed-K8 ISA ports and the original atomic schedule. All inputs
are raw packed/scalar memory, shape/stride geometry or primitive models. -/
theorem matmul_grid_native (x : Raw.Arguments) (geometry : Raw.Geometry x)
    (tile : Tile) (member : tile∈Host.sourceTiles)
    (instructions : Native.Instructions) (isa : Native.ISA) (rest : FP.Structural.Algebra Word)
    (initial : ScalarState Word) (scratch : RegisterState Word) (lanes : List Lane)
    (complete : lanes.Perm (rawLanes x.shape tile)) :
    (∀ l∈lanes, ∃ final, KernelExecution.exec (Native.ops instructions isa) rest (Raw.parameters x tile)
      (atProgram (attach (zeroOutput x.shape (Native.ops instructions isa) initial) scratch) l.pid l.split)=some final) ∧
    (∀ row col,row<x.shape.m → col<x.shape.n →
      (Raw.launch x tile (Native.ops instructions isa) rest initial scratch lanes).mem "out" (cAddress x.shape row col)=
        FP.Structural.Cell.mk .real (referenceOutput x.shape tile (Native.ops instructions isa) initial row col
          (laneSchedule x.shape tile lanes) 0)) ∧
    (∀ (r : RegionName) (i : Nat), r ≠ "out" ∨ x.shape.m * x.shape.n ≤ i →
      (Raw.launch x tile (Native.ops instructions isa) rest initial scratch lanes).mem r i=(attach initial scratch).mem r i) := by
  refine ⟨?_,?_,?_⟩
  · intro l _
    obtain ⟨final,run,_⟩ := source_program_events (Native.ops instructions isa) rest (Raw.parameters x tile)
      (atProgram (attach (zeroOutput x.shape (Native.ops instructions isa) initial) scratch) l.pid l.split)
    exact ⟨final,run⟩
  · exact Raw.launch_value x geometry tile member (Native.ops instructions isa) rest initial scratch lanes complete
  · exact Raw.launch_frame x geometry tile member (Native.ops instructions isa) rest initial scratch lanes complete

#axiomsClean matmul_grid_native
#specNonCircular referenceOutput avoiding [KernelSource.kernel,Raw.launch,parallel,computed,partialValue]
end
end VeriTile.Bench.Repairs.Int4Matmul.RawNative
