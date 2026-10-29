import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IONativeSafety
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IONativeMath

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int4Matmul.FPSoftware (Word)
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open IONativeGraph
noncomputable section

theorem implements (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17) :
    (IOData.program Word c (jobs actual c i)).Implements (IONativeMath.formula mma c i) := by
  intro x _ bounds hb initial binding
  refine ⟨IONativeSafety.safe actual mma ports c valid i x bounds hb initial,?_⟩
  obtain ⟨final,ran,values,frame⟩ := Grid.whole_grid (mma (tunings i).opcode) (params c.call i)
    (legal c.call (c.valid valid) i) (geometry_valid c.call (c.valid valid) i)
    (by change ("A" : RegionName)≠"C"; decide)
    (by change ("Q" : RegionName)≠"C"; decide)
    (by change ("S" : RegionName)≠"C"; decide)
    (by change ("Z" : RegionName)≠"C"; decide) initial
  have post : GridExecution.Post "C" (Geometry.address (Grid.geometry (params c.call i)))
      (fun j : Grid.Index (params c.call i)=>Cell.mk .fp16
        (IONativeMath.value (mma (tunings i).opcode) c i (IOData.arrays c x) j.1.val j.2.val))
      (fun _=>True) initial final := by
    refine ⟨?_,frame⟩
    intro j _
    exact (values j trivial).trans (congrArg (Cell.mk .fp16)
      (IONativeMath.output (mma (tunings i).opcode) c valid i (jobs actual c i) x initial binding
        j.1.val j.2.val j.1.isLt j.2.isLt))
  refine ⟨final,?_,IOExecution.output_binding c (jobs actual c i) x _ i initial final post⟩
  rw [IONativeAdapter.grid_run actual mma ports,Native.grid_transport actual mma ports]
  exact ran

/-- The retained physical API serializes this very execution. Every packed
input byte, every half output byte and the exact exterior byte frame remain
covered by the original `Physical.whole_bytes` theorem. -/
theorem physical_source (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (c : RawKernel.Layout) (i : Fin 17)
    (initial : Physical.Bytes) :
    (runSchedule (jobs actual c i) (Physical.load initial)).map (Physical.commit initial)=
      Physical.source actual c.call i initial := by
  rw [IONativeAdapter.grid_run actual mma ports]
  rfl

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONative
