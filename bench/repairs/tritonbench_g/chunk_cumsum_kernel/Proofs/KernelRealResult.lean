import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Kernels
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Contract
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.NativeIndex
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.RealComplete
import VeriTile.Meta.Specification

/-! Complete per-program Real interpretation of the selected FP32 source.
The independent target is the inclusive prefix of initial input memory.
Unspecified final-tail lanes cannot enter any active output. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.KernelRealResult
open VeriTile.Triton VeriTile.Torch
noncomputable section

abbrev source := repairedKernel .fp32 .fp32

def readReal (cell : MemCell) : ℝ := match cell.readAs .real with
  | some (some value) => value
  | _ => 0

theorem memory_correct (input output : RegionName) (T : Nat) (c : Config)
    (s : BlockState) (separate : output ≠ input) :
    Spec.Real (∃ t, exec (source input output T c.block).eraseDType.toAlgKernel s = some t ∧
      (∀ i, i < T → t.readMem output (s.pids 0*T+i) =
        mathematical (fun j => s.readMem input (s.pids 0*T+j)) T i) ∧
      (∀ r off, (r ≠ output ∨ ∀ i, i < T → off ≠ s.pids 0*T+i) →
        t.mem r off = s.mem r off)) :=
  RealIndexBridge.complete input output T c s separate

abbrev native_loop := NativeIndex.loop_native
abbrev native_indices := NativeIndex.source_indices
abbrev native_pointers := NativeIndex.active_pointers
abbrev original_overflow := NativeIndex.original_row_overflow

#specNonCircular mathematical avoiding [repairedKernel,sumLeftKernel]
#axiomsClean memory_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.KernelRealResult
