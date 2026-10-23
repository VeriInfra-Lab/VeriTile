import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelPrefix
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReference
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelReversePointers

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReverseInit
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelReversePointers
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem fixed (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α) (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    Fixed c p (KernelPrefix.backward M c s seed u).val := by
  constructor <;>
    simp [KernelPrefix.backward,hh,hk,hv,State.setReg,integralFloorDiv,Shape.K,Shape.V]
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem registers (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α) (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    let v := (KernelPrefix.backward M c s seed u).val
    v.regs .ptr [64,16] "S"=some (ptr "S" (off c p (c.N-2) 0)) ∧
    v.regs .ptr [64,16] "DI"=some (ptr "DI" (off c p (c.N-2) 0)) ∧
    v.regs .ptr [64,16] "DS"=some (ptr "DS" (off c p (c.N-1) 0)) ∧
    v.regs .ptr [] "D"=some (fun _ : TileIndex [] => (("D" : RegionName),gate c p 0)) ∧
    v.regs .ptr [] "DG"=some (fun _ : TileIndex [] => (("DG" : RegionName),gradient c p 0)) ∧
    v.regs .real [64,16] "Dacc"=some (fun _ => KernelReference.zero M) := by
  simp [KernelPrefix.backward,ptr,off,gate,gradient,address,layout,launch,gateAddress,gradientAddress,
    TritonBenchG.ChunkGateRecurrence.RealProgram.address,hh,hk,hv,State.setReg,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    integralFloorDiv,integralMod,Shape.K,Shape.V,KernelReference.zero]
  all_goals repeat' constructor
  all_goals funext i
  all_goals simp [FP.Structural.bop,KernelProgram.ptr,KernelReversePointers.off,
    address,layout,launch,TritonBenchG.ChunkGateRecurrence.RealProgram.address,
    Broadcast.leftIndex,Broadcast.rightIndex,Shape.K,Shape.V]

#axiomsClean fixed
#axiomsClean registers
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelReverseInit
