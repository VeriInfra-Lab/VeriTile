import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelPrefix

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelInitial
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def initialCarry (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α) : Lane → α := fun i =>
  if seed then M.cast (some .fp32) .real .real (raw "Seed" .real (fun e j => seedAddress c p s e.val j.val) u i)
  else M.literal (some .fp32) .real 0
def stored (M : Algebra α) (x : α) : Cell α :=
  .mk .real (M.cast (some .fp32) .real .real x)

theorem forward_pids (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (u : State α) :
    (KernelPrefix.forward M c s seed u).val.pids=u.pids := by cases seed <;> rfl
theorem backward_memory (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (u : State α) :
    (KernelPrefix.backward M c s seed u).val.mem=u.mem := rfl
theorem backward_pids (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (u : State α) :
    (KernelPrefix.backward M c s seed u).val.pids=u.pids := rfl

theorem forward_registers (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool)
    (p : Program c) (u : State α) (hh : u.pids 0=p.head) (hk : u.pids 1=p.key) (hv : u.pids 2=p.value) :
    let v := (KernelPrefix.forward M c s seed u).val
    v.regs .ptr [64,16] "S"=some (ptr "S" (fun e j => address c p 0 e.val j.val)) ∧
    v.regs .ptr [64,16] "O"=some (ptr "O" (fun e j => address c p 1 e.val j.val)) ∧
    v.regs .ptr [] "D"=some (fun _ : TileIndex [] => (("D" : RegionName),gateAddress c p 0)) ∧
    v.regs .real [64,16] "acc"=some (initialCarry M c s seed p u) := by
  cases seed <;>
    simp [KernelPrefix.forward,initialCarry,raw,ptr,address,layout,launch,gateAddress,seedAddress,
      TritonBenchG.ChunkGateRecurrence.RealProgram.address,hh,hk,hv,State.setReg,
      bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
      integralFloorDiv,integralMod]
  all_goals repeat' constructor
  all_goals funext i
  all_goals simp [FP.Structural.bop,KernelProgram.ptr,initialCarry,KernelProgram.raw,seedAddress,
    FP.Structural.toFloat,FP.Structural.ofFloat,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex]

#axiomsClean forward_pids
#axiomsClean backward_memory
#axiomsClean forward_registers
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelInitial
