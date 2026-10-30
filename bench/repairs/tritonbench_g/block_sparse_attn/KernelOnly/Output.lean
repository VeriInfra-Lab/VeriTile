import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Channel

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

def outputOffset {α : Type} (p : Parameters) (initial : State α) (i : TileIndex [16,16]) :=
  batch p initial*p.ob + head p initial*p.oh + row initial i.1*p.om + initial.pids 2*16+i.2.1.val
def writeMemory {α : Type} (m : RegionName → Nat → Cell α) (offset : Nat) (v : α) :=
  fun (region : RegionName) (address : Nat) =>
    if region="out" ∧ address=offset then Cell.mk .real v else m region address
def outputMemory {α : Type} (p : Parameters) (initial : State (Replicas α)) (values : Matrix α) :=
  (TileShape.allIndices [16,16]).foldl (fun m i =>
    if row initial i.1<p.length then writeMemory m (outputOffset p initial i) (values i) else m) initial.mem

theorem scatter_memory_general {α ι : Type} (off : ι → Nat) (values : ι → α)
    (active : ι → Prop) [DecidablePred active] (current : State α) (lanes : List ι) :
    (lanes.foldl (fun s i => if active i then
      s.write "out" (off i) (.mk .real (values i)) else s) current).mem =
    lanes.foldl (fun m i => if active i then writeMemory m (off i) (values i) else m) current.mem := by
  induction lanes generalizing current with
  | nil => rfl
  | cons i lanes ih =>
    simp only [List.foldl_cons]
    split_ifs <;> exact ih _

theorem scatter_memory {α : Type} (p : Parameters) (initial current : State (Replicas α))
    (values : Matrix α) (lanes : List (TileIndex [16,16])) :
    (lanes.foldl (fun s i => if row initial i.1<p.length then
      s.write "out" (outputOffset p initial i) (.mk .real (values i)) else s) current).mem =
    lanes.foldl (fun m i => if row initial i.1<p.length then
      writeMemory m (outputOffset p initial i) (values i) else m) current.mem := by
  induction lanes generalizing current with
  | nil => rfl
  | cons i lanes ih =>
    simp only [List.foldl_cons]
    split_ifs with active
    · exact ih _
    · exact ih _

def epilogueExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (h : Runtime a p initial carry current) :
    {t : State (Replicas α) // run a.model (epilogue p) current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [epilogue,source,declared,ComputeKernel.surfaceBody,run,FP.Structural.step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Arithmetic.model,FP.Structural.store,
    h.batch,h.head,h.channel,h.rows,h.dimensions,h.length,h.accumulator,
    Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem epilogueMemory {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (h : Runtime a p initial carry current) :
    (epilogueExecution a p initial current carry h).val.mem =
      outputMemory p initial carry.output := by
  simp [epilogueExecution,outputMemory,outputOffset,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  rw [scatter_memory_general]
  simp [h.memory]

#axiomsClean scatter_memory
#axiomsClean epilogueMemory
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
