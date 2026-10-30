import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelForwardH
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelOutputMemory
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelForward
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (Parameters PID Matrix)
abbrev Memory := KernelBackward.Memory
open KernelReal (parameters)
open KernelOutput (shape sharedDot withDot_eq)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (scatter hActive hOffset fOffset foldRange_succ foldRange_zero)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
namespace H
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
  (initValue update stored advance accumulated written finalStored active historyAddress stateAddress feature column keyLoad valueLoad gatedValue)
end H
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def state (c : Shape) (pid : PID) (m : Memory α) : State α :=
  VeriTile.Bench.Repairs.ChunkLinearAttn.launchState .forwardState (parameters c) pid m

def values (M : Algebra α) (D : DotPrimitive α) (c : Shape) (pid : PID)
    (input : Memory α) (seed : Bool) : Nat → Matrix α c.BK c.BV
  | 0=>H.initValue M (shape c) seed (state c pid input)
  | n+1=>H.update M (sharedDot D) (shape c) (state c pid input) n (values M D c pid input seed n)

def writeHistory (M : Algebra α) (c : Shape) (pid : PID) (n : Nat)
    (carry : Matrix α c.BK c.BV) (m : Memory α) : Memory α :=
  scatter (TileShape.allIndices [c.BK,c.BV]) "h" (hActive (parameters c) pid)
    (hOffset (parameters c) pid n) (fun i=>.mk .real (M.cast (some .fp32) .real .real (carry i))) m

def writeFinal (M : Algebra α) (c : Shape) (pid : PID) (final : Bool)
    (carry : Matrix α c.BK c.BV) (m : Memory α) : Memory α :=
  if final then scatter (TileShape.allIndices [c.BK,c.BV]) "ht" (hActive (parameters c) pid)
    (fOffset (parameters c) pid) (fun i=>.mk .real (M.cast (some .fp32) .real .real (carry i))) m else m

def memory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (pid : PID)
    (input current : Memory α) (seed final : Bool) : Memory α :=
  writeFinal M c pid final (values M D c pid input seed c.chunks)
    (foldRange (fun n mem=>writeHistory M c pid n (values M D c pid input seed n) mem) 0 c.chunks 1 current)

theorem stored_memory (M : Algebra α) (c : Shape) (pid : PID) (input : Memory α)
    (s : State α) (n : Nat) (carry : Matrix α c.BK c.BV) :
    (H.stored M (shape c) (state c pid input) s n carry).mem=writeHistory M c pid n carry s.mem := by
  simp [H.stored,writeHistory,scatter,masked_scatter_memory,H.active,H.historyAddress,H.feature,H.column,
    hActive,hOffset,state,shape,parameters,VeriTile.Bench.Repairs.ChunkLinearAttn.launchState]
  rfl

theorem accumulated_pair (M : Algebra α) (D : DotPrimitive α) (c : Shape) (pid : PID)
    (input : Memory α) (seed : Bool) (n : Nat) :
    let a:=foldRange (H.advance M (sharedDot D) (shape c) (state c pid input)) 0 n 1
      ⟨H.initValue M (shape c) seed (state c pid input),state c pid input⟩
    a.1=values M D c pid input seed n ∧
      a.2.mem=foldRange (fun n mem=>writeHistory M c pid n (values M D c pid input seed n) mem) 0 n 1 input := by
  induction n with
  | zero=>simp [foldRange_zero,values,state,VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,
      VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState]
  | succ n ih=>
    dsimp only at ih ⊢
    rw [foldRange_succ,foldRange_succ]
    dsimp only [H.advance]
    constructor
    · rw [ih.1]
      rfl
    · rw [stored_memory,ih.1,ih.2]

theorem written_memory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (pid : PID)
    (input : Memory α) (seed final : Bool) :
    (H.written M (sharedDot D) (shape c) seed final (state c pid input)).mem=
      memory M D c pid input input seed final := by
  have ha:=accumulated_pair M D c pid input seed c.chunks
  change (H.accumulated M (sharedDot D) (shape c) seed (state c pid input)).1=_ ∧
    (H.accumulated M (sharedDot D) (shape c) seed (state c pid input)).2.mem=_ at ha
  cases final
  · exact ha.2
  · change (H.finalStored M (shape c) (state c pid input)
      (H.accumulated M (sharedDot D) (shape c) seed (state c pid input)).2
      (H.accumulated M (sharedDot D) (shape c) seed (state c pid input)).1).mem=_
    simp only [H.finalStored,masked_scatter_memory]
    rw [ha.1,ha.2]
    rfl

theorem source (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (pid : PID) (m : Memory α) :
    ∃ t,FP.Structural.exec (engine (withDot M D)) (kernel c seed final) (state c pid m)=some t ∧
      t.mem=memory M D c pid m m seed final := by
  obtain ⟨t,ht,hm⟩:=program M (sharedDot D) c seed final (state c pid m)
  rw [withDot_eq] at ht
  exact ⟨t,ht,hm.trans (written_memory M D c pid m seed final)⟩

#axiomsClean source
#axiomsClean accumulated_pair
#specNonCircular values avoiding [run,FP.Structural.exec,program,source]
#specNonCircular memory avoiding [run,FP.Structural.exec,program,source]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelForward
