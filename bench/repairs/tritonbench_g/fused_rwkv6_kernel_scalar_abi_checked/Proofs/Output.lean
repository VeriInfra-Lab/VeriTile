import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Loop

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite FusedRwkv6Kernel
open AttentionForwardTriton.Scatter (masked_scatter_memory masked_scatter_regs)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The exit code reads only these registers. In particular none of the five
advanced pointers, the loop index, the scalar, or the loaded step operands is
a prerequisite for the final-state store. -/
structure FinalContext (M : Algebra α) (c : Shape) (initial : State α) (scale : α)
    (current : State α) : Prop where
  mem : current.mem=loopMemory M c initial scale c.T
  iv : current.regs .nat [] "i_v"=some (fun _ => initial.pids 0)
  ik : current.regs .nat [] "i_k"=some (fun _ => initial.pids 1)
  iz : current.regs .nat [] "i_bh"=some (fun _ => initial.pids 2)
  kvm : current.regs .bool [c.BV,c.BK] "mask_kv"=
    some (fun i => valueMask c initial i.1 && keyMask c initial i.2.1)
  state : current.regs .real [c.BV,c.BK] "b_h"=some (carry M c initial c.T)

def Context.final (M : Algebra α) (c : Shape) (initial current : State α) (scale : α)
    (ctx : Context M c initial scale c.T current) : FinalContext M c initial scale current :=
  ⟨ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.kvm,ctx.state⟩

def terminalAddress (c : Shape) (s : State α) (i : TileIndex [c.BV,c.BK]) : Nat :=
  s.pids 2*c.K*c.V+(s.pids 1*c.BK+i.2.1.val)*c.V+(s.pids 0*c.BV+i.1.val)
def terminalValue (M : Algebra α) (c : Shape) (s : State α)
    (i : TileIndex [c.BV,c.BK]) : Cell α :=
  .mk .real ((ops M).storeCast (carry M c s c.T i))
def finalMemory (M : Algebra α) (c : Shape) (s : State α) (scale : α) :
    RegionName → Nat → Cell α :=
  if c.final then (TileShape.allIndices [c.BV,c.BK]).foldl (fun m i =>
    if valueMask c s i.1 && keyMask c s i.2.1 then
      fun r b => if r="ht" ∧ b=terminalAddress c s i then terminalValue M c s i else m r b
    else m) (loopMemory M c s scale c.T)
  else loopMemory M c s scale c.T

def finishExec (M : Algebra α) (c : Shape) (initial current : State α) (scale : α)
    (ctx : FinalContext M c initial scale current) :
    {t : State α // run (engine M) (parts c).after current=some t} := by
  by_cases hf : c.final=Bool.true
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,ComputeKernel.surfaceBody,Triton.Source.intRange?,
      Option.map,Option.getD,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralMod,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,hf,
      ctx.mem,ctx.iv,ctx.ik,ctx.iz,ctx.kvm,ctx.state,FP.Structural.store]
    rfl

theorem finish_memory (M : Algebra α) (c : Shape) (initial current : State α) (scale : α)
    (ctx : FinalContext M c initial scale current) :
    (finishExec M c initial current scale ctx).val.mem=finalMemory M c initial scale := by
  by_cases hf : c.final=Bool.true
  all_goals simp [finishExec,hf,finalMemory,ctx.mem,terminalAddress,terminalValue,ops,
    masked_scatter_memory,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  apply congrArg (fun step => List.foldl step (loopMemory M c initial scale c.T)
    (TileShape.allIndices [c.BV,c.BK]))
  funext mem i
  split_ifs
  · funext r b
    congr 2
    apply propext
    simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    simp [Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  · rfl

/-- Formal dead-register noninterference: completely arbitrary values in the
advanced pointers at exit produce the same final memory. -/
theorem finish_noninterference (M : Algebra α) (c : Shape) (initial s t : State α) (scale : α)
    (hs : FinalContext M c initial scale s) (ht : FinalContext M c initial scale t) :
    (finishExec M c initial s scale hs).val.mem=
      (finishExec M c initial t scale ht).val.mem := by
  rw [finish_memory,finish_memory]

/-- Entire shared source executes to the independent input recurrence's stores.
This includes all iterations, optional state seed/writeback, both directions,
and the zero-trip loop; the exterior cells are still the initial cells. -/
theorem source_memory (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (hs : s.regs .real [] "scale"=some (fun _ => scale)) :
    ∃ t, FP.Structural.exec (engine M) (kernel c) s=some t ∧
      t.mem=finalMemory M c s scale := by
  let start:=entry M c s
  obtain ⟨mid,hm,hc⟩:=loopExec M c s start.val scale 0 (by omega) (entry_context M c s scale hs)
  let done:=finishExec M c s mid scale (hc.final M c s mid scale)
  refine ⟨done.val,?_,finish_memory M c s mid scale _⟩
  have hl : FP.Structural.step (engine M) (.forRangeInt "_" (.constInt 0) (.constInt (c.T : Int)) (.constNat 1) (parts c).body)
      start.val=some mid := by simpa only [FP.Structural.step,evalOp_unfold,evalOffsets,Option.bind_some] using hm
  change run (engine M) (kernel c).surfaceBody s=some done.val
  rw [source_body,List.append_assoc,run_append,start.property]
  simp only [Option.bind_some,List.singleton_append,run,hl,bind,done.property]

#axiomsClean finish_noninterference
#axiomsClean source_memory
#specNonCircular finalMemory avoiding [run,FP.Structural.exec,entry,bodyExec,loopExec,finishExec]
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
