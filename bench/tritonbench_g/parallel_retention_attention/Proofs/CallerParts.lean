import bench.tritonbench_g.parallel_retention_attention.Proofs.Support.ShellSyntax
import bench.tritonbench_g.parallel_retention_attention.Proofs.ForwardPrefix
import bench.tritonbench_g.parallel_retention_attention.Proofs.QueryTail
import bench.tritonbench_g.parallel_retention_attention.Proofs.BackwardTail

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.Caller
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (BackwardConfig DotPrimitive withDot)
noncomputable section

def kernel (c : BackwardConfig) (dq : RegionName) := FPBackward.shellSource c dq
def prologue (c : BackwardConfig) (dq : RegionName) := (kernel c dq).surfaceBody.take 23
def queryArguments (c : BackwardConfig) (dq : RegionName) := ((kernel c dq).surfaceBody.drop 23).take 28
def queryTail (c : BackwardConfig) (dq : RegionName) := ((kernel c dq).surfaceBody.drop 51).take 17
def keyArguments (c : BackwardConfig) (dq : RegionName) := ((kernel c dq).surfaceBody.drop 69).take 28
def keyTail (c : BackwardConfig) (dq : RegionName) := (kernel c dq).surfaceBody.drop 97

theorem prologue_eq (c : BackwardConfig) (dq : RegionName) :
    prologue c dq=FPStages.argumentPrefix c.toForwardConfig++FPStages.decodedPrefix c.toForwardConfig:=rfl
theorem query_tail_eq (c : BackwardConfig) (dq : RegionName) :
    queryTail c dq=FPQuery.tailCode c dq:=rfl
theorem key_tail_eq (c : BackwardConfig) (dq : RegionName) :
    keyTail c dq=FPStages.backwardTail c:=rfl
theorem source_split (c : BackwardConfig) (dq : RegionName) :
    (kernel c dq).surfaceBody=prologue c dq++queryArguments c dq++queryTail c dq++
      [.effectMarker "tl.debug_barrier"]++keyArguments c dq++keyTail c dq:=rfl

structure Registers (c : BackwardConfig) (s : State α) : Prop where
  head : s.regs .int [] "i_bh"=some (fun _=>(c.i_bh : Int))
  chunk : s.regs .int [] "i_c"=some (fun _=>(c.i_c : Int))
  key : s.regs .int [] "i_k"=some (fun _=>(c.i_k : Int))
  value : s.regs .int [] "i_v"=some (fun _=>(c.i_v : Int))
  decay : s.regs .int [] "i_h"=some (fun _=>(c.i_h : Int))

structure Indices (c : BackwardConfig) (s : State α) : Prop where
  head : s.pids 2=c.i_bh
  chunk : s.pids 1=c.i_c
  key : s.pids 0/((c.V+c.BV-1)/c.BV)=c.i_k
  value : s.pids 0%((c.V+c.BV-1)/c.BV)=c.i_v
  decay : s.pids 2%c.H=c.i_h

theorem prologue_run {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (dq : RegionName) (s : State α) (indices : Indices c s) :
    ∃ t,run M (prologue c dq) s=some t ∧ FPStages.DecodedReady c.toForwardConfig s t ∧ Registers c t := by
  obtain ⟨captured,ran,args⟩:=FPStages.arguments_run M c.toForwardConfig s
  obtain ⟨decoded,finished,ready⟩:=FPStages.decoded_run M c.toForwardConfig s captured args
  refine ⟨decoded,?_,ready,?_⟩
  · rw [prologue_eq,run_append,ran,Option.bind_some,finished]
  · constructor
    · simpa only [indices.head] using ready.head
    · simpa only [indices.chunk] using ready.chunk
    · simpa only [indices.key] using ready.keyBlock
    · simpa only [indices.value] using ready.valueBlock
    · simpa only [indices.decay] using ready.decayHead

theorem registers_after_query {c : BackwardConfig} {s t : State α}
    (r : Registers c s) (frame : FPQuery.CallerFrame s t) : Registers c t := by
  constructor
  · exact frame.head.trans r.head
  · exact frame.chunk.trans r.chunk
  · exact frame.key.trans r.key
  · exact frame.value.trans r.value
  · exact frame.decay.trans r.decay

#axiomsClean prologue_run
#axiomsClean source_split
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.Caller
