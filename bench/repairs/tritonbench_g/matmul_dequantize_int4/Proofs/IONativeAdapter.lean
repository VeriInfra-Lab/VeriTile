import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IONativeGraph

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONativeAdapter
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4 MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word)
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open IONativeGraph
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev scalar : Algebra Word := FPSoftware.scalar FPCompiled.unused

theorem model_eq (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (op : Opcode) :
    model actual op=FPCompiled.backend scalar (ISA.dot (mma op)) := by
  have hm : actual.mma op=mma op := funext fun a=>funext fun b=>funext fun acc=>ports.mma op a b acc
  simp only [model,Native.scalar_eq actual mma ports,hm]

theorem fma_eq (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) : Native.fmaAssign actual=FPFused.fmaAssign := by
  funext p s
  simp only [Native.fmaAssign,FPFused.fmaAssign,ports.fma]

theorem block_run (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (op : Opcode) (p : Parameters)
    (initial current : State Word) (k : Nat)
    (h : FPFused.Invariant scalar (ISA.dot (mma op)) p initial k current) :
    runSchedule (blockJobs actual op p k) current=
      some (FPFused.bodyExecution scalar (ISA.dot (mma op)) p initial current k h).val := by
  let start := FPFused.prefixExecution scalar (ISA.dot (mma op)) p initial current k h
  have facts := FPFused.prefixFacts scalar (ISA.dot (mma op)) p initial current k h
  have middle := fma_step_eq actual p start.val _ _ _ facts.b facts.scales facts.zeros
    (fun j=>by
      change -15≤weightNibble p initial k j ∧ weightNibble p initial k j≤15
      have hv:=weightNibble_bounds p initial k j
      omega)
  rw [fma_eq actual mma ports] at middle
  have hm := FPFused.fmaAssign_correct scalar (ISA.dot (mma op)) p initial start.val k facts
  have hs := (FPFused.suffixExecution scalar (ISA.dot (mma op)) p initial start.val k facts).property
  simp only [blockJobs,runSchedule,Invocation.run,held_initial,counter_initial]
  simp only [held,FP.Structural.exec,ComputeKernel.surfaceBody,prefixKernel,fmaKernel,suffixKernel,
    model_eq actual mma ports,start.property,bind,Option.bind,FP.Structural.run,
    middle,hm,hs,FPFused.bodyExecution]
  rfl

theorem loop_run (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (op : Opcode) (p : Parameters)
    (initial current : State Word) (k count : Nat)
    (h : FPFused.Invariant scalar (ISA.dot (mma op)) p initial k current) :
    runSchedule (loopJobs actual op p k count) current=
      FPFused.loop scalar (ISA.dot (mma op)) p k count current := by
  induction count generalizing k current with
  | zero=>rfl
  | succ count ih=>
    rw [loopJobs,runSchedule_append,block_run actual mma ports op p initial current k h]
    simp only [Option.bind_some,FPFused.loop,
      (FPFused.bodyExecution scalar (ISA.dot (mma op)) p initial current k h).property]
    exact ih _ _ (FPFused.bodyInvariant scalar (ISA.dot (mma op)) p initial current k h)

/-- Every raw initial state reaches exactly the original native compiled
executor under the same four local primitive obligations. No unfused/fused
floating identity is asserted. -/
theorem program_run (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (op : Opcode) (p : Parameters)
    (pid : Nat) (s : State Word) :
    runSchedule (programJobs actual op p pid) s=
      Native.exec actual op p (GridExecution.atProgram s pid) := by
  rw [Native.exec_transport actual mma ports]
  let initial := GridExecution.atProgram s pid
  let start := FPFused.preludeExecution scalar (ISA.dot (mma op)) p initial
  have invariant := FPFused.preludeInvariant scalar (ISA.dot (mma op)) p initial
  have hstart : run (FPCompiled.backend scalar (ISA.dot (mma op))) (FPCompiled.prelude p)
      (GridExecution.atProgram s pid)=some start.val := start.property
  have numk : start.val.regs .nat [] "num_pid_k"=some (fun _=>steps p) := invariant.numk
  unfold programJobs
  rw [runSchedule_append,runSchedule_append]
  simp only [runSchedule_singleton,Invocation.run,IOExecution.initial_eq,held_initial]
  simp only [IOExecution.launch,held,FP.Structural.exec,preludeKernel,epilogueKernel,ComputeKernel.surfaceBody,
    model_eq actual mma ports,hstart,Option.bind_some]
  rw [loop_run actual mma ports op p initial start.val 0 (steps p) invariant]
  simp only [FPFused.exec,hstart,Option.bind_some,bind,Option.bind,numk]

theorem grid_run (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (c : RawKernel.Layout) (i : Fin 17) (s : State Word) :
    runSchedule (jobs actual c i) s=Native.grid actual c.call i s := by
  have lists (pids : List Nat) (s : State Word) :
      runSchedule (pids.flatMap (programJobs actual (tunings i).opcode (params c.call i))) s=
        GridExecution.sequence (fun pid t=>Native.exec actual (tunings i).opcode (params c.call i)
          (GridExecution.atProgram t pid)) pids s := by
    induction pids generalizing s with
    | nil=>rfl
    | cons pid rest ih=>
      rw [List.flatMap_cons,runSchedule_append,program_run actual mma ports]
      simp only [GridExecution.sequence]
      cases h : Native.exec actual (tunings i).opcode (params c.call i) (GridExecution.atProgram s pid)
      · rfl
      · simpa only [Option.bind_some] using ih _
  exact lists _ s

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IONativeAdapter
