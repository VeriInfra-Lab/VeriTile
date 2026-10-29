import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Store
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Native

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

def normExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State α)
    (h : Invariant a Bool.true p initial (steps p) current) :
    {t : State α // run a.model ((epilogue p).take 2) current=some t} := by
  have hep : (epilogue p).take 2=(epilogue {p with byte:=Bool.false}).take 2 := by
    rcases p with ⟨M,N,K,fp8,eps⟩
    cases fp8 <;> rfl
  refine ⟨?_,?_⟩
  swap
  rw [hep]
  simp [epilogue,source,Source.kernel,Source.half,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,numeric,
    FP.Structural.bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
    Region.cast,Option.bind,h.sum,h.k,Arithmetic.model]
  rfl

theorem normPost {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State α)
    (h : Invariant a Bool.true p initial (steps p) current) :
    let next := (normExecution a p initial current h).val
    StoreContext p initial next ∧
      next.regs .real [16,16] "acc1"=some (accumulation a Bool.true p initial "W1" (steps p)) ∧
      next.regs .real [16,16] "acc2"=some (accumulation a Bool.true p initial "W3" (steps p)) ∧
      next.regs .real [16] "a_norm"=some (fun i => normalization a p initial i.1) := by
  refine ⟨?_,?_,?_,?_⟩
  · constructor <;> simp [normExecution,h.memory,h.m,h.n,h.pidm,h.pidn,h.outm,h.outn]
  · simp [normExecution,h.acc1]
  · simp [normExecution,h.acc2]
  · simp [normExecution,normalization,Arithmetic.model,reduction,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.insertAxisIndex,TileShape.axisDim,TileShape.eraseAxis]
    funext i
    rcases i with ⟨i,⟨⟩⟩
    rfl

def nativeValues {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (initial : State α) (i : TileIndex [16,16]) : α :=
  gateValue a g (accumulation a Bool.true p initial "W1" (steps p) i)
    (accumulation a Bool.true p initial "W3" (steps p) i) (normalization a p initial i.1)
def nativeOutput {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (initial : State α) (i : TileIndex [16,16]) : α :=
  a.cast none .real .fp16 (nativeValues a g p initial i)

def terminalExecution {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (initial current : State α)
    (h : Invariant a Bool.true p initial (steps p) current) :
    {t : State α // (do
      let norm ← run a.model ((epilogue p).take 2) current
      let gate ← gateExecution a g norm
      run a.model ((epilogue p).drop 5) gate)=some t ∧
        Post p initial t (nativeOutput a g p initial)} := by
  let norm := normExecution a p initial current h
  have hn : StoreContext p initial norm.val ∧
      norm.val.regs .real [16,16] "acc1"=some (accumulation a Bool.true p initial "W1" (steps p)) ∧
      norm.val.regs .real [16,16] "acc2"=some (accumulation a Bool.true p initial "W3" (steps p)) ∧
      norm.val.regs .real [16] "a_norm"=some (fun i => normalization a p initial i.1) :=
    normPost a p initial current h
  let gate := norm.val.setReg "accumulator" .real [16,16] (nativeValues a g p initial)
  have hg : gateExecution a g norm.val=some gate := by
    simp [gateExecution,hn.2.1,hn.2.2.1,hn.2.2.2,gate,nativeValues]
    rfl
  have hc : StoreContext p initial gate := by
    constructor <;> simp [gate,hn.1.memory,hn.1.m,hn.1.n,hn.1.pidm,hn.1.pidn,hn.1.outm,hn.1.outn]
  have hv : gate.regs .real [16,16] "accumulator"=some (nativeValues a g p initial) := by simp [gate]
  let finish := storeExecution a p initial gate (nativeValues a g p initial) hc hv
  refine ⟨finish.val,?_,storePost a p initial gate (nativeValues a g p initial) hc hv⟩
  rw [norm.property]
  change (gateExecution a g norm.val).bind (run a.model ((epilogue p).drop 5))=some finish.val
  rw [hg]
  exact finish.property

#axiomsClean normPost
#axiomsClean terminalExecution
#specNonCircular nativeOutput avoiding [source,body,prelude,compiledBody,run,nativeExecution,gateExecution]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
