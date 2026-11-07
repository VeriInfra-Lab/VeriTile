import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerPrelude
import VeriTile.Triton.Float.SignedCast
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.QuantizePass
open VeriTile.Triton FP.Structural ControlArithmetic
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod FP.Structural.eval_intToSigned
abbrev Index (p : Parameters) := TileIndex [1,p.width]

def inputOffset (p : Parameters) (s : State α) (k : Nat) (i : Index p) : Nat :=
  (s.pids 0%p.rows)*p.inputRowStride+i.2.1.val*p.inputColumnStride+k*(p.width*p.inputColumnStride)
def outputOffset (p : Parameters) (s : State α) (k : Nat) (i : Index p) : Nat :=
  (s.pids 0%p.rows)*p.outputRowStride+i.2.1.val*p.outputColumnStride+k*(p.width*p.outputColumnStride)

/-- Source registers needed by the write pass. Input memory remains the
actual current memory, so existence does not assume buffer separation. -/
structure Invariant (p : Parameters) (original : State α) (scale : α) (k : Nat)
    (current : State α) : Prop where
  pids : current.pids=original.pids
  pidm : current.regs .int [] "pid_m"=some (fun _=>(original.pids 0 : Int))
  offsk : current.regs .nat [p.width] "offs_k"=some (fun i=>i.1.val)
  input : current.regs .ptr [1,p.width] "fpa_ptrs"=some (fun i=>(p.input,inputOffset p original k i))
  output : current.regs .ptr [1,p.width] "a_ptrs"=some (fun i=>(p.output,outputOffset p original k i))
  scale : current.regs .real [1] "a_scale"=some (fun _=>scale)

/-- Actual masked FP32 load, FP32 divide, INT8 conversion and store, followed
by both source pointer increments. No finite-input or nonzero-scale guard. -/
def iteration {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (scale : α) (k : Nat)
    (h : Invariant p original scale k current) :
    {t : State α // run (backend M integer8) (quantizeBody p)
      (current.setReg "k" .int [] (fun _=>(k : Int)))=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [quantizeBody,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,numericLt,
    natLt,bop,backend,ComputeDType.eraseDType,Option.bind,h.input,h.output,h.offsk,h.scale,
    NumericDType.sub,ofFloat,toFloat,fold_regs,fold_pids]
  rfl

theorem iteration_invariant {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (scale : α) (k : Nat)
    (h : Invariant p original scale k current) :
    Invariant p original scale (k+1) (iteration M integer8 p original current scale k h).val := by
  constructor
  all_goals simp [iteration,h.pids,h.pidm,h.offsk,h.input,h.output,h.scale,
    inputOffset,outputOffset,fold_regs,fold_pids,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try (funext i; simp [Nat.add_mul,Nat.add_assoc,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex])
  all_goals try rfl

/-- Every source write-pass iteration succeeds on the actual predecessor memory,
including zero iterations and aliased generic memory. -/
theorem loop {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (scale : α) (start stop : Nat)
    (ordered : start≤stop) (h : Invariant p original scale start current) :
    ∃ t, rangeInt (backend M integer8) "k" (start : Int) (stop : Int) 1 (quantizeBody p) current=some t ∧
      Invariant p original scale stop t := by
  have aux : ∀ remaining start (current : State α), stop-start=remaining → start≤stop →
      Invariant p original scale start current →
      ∃ t, rangeInt (backend M integer8) "k" (start : Int) (stop : Int) 1 (quantizeBody p) current=some t ∧
        Invariant p original scale stop t := by
    intro remaining
    induction remaining using Nat.strong_induction_on with
    | h remaining ih =>
      intro start current distance ordered inv
      by_cases active : start<stop
      · let next:=iteration M integer8 p original current scale start inv
        have nextInv:=iteration_invariant M integer8 p original current scale start inv
        obtain ⟨final,ran,valid⟩:=ih (stop-(start+1)) (by omega) (start+1) next.val rfl (by omega) nextInv
        refine ⟨final,?_,valid⟩
        rw [rangeInt,if_neg (by decide),if_pos (by exact_mod_cast active),next.property]
        change rangeInt (backend M integer8) "k" ((start+1 : Nat) : Int) (stop : Int) 1 (quantizeBody p) next.val=some final
        exact ran
      · have endOfLoop : start=stop:=by omega
        subst start
        refine ⟨current,?_,inv⟩
        rw [rangeInt,if_neg (by decide),if_neg (by omega)]
  exact aux _ _ _ rfl ordered h

#axiomsClean iteration
#axiomsClean iteration_invariant
#axiomsClean loop
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.QuantizePass
