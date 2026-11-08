import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerPrelude
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.MaxPass
open VeriTile.Triton FP.Structural ControlArithmetic
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod
abbrev Index (p : Parameters) := TileIndex [1,p.width]

def inputOffset (p : Parameters) (s : State α) (k : Nat) (i : Index p) : Nat :=
  (s.pids 0 % p.rows)*p.inputRowStride+i.2.1.val*p.inputColumnStride+
    k*(p.width*p.inputColumnStride)

def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (k : Nat) (i : Index p) : α :=
  M.fp32Load (if i.2.1.val<p.columns-k*p.width then
    (s.mem p.input (inputOffset p s k i)).read .real else M.literal (some .fp32) .real 0)

def blockMaximum {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (k : Nat) : α :=
  M.reduceMax (some .fp32) (shape:=[1,p.width]) ⟨1,by simp⟩ Bool.false
    (fun i => M.unary (some .fp32) (.abs .real) (loaded M p s k i)) (0,PUnit.unit)

def maximum {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : Nat → α
  | 0 => M.literal (some .fp32) .real 0
  | k+1 => M.binary (some .fp32) .real .max (maximum M p s k) (blockMaximum M p s k)

structure Invariant {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (initial : State α) (k : Nat) (current : State α) : Prop where
  memory : current.mem=initial.mem
  pids : current.pids=initial.pids
  pidm : current.regs .int [] "pid_m"=some (fun _=>(initial.pids 0 : Int))
  offsm : current.regs .int [1] "offs_am"=some (fun _=>((initial.pids 0%p.rows : Nat) : Int))
  output : current.regs .ptr [1,p.width] "a_ptrs"=some (fun i=>(p.output,
    (initial.pids 0%p.rows)*p.outputRowStride+i.2.1.val*p.outputColumnStride))
  offsk : current.regs .nat [p.width] "offs_k"=some (fun i=>i.1.val)
  pointers : current.regs .ptr [1,p.width] "fpa_ptrs" =
    some (fun i => (p.input,inputOffset p initial k i))
  maximum : current.regs .real [1] "a_max"=some (fun _=>maximum M p initial k)

private theorem fin_one (i : Fin 1) : i=0 := Subsingleton.elim _ _

theorem initial {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (s : State α) :
    Invariant M p s 0 (prefixExecution M integer8 p s).val := by
  constructor
  all_goals simp [prefixExecution,inputOffset,maximum,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,fin_one]
  all_goals funext i
  all_goals try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,fin_one]
  all_goals try simp only [← Int.natCast_mul, ← Int.natCast_add, tmod_nat, Int.toNat_natCast]
  all_goals try simp only [← Int.natCast_emod, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast, Nat.zero_add]

  all_goals try simp only [Int.mul_one,Int.add_zero,tmod_nat]
  all_goals try rfl

/-- Exactly the source's three max-pass statements, with a signed loop slot
and the actual masked FP32 load. No maximum identity is assumed. -/
def iteration {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (k : Nat) (h : Invariant M p original k current)
    (width : 0<p.width) :
    {t : State α // run (backend M integer8) (maxBody p)
      (current.setReg "k" .int [] (fun _=>(k : Int)))=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [maxBody,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,numeric,numericLt,
    natLt,bop,backend,ComputeDType.eraseDType,Option.bind,TileShape.axisDim,width,
    h.memory,h.offsk,h.pointers,h.maximum,NumericDType.sub]
  rfl

theorem iteration_invariant {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (k : Nat) (h : Invariant M p original k current)
    (width : 0<p.width) :
    Invariant M p original (k+1) (iteration M integer8 p original current k h width).val := by
  constructor
  all_goals simp [iteration,h.memory,h.pids,h.pidm,h.offsm,h.output,h.offsk,h.pointers,h.maximum,inputOffset,maximum,
    blockMaximum,loaded,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    show (0.0:ℝ)=0 by norm_num,ofFloat,toFloat,fin_one]
  all_goals try (funext i; simp [Nat.add_mul,Nat.add_assoc,ofFloat,toFloat,fin_one,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex])

  all_goals try rfl

/-- The full first pass, retaining every signed source loop counter and the
backend's ordered maximum recurrence. Zero and multiple iterations are included. -/
theorem loop {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (start stop : Nat)
    (ordered : start≤stop) (h : Invariant M p original start current) (width : 0<p.width) :
    ∃ t, rangeInt (backend M integer8) "k" (start : Int) (stop : Int) 1 (maxBody p) current=some t ∧
      Invariant M p original stop t := by
  have aux : ∀ remaining start (current : State α), stop-start=remaining → start≤stop →
      Invariant M p original start current →
      ∃ t, rangeInt (backend M integer8) "k" (start : Int) (stop : Int) 1 (maxBody p) current=some t ∧
        Invariant M p original stop t := by
    intro remaining
    induction remaining using Nat.strong_induction_on with
    | h remaining ih =>
      intro start current distance ordered inv
      by_cases active : start<stop
      · let next:=iteration M integer8 p original current start inv width
        have nextInv:=iteration_invariant M integer8 p original current start inv width
        obtain ⟨final,ran,valid⟩:=ih (stop-(start+1)) (by omega) (start+1) next.val rfl (by omega) nextInv
        refine ⟨final,?_,valid⟩
        rw [rangeInt,if_neg (by decide),if_pos (by exact_mod_cast active),next.property]
        change rangeInt (backend M integer8) "k" ((start+1 : Nat) : Int) (stop : Int) 1 (maxBody p) next.val=some final
        exact ran
      · have endOfLoop : start=stop:=by omega
        subst start
        refine ⟨current,?_,inv⟩
        rw [rangeInt,if_neg (by decide),if_neg (by omega)]
  exact aux _ _ _ rfl ordered h

#axiomsClean initial
#axiomsClean iteration
#axiomsClean iteration_invariant
#axiomsClean loop
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.MaxPass
