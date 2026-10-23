import bench.repairs.tritonbench_g.attention_kernel.KernelOnly.Binding
import bench.repairs.tritonbench_g.attention_kernel.Proofs.FamilyNativeRouting

namespace VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-- Independent softmax attention on the caller's original pointers. The
general `actual_read` theorem identifies every valid operand with its Tensor
physicalIndex; no private copies or unconstrained input arrays occur here. -/
def mathematicalTarget (c : Call) (read : Nat → ℝ) (b h n d : Nat) :=
  RealReturned.target (parameters c) (caller c read) b h n d
def realEntry (c : Call) (read : Nat → ℝ) :=
  entry c (fun r off=>some (caller c read r off))
def realDecode (x : Real.R) : ℝ := x.getD 0

theorem kernel_real (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (read : Nat → ℝ) (work : List Program)
    (order : (programs (grid (parameters c))).Perm work) :
    ∃ after,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel)
        (parameters c) (grid (parameters c)) work (realEntry c read)=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,
        ∀ d<(parameters c).dim,
        logical (observe realDecode c read after) c .output b h n d=
          mathematicalTarget c read b h n d) ∧
      RawFrame c (realEntry c read) after ∧
      (∀ a,¬Stored c a → observe realDecode c read after a=read a) ∧
      (∀ r∈reads,∀ b h n d,Valid (parameters c) r b h n d →
        logical (observe realDecode c read after) c r b h n d=logical read c r b h n d) ∧
      Closed (parameters c) (pointers c) := by
  obtain ⟨after,run,values,frame⟩ := complete Real.algebra Real.dotModel c hc (realEntry c read) work order
  refine ⟨after,run,?_,frame,?_,?_,Machine.closed _ _ (Contract.domain _ _ _ hc)⟩
  · intro b hb h hh n hn d hd
    have valid : Valid (parameters c) .output b h n d := by exact ⟨hb,hh,hn,hd⟩
    rw [actual_read _ c hc laws .output b h n d valid,observe_store _ _ _ _ b h n d valid,
      values b hb h hh n hn d hd]
    have hbm := Contract.positive_block _ _ _ hc
    have hbn := ((Contract.domain _ _ _ hc).launch ⟨by omega,by omega,by omega⟩).bn
    have input : Real.Reads (parameters c)
        (start (grid (parameters c)) (n/c.bm,b*(parameters c).heads+h,0) (realEntry c read)) (caller c read) := by
      intro r hr off
      rfl
    have active : Source.active (parameters c)
        (start (grid (parameters c)) (n/c.bm,b*(parameters c).heads+h,0) (realEntry c read))
        (lane (parameters c) hbm n d hd) := by
      change n/c.bm*c.bm+n%c.bm < (parameters c).length
      simpa only [Nat.mul_comm,Nat.div_add_mod] using hn
    have answer := Real.answer_real (parameters c) _ (caller c read) input _ hbn active
    have target := RealReturned.target_at_program (parameters c) (realEntry c read) (caller c read) hbm b h n d hh hd
    simpa [Cell.read,Real.float_roundtrip,realDecode,mathematicalTarget] using
      congrArg realDecode (answer.trans (congrArg some target))
  · exact fun a outside=>observe_frame _ _ _ _ a outside
  · intro r hr b h n d valid
    unfold logical
    exact observe_frame _ _ _ _ _ (read_exterior c hc laws r hr b h n d valid)

def nativeDecode (I : Instructions) (plan : FamilyNative.Layout) (d : Nat) (x : FamilyNative.Expr) : BitVec 16 :=
  ((Native.lower (x (FamilyNative.physicalOwner plan d))).eval I.isa).setWidth 16
def nativeReference (I : Instructions) (plan : FamilyNative.Layout) (c : Call)
    (words : Nat → BitVec 16) (b h n d : Nat) (hbm : 0<c.bm) (hd : d<(parameters c).dim) : BitVec 16 :=
  ((Specification.nativeTarget plan c words (n/c.bm) b h
    (lane (parameters c) hbm n d hd)).eval I.isa).setWidth 16

theorem kernel_native (I : Instructions) (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (plan : NativeRouting.CompilerPlan (parameters c) (Contract.abi _ _ hc.1))
    (words : Nat → BitVec 16) (work : List Program)
    (order : (programs (grid (parameters c))).Perm work) :
    let layout := plan.layout c.scaleWord
    let before := entry c (Specification.nativeReader c words)
    ∃ after,runPrograms (fun _=>FamilyNative.algebra layout) (fun _=>FamilyNative.dotModel layout)
        (parameters c) (grid (parameters c)) work before=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,
        ∀ d (hd : d<(parameters c).dim),
        logical (observe (nativeDecode I layout d) c words after) c .output b h n d=
          nativeReference I layout c words b h n d (Contract.positive_block _ _ _ hc) hd) ∧
      RawFrame c before after ∧
      (∀ d a,¬Stored c a → observe (nativeDecode I layout d) c words after a=words a) ∧
      (∀ owner r, r∈reads → ∀ b h n d,Valid (parameters c) r b h n d →
        logical (observe (nativeDecode I layout owner) c words after) c r b h n d=logical words c r b h n d) ∧
      Closed (parameters c) (pointers c) := by
  dsimp only
  let layout := plan.layout c.scaleWord
  obtain ⟨after,run,values,frame⟩ := complete (FamilyNative.algebra layout) (FamilyNative.dotModel layout)
    c hc (entry c (Specification.nativeReader c words)) work order
  refine ⟨after,run,?_,frame,?_,?_,Machine.closed _ _ (Contract.domain _ _ _ hc)⟩
  · intro b hb h hh n hn d hd
    have valid : Valid (parameters c) .output b h n d := ⟨hb,hh,hn,hd⟩
    rw [actual_read _ c hc laws .output b h n d valid,observe_store _ _ _ _ b h n d valid,
      values b hb h hh n hn d hd]
    simp only [Cell.read_same,Real.float_roundtrip]
    rw [algorithm_at _ _ c (Specification.nativeReader c words) b h n hh]
    rfl
  · exact fun d a outside=>observe_frame _ _ _ _ a outside
  · intro owner r hr b h n d valid
    unfold logical
    exact observe_frame _ _ _ _ _ (read_exterior c hc laws r hr b h n d valid)

theorem empty_kernel {α β : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (c : Call) (hc : InputContract c) (empty : ¬Live (parameters c))
    (before : State α) (work : List Program) (order : (programs (grid (parameters c))).Perm work)
    (decode : α → β) (words : Nat → β) :
    work=[] ∧ runPrograms (fun _=>M) (fun _=>dot) (parameters c) (grid (parameters c)) work before=some before ∧
      ∀ a,observe decode c words before a=words a := by
  have nil : work=[] := List.nil_perm.mp (by simpa [no_grid c hc empty] using order)
  exact ⟨nil,by simp [nil,runPrograms],empty_observe decode c empty words before⟩

#axiomsClean kernel_real
#axiomsClean kernel_native
#axiomsClean empty_kernel
#specNonCircular mathematicalTarget avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
#specNonCircular nativeReference avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
end
end VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
