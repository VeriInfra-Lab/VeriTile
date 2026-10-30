import bench.repairs.tritonbench_g.attention_fwd_triton2.KernelOnly.Binding

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def mathematicalTarget (c : Call) (read : Nat→ℝ) (b h n d : Nat) : ℝ :=
  RealReturned.target (parameters c) (caller c read) b h n d
def realEntry (c : Call) (read : Nat→ℝ) := entry (fun r off=>some (caller c read r off))
def realDecode (x : Real.R) : ℝ := x.getD 0

theorem kernel_real (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (read : Nat→ℝ) (work : List Program) (order : (programs (grid (parameters c))).Perm work) :
    ∃ after,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel)
        (parameters c) (grid (parameters c)) work (realEntry c read)=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,∀ d<96,
        logical (observe realDecode c read after) c .output b h n d=mathematicalTarget c read b h n d) ∧
      RawFrame c (realEntry c read) after ∧
      (∀ a,¬Stored c a → observe realDecode c read after a=read a) ∧
      (∀ r∈reads,∀ b h n d,Valid (parameters c) r b h n d →
        logical (observe realDecode c read after) c r b h n d=logical read c r b h n d) ∧
      Access.Safe (parameters c) ∧ Closed (parameters c) (pointers c) := by
  obtain ⟨after,run,values,frame⟩:=complete Real.algebra Real.dotModel c hc laws (realEntry c read) work order
  refine ⟨after,run,?_,frame,?_,?_,Access.safe _ (domain c hc laws).canonical,Machine.closed _ _ (domain c hc laws)⟩
  · intro b hb h hh n hn d hd
    have valid : Valid (parameters c) .output b h n d:=⟨hb,hh,hn,hd⟩
    rw [actual_read _ c hc laws .output b h n d valid,observe_store _ _ _ _ b h n d valid,
      values b hb h hh n hn d hd]
    have input : Real.Reads (parameters c)
        (start (grid (parameters c)) (n/128,b*(parameters c).heads+h,0) (realEntry c read)) (caller c read) := by
      intro r hr off
      simp only [parameters,List.mem_cons,List.not_mem_nil,or_false] at hr
      rcases hr with rfl|rfl|rfl|rfl|rfl <;> simp [start,realEntry,entry,tag] <;> rfl
    have active : Source.active (parameters c)
        (start (grid (parameters c)) (n/128,b*(parameters c).heads+h,0) (realEntry c read)) (lane n d hd) := by
      change n/128*128+n%128<(parameters c).length ∧ d<96
      exact ⟨by omega,hd⟩
    have answer:=Real.answer_real (parameters c) _ (caller c read) input _ active
    have target:=RealReturned.target_at_program (parameters c) (realEntry c read) (caller c read) b h n d hh hd
    simpa [Cell.read,realDecode,mathematicalTarget] using congrArg realDecode (answer.trans (congrArg some target))
  · exact fun a outside=>observe_frame _ _ _ _ a outside
  · intro r hr b h n d valid
    unfold logical
    exact observe_frame _ _ _ _ _ (read_exterior c hc laws r hr b h n d valid)

/-- One fixed scalar/conversion/ISA model; raw input words are identity leaves.
Only single coordinates of fixed K8 TF32 and K16 FP16 instructions are ports. -/
structure Instructions where
  literal32 : ℝ→BitVec 32
  add32 : BitVec 32→BitVec 32→BitVec 32
  sub32 : BitVec 32→BitVec 32→BitVec 32
  mul32 : BitVec 32→BitVec 32→BitVec 32
  maxNum32 : BitVec 32→BitVec 32→BitVec 32
  divFull32 : BitVec 32→BitVec 32→BitVec 32
  exp2ApproxFTZ : BitVec 32→BitVec 32
  narrow16 : BitVec 32→BitVec 32
  widen16 : BitVec 32→BitVec 32
  narrowBF16 : BitVec 32→BitVec 32
  fma32 : BitVec 32→BitVec 32→BitVec 32→BitVec 32
  mmaTF32 : Fin 16→Fin 8→(Fin 8→BitVec 32)→(Fin 8→BitVec 32)→BitVec 32→BitVec 32
  mmaHalf : Fin 16→Fin 8→(Fin 16→BitVec 32)→(Fin 16→BitVec 32)→BitVec 32→BitVec 32
def Instructions.isa (I : Instructions) : Native.ISA (BitVec 32) where
  bits:=id
  literal32:=I.literal32; add32:=I.add32; sub32:=I.sub32; mul32:=I.mul32
  maxNum32:=I.maxNum32; divFull32:=I.divFull32; exp2ApproxFTZ:=I.exp2ApproxFTZ
  narrow16:=I.narrow16; widen16:=I.widen16; narrowBF16:=I.narrowBF16; fma32:=I.fma32
  mmaTF32:=I.mmaTF32; mmaHalf:=I.mmaHalf

/-- Word-address observation uses a zero-extended BF16 output so that the
exterior reader continues to preserve every original FP32 input word. -/
def nativeDecode (I : Instructions) (d : Nat) (x : Native.Expr) : BitVec 32 :=
  (((Native.lower (x (Native.physicalOwner d))).eval I.isa).setWidth 16).zeroExtend 32
def nativeReference (I : Instructions) (c : Call) (words : Nat→BitVec 32)
    (b h n d : Nat) (hd : d<96) : BitVec 16 :=
  ((Specification.nativeTarget c words (n/128) b h (lane n d hd)).eval I.isa).setWidth 16

theorem kernel_native (I : Instructions) (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (words : Nat→BitVec 32) (work : List Program) (order : (programs (grid (parameters c))).Perm work) :
    let before:=entry (Specification.nativeReader c words)
    ∃ after,runPrograms (fun _=>Native.algebra) (fun _=>Native.dotModel)
        (parameters c) (grid (parameters c)) work before=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,∀ d (hd : d<96),
        (logical (observe (nativeDecode I d) c words after) c .output b h n d).setWidth 16=
          nativeReference I c words b h n d hd) ∧
      RawFrame c before after ∧
      (∀ d a,¬Stored c a → observe (nativeDecode I d) c words after a=words a) ∧
      (∀ owner r, r∈reads → ∀ b h n d,Valid (parameters c) r b h n d →
        logical (observe (nativeDecode I owner) c words after) c r b h n d=logical words c r b h n d) ∧
      Access.Safe (parameters c) ∧ Closed (parameters c) (pointers c) := by
  dsimp only
  obtain ⟨after,run,values,frame⟩:=complete Native.algebra Native.dotModel c hc laws
    (entry (Specification.nativeReader c words)) work order
  refine ⟨after,run,?_,frame,?_,?_,Access.safe _ (domain c hc laws).canonical,Machine.closed _ _ (domain c hc laws)⟩
  · intro b hb h hh n hn d hd
    have valid : Valid (parameters c) .output b h n d:=⟨hb,hh,hn,hd⟩
    rw [actual_read _ c hc laws .output b h n d valid,observe_store _ _ _ _ b h n d valid,
      values b hb h hh n hn d hd]
    simp only [Cell.read_same]
    rw [algorithm_at _ _ c (Specification.nativeReader c words) b h n hh]
    simp [nativeDecode,nativeReference,Specification.nativeTarget,lane]
  · exact fun d a outside=>observe_frame _ _ _ _ a outside
  · intro owner r hr b h n d valid
    unfold logical
    exact observe_frame _ _ _ _ _ (read_exterior c hc laws r hr b h n d valid)

theorem empty_kernel {α β : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (c : Call) (empty : ¬Live (parameters c)) (before : State α) (work : List Program)
    (order : (programs (grid (parameters c))).Perm work) (decode : α→β) (words : Nat→β) :
    work=[] ∧ runPrograms (fun _=>M) (fun _=>dot) (parameters c) (grid (parameters c)) work before=some before ∧
      ∀ a,observe decode c words before a=words a := by
  have nil : work=[]:=List.nil_perm.mp (by simpa [no_grid c empty] using order)
  exact ⟨nil,by simp [nil,runPrograms],empty_observe decode c empty words before⟩

#axiomsClean kernel_real
#axiomsClean kernel_native
#axiomsClean empty_kernel
#specNonCircular mathematicalTarget avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
#specNonCircular nativeReference avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly
