import bench.repairs.tritonbench_g.attention_kernel_aligned.KernelOnly.BF16Binding
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16AcceptedReal
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16NativeProjection
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.KernelOnly

open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue
open VeriTile.Bench.Repairs.AttentionKernelAligned.KernelOnly
open VeriTile.Bench.Repairs.AttentionKernelAligned.BF16 Source Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-- Independent direct-address base-two bias softmax. -/
def mathematicalTarget (c : Call) (read : Nat → ℝ) (b h n d : Nat) :=
  Real.expected (parameters c) (caller c read) b h n d
def realEntry (c : Call) (read : Nat → ℝ) := entry (fun r off=>some (caller c read r off))
def realDecode (x : Real.R) : ℝ := x.getD 0

theorem kernel_real (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (_dtype : (c .query).metadata.dtype=.floating .bf16)
    (read : Nat → ℝ) (work : List Program) (order : (programs (grid (parameters c))).Perm work) :
    ∃ after,runPrograms (fun _=>Real.algebra) (fun _=>Real.model)
        (parameters c) (grid (parameters c)) work (realEntry c read)=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,
        ∀ d<(parameters c).head,
        logical (observe realDecode c read after) c .output b h n d=mathematicalTarget c read b h n d) ∧
      RawFrame c (realEntry c read) after ∧
      (∀ a,¬Stored c a → observe realDecode c read after a=read a) ∧
      (∀ r∈reads,∀ b h n d,Valid (parameters c) r b h n d →
        logical (observe realDecode c read after) c r b h n d=logical read c r b h n d) := by
  obtain ⟨after,run,values,frame⟩:=complete Real.algebra Real.model c hc (realEntry c read) work order
  refine ⟨after,run,?_,frame,?_,?_⟩
  · intro b hb h hh n hn d hd
    have valid : Valid (parameters c) .output b h n d := ⟨hb,hh,hn,hd⟩
    rw [actual_read _ c hc laws .output b h n d valid,observe_store _ _ _ _ b h n d valid,
      values b hb h hh n hn d hd]
    have input : Real.Reads (parameters c)
        (start (grid (parameters c)) (n/c.bm,b*(parameters c).heads+h,0) (realEntry c read)) (caller c read) := by
      intro r hr off
      rfl
    have active : Source.row (parameters c)
        (start (grid (parameters c)) (n/c.bm,b*(parameters c).heads+h,0) (realEntry c read))
        (lane (parameters c) (guard c hc).selected n d hd).1<(parameters c).length := by
      change n/c.bm*c.bm+n%c.bm<(parameters c).length
      simpa only [Nat.mul_comm,Nat.div_add_mod] using hn
    have answer:=Real.answer_real (parameters c) _ (caller c read) input _ active
      ((guard c hc).selected.blockN_pos _)
    have target:=Real.expected_at (parameters c) (guard c hc).selected (realEntry c read) (caller c read) b h n d hh hd
    simpa [Cell.read_same,Whole.answer,realDecode,mathematicalTarget] using
      congrArg realDecode (answer.trans (congrArg some target))
  · exact fun a outside=>observe_frame _ _ _ _ a outside
  · intro r hr b h n d valid
    unfold logical
    exact observe_frame _ _ _ _ _ (read_exterior c hc laws r hr b h n d valid)

/-- Compiler-selected integer routes only. Resource and runtime availability
remain external; no numerical or complete lowering equation is a field. -/
structure CompilerPlan (c : Call) where
  columns : c.nativeColumns∈[1,2,4,8]
  columns_bound : c.nativeColumns≤c.numWarps
  routes : VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan.Routes
  routing : VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan.LocalRouting (parameters c) routes
  serialSource : Fin 128 → Nat
  denominatorSource : Fin 16 → Fin 8 → Nat
  serialRouting : c.nativeSerial=Bool.true →
    (∀ j : Fin 128,j.val<(parameters c).blockN → serialSource j=j.val) ∧
    (∀ row col,denominatorSource row col=row.val)

/-- Preserve every original BF16 compiler binding, including the BM128/W4
serial normalizer. Only integer routing metadata is projected; no serial-mode
restriction, numerical equality, or runtime/resource condition is added. -/
def CompilerPlan.ofOriginal (c : Call) (storage : TensorStorage (parameters c))
    (abi : InputABI (parameters c)) (pl : Placement (parameters c) storage)
    (profile : Profile) (binding : CompilerBinding (parameters c) storage abi pl profile) :
    CompilerPlan c where
  columns:=binding.columns
  columns_bound:=binding.columns_bound
  routes:=binding.routes
  routing:=binding.routing
  serialSource:=binding.serialSource
  denominatorSource:=binding.denominatorSource
  serialRouting:=binding.serialRouting

theorem compilerPlan_original_domain (c : Call) (storage : TensorStorage (parameters c))
    (abi : InputABI (parameters c)) (pl : Placement (parameters c) storage)
    (profile : Profile) (binding : CompilerBinding (parameters c) storage abi pl profile) :
    Nonempty (CompilerPlan c) := ⟨CompilerPlan.ofOriginal c storage abi pl profile binding⟩

def nativeReader (c : Call) (words : Nat → BitVec 16) (r : RegionName) (off : Nat) : Native.Expr :=
  fun _=>.ready (.bits ((caller c words r off).zeroExtend 32))
def nativeDecode (I : Native.Instructions) (d : Nat) (x : Native.Expr) : BitVec 16 :=
  ((Native.lower (x (Native.physicalOwner d))).eval I.ops).setWidth 16
def nativeReference (I : Native.Instructions) (c : Call) (words : Nat → BitVec 16)
    (b h n d : Nat) (selected : Selected (parameters c)) (hd : d<(parameters c).head) : BitVec 16 :=
  nativeDecode I d (Specification.outputValue (Native.modelFor (parameters c)) (parameters c)
    (nativeReader c words) (n/c.bm) b h (lane (parameters c) selected n d hd))

theorem kernel_native (I : Native.Instructions) (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (_dtype : (c .query).metadata.dtype=.floating .bf16) (_plan : CompilerPlan c)
    (words : Nat → BitVec 16) (work : List Program) (order : (programs (grid (parameters c))).Perm work) :
    let before:=entry (nativeReader c words)
    ∃ after,runPrograms (fun _=>Native.algebraFor (parameters c)) (fun _=>Native.modelFor (parameters c))
        (parameters c) (grid (parameters c)) work before=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,
        ∀ d (hd : d<(parameters c).head),
        logical (observe (nativeDecode I d) c words after) c .output b h n d=
          nativeReference I c words b h n d (guard c hc).selected hd) ∧
      RawFrame c before after ∧
      (∀ d a,¬Stored c a → observe (nativeDecode I d) c words after a=words a) ∧
      (∀ owner r,r∈reads → ∀ b h n d,Valid (parameters c) r b h n d →
        logical (observe (nativeDecode I owner) c words after) c r b h n d=logical words c r b h n d) := by
  dsimp only
  obtain ⟨after,run,values,frame⟩:=complete (Native.algebraFor (parameters c)) (Native.modelFor (parameters c))
    c hc (entry (nativeReader c words)) work order
  refine ⟨after,run,?_,frame,?_,?_⟩
  · intro b hb h hh n hn d hd
    have valid : Valid (parameters c) .output b h n d := ⟨hb,hh,hn,hd⟩
    rw [actual_read _ c hc laws .output b h n d valid,observe_store _ _ _ _ b h n d valid,
      values b hb h hh n hn d hd]
    simp only [Cell.read_same]
    rw [algorithm_at _ c (nativeReader c words) (n/c.bm) b h hh]
    rfl
  · exact fun d a outside=>observe_frame _ _ _ _ a outside
  · intro owner r hr b h n d valid
    unfold logical
    exact observe_frame _ _ _ _ _ (read_exterior c hc laws r hr b h n d valid)

theorem empty_kernel {α β : Type} [Inhabited α] (M : Algebra α) (P : Primitives α)
    (c : Call) (hc : InputContract c) (empty : ¬Whole.Nonempty (parameters c))
    (before : State α) (work : List Program) (order : (programs (grid (parameters c))).Perm work)
    (decode : α → β) (words : Nat → β) :
    work=[] ∧ runPrograms (fun _=>M) (fun _=>P) (parameters c) (grid (parameters c)) work before=some before ∧
      ∀ a,observe decode c words before a=words a := by
  have nil : work=[] := List.nil_perm.mp (by simpa [no_grid c hc empty] using order)
  exact ⟨nil,by simp [nil,runPrograms],empty_observe decode c empty words before⟩

#axiomsClean kernel_real
#axiomsClean kernel_native
#axiomsClean empty_kernel
#axiomsClean compilerPlan_original_domain
#specNonCircular CompilerPlan avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.rounds,Source.advance]
#specNonCircular mathematicalTarget avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.rounds,Source.advance]
#specNonCircular nativeReference avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.rounds,Source.advance]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.KernelOnly
