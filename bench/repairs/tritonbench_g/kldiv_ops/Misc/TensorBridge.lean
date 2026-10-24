import bench.repairs.tritonbench_g.kldiv_ops.Proofs.KernelExecution
import bench.repairs.tritonbench_g.kldiv_ops.Misc.NativeCorrect
import bench.repairs.tritonbench_g.kldiv_ops.Misc.RealHost
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Native

/-! Complete Tensor applications retained with their original statements and source correspondence. -/

namespace VeriTile.Bench.Repairs.KldivOps.Kernel.TensorBridge.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Contiguous
noncomputable section
set_option autoImplicit false

/-- Ideal-real interpretation of the same typed source. It makes no claim
that floating words encode arbitrary reals or that native log/exp are exact. -/
def adapter : Adapter ℝ ℝ := ⟨RealExecution.algebra,id,id⟩
def negation : Negation ℝ := ⟨fun x => -x,neg_neg⟩

theorem forward_expected {M β : Type} (ops : Ops M ℝ β) (initial : M)
    (prediction target : View) (cols : Nat) (flag : Bool) (red : Host.Reduction)
    (eps : ℝ) (i : Nat) :
    forwardExpected ops negation adapter initial prediction target cols flag red eps i =
      forwardTarget flag eps red cols (logical ops negation initial prediction)
        (logical ops negation initial target) i := by
  unfold forwardExpected forwardTarget
  split
  · exact RealExecution.scalar_real flag eps _ _
  · change FPGraph.row RealExecution.algebra cols (Host.blockSize cols) flag eps _ _ = _
    rw [RealExecution.row_ideal flag eps cols (Host.blockSize cols) (Host.block_positive cols)]
    rfl

/-- Every admitted forward launch terminates, writes the independent KL
formula at all elements or row outputs, and preserves the entire memory
complement. Tensor logical values, native pointers and signed control are
composed in this result. Python sum/mean/batchmean postprocessing is separate. -/
theorem forward {M β : Type} (ops : Ops M ℝ β) (initial : M)
    (p t : Tensor) (prediction target output : View) (rows cols : Nat)
    (flag : Bool) (red : Host.Reduction) (eps : ℝ)
    (domain : ForwardDomain p t prediction target output rows cols red) (dispatch : Dispatch rows) :
    ∃ after, ForwardPost ops negation adapter p t prediction target output rows cols flag red eps
      (forwardTarget flag eps red cols (logical ops negation initial prediction)
        (logical ops negation initial target)) initial after :=
  completeForward ops negation adapter initial p t prediction target output rows cols flag red eps
    domain dispatch _ (forward_expected ops initial prediction target cols flag red eps)

/-- The backward launch writes the independent unscaled derivative directly
into the supplied destination. The selected alias guard gives stability of
all target loads through every write; no cloned snapshot is introduced. -/
theorem backward {M β : Type} (ops : Ops M ℝ β) (initial : M)
    (d t g : Tensor) (target output : View) (rows cols : Nat) (flag : Bool)
    (domain : BackwardDomain d t g target output rows cols) (dispatch : Dispatch rows) :
    ∃ after, BackwardPost ops negation adapter d t target output rows cols flag
      (fun i => backwardTarget flag (logical ops negation initial target i)) initial after := by
  apply completeBackward ops negation adapter initial d t g target output rows cols flag domain dispatch
  intro i
  exact RealExecution.backward_scalar_ideal flag _

#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Kernel.TensorBridge.Correct

namespace VeriTile.Bench.Repairs.KldivOps.Kernel.TensorBridge.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Contiguous
open KldivTriton.Native (Word)
noncomputable section
set_option autoImplicit false

/-- Independent source-ordered FP32 graph. The primitive ports and ordered
reduction trees are an ordinary compiler model, not numerical experiments or
all-words equality assumptions. The raw Python float is bound exactly once. -/
def forwardTarget (p : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (flag : Bool) (eps : BitVec 64) (red : Host.Reduction) (cols : Nat)
    (prediction target : Nat → Word) (i : Nat) : Word :=
  if red = .none then Contiguous.Native.loss p flag (p.bindEpsilon eps) (prediction i) (target i)
  else Contiguous.Native.row p plans cols flag (p.bindEpsilon eps)
    (fun c => prediction (i*cols+c)) (fun c => target (i*cols+c))

def backwardTarget (p : Contiguous.Native.Ports) (flag : Bool) (target : Word) : Word :=
  if flag then p.negate (p.exponential target) else p.multiplyMinusOne target

def OutputBytes {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte)
    (output : View) (expected : Nat → Word) (after : M) : Prop :=
  ∀ i, i < output.numel → ∀ byte, byte < 4 →
    ops.observe after (output.key i).owner ((output.key i).address+byte) =
      (expected i).extractLsb' (8*byte) 8

theorem forward_expected {M β : Type} (ops : Ops M Word β)
    (p : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans) (initial : M)
    (prediction target : View) (cols : Nat) (flag : Bool) (red : Host.Reduction)
    (eps : BitVec 64) (i : Nat) :
    forwardExpected ops SwigluTriton.Words.signNegation (Contiguous.Native.adapter p plans)
      initial prediction target cols flag red (p.bindEpsilon eps) i =
    forwardTarget p plans flag eps red cols
      (logical ops SwigluTriton.Words.signNegation initial prediction)
      (logical ops SwigluTriton.Words.signNegation initial target) i := by
  unfold forwardExpected forwardTarget
  split
  · exact congrArg KldivTriton.Native.Node.value
      (Contiguous.Native.scalar_lowering p plans flag (p.bindEpsilon eps) _ _)
  · exact Contiguous.Native.row_lowering p plans cols flag (p.bindEpsilon eps) _ _

/-- Complete selected kernel versus an independently defined ordered native
word graph. This includes successful typed execution on evolving shared
memory, every output word and byte, exact pointers, control and the full
memory frame. No numerical transformation or deferred numerical law is used. -/
theorem forward {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte)
    (codec : SwigluTriton.Words.Codec ops) (p : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (initial : M) (predictionTensor targetTensor : Tensor) (prediction target output : View)
    (rows cols : Nat) (flag : Bool) (red : Host.Reduction) (eps : BitVec 64)
    (domain : ForwardDomain predictionTensor targetTensor prediction target output rows cols red)
    (dispatch : Dispatch rows) :
    ∃ after, ForwardPost ops SwigluTriton.Words.signNegation (Contiguous.Native.adapter p plans)
      predictionTensor targetTensor prediction target output rows cols flag red (p.bindEpsilon eps)
      (forwardTarget p plans flag eps red cols
        (logical ops SwigluTriton.Words.signNegation initial prediction)
        (logical ops SwigluTriton.Words.signNegation initial target)) initial after ∧
      OutputBytes ops output (forwardTarget p plans flag eps red cols
        (logical ops SwigluTriton.Words.signNegation initial prediction)
        (logical ops SwigluTriton.Words.signNegation initial target)) after := by
  obtain ⟨after,result⟩ := completeForward ops SwigluTriton.Words.signNegation
    (Contiguous.Native.adapter p plans) initial predictionTensor targetTensor prediction target output
    rows cols flag red (p.bindEpsilon eps) domain dispatch _
    (forward_expected ops p plans initial prediction target cols flag red eps)
  refine ⟨after,result,?_⟩
  intro i hi byte hb
  simpa only [domain.outputPositive,Negation.encode,Bool.false_eq_true,if_false] using
    SwigluTriton.Words.logical_bytes ops codec after output i _ (result.values i hi) byte hb

theorem backward {M : Type} (ops : Ops M Word SwigluTriton.Words.Byte)
    (codec : SwigluTriton.Words.Codec ops) (p : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (initial : M) (d t g : Tensor) (target output : View) (rows cols : Nat) (flag : Bool)
    (domain : BackwardDomain d t g target output rows cols) (dispatch : Dispatch rows) :
    ∃ after, BackwardPost ops SwigluTriton.Words.signNegation (Contiguous.Native.adapter p plans)
      d t target output rows cols flag
      (fun i => backwardTarget p flag (logical ops SwigluTriton.Words.signNegation initial target i)) initial after ∧
      OutputBytes ops output
        (fun i => backwardTarget p flag (logical ops SwigluTriton.Words.signNegation initial target i)) after := by
  obtain ⟨after,result⟩ := completeBackward ops SwigluTriton.Words.signNegation
    (Contiguous.Native.adapter p plans) initial d t g target output rows cols flag domain dispatch
    (fun i => backwardTarget p flag (logical ops SwigluTriton.Words.signNegation initial target i))
    (fun i => congrArg KldivTriton.Native.Node.value (Contiguous.Native.backward_lowering p plans flag _))
  refine ⟨after,result,?_⟩
  intro i hi byte hb
  have positive := domain.destinationBinding.negative.trans domain.admitted.left.noNeg
  simpa only [positive,Negation.encode,Bool.false_eq_true,if_false] using
    SwigluTriton.Words.logical_bytes ops codec after output i _ (result.values i hi) byte hb

#specNonCircular forwardTarget avoiding [KldivOps.forward,executeForward,Contiguous.forwardKernel,Contiguous.forwardExpected]
#specNonCircular backwardTarget avoiding [KldivOps.backward,executeBackward,Contiguous.backwardKernel]
#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Kernel.TensorBridge.FPEquiv
