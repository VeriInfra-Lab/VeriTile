import bench.repairs.tritonbench_g.kldiv_triton.Proofs.KernelExecution
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.RealBridge
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.SourceNative
import bench.repairs.tritonbench_g.kldiv_triton.Correct
import bench.repairs.tritonbench_g.kldiv_triton.Misc.NativeCorrect

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.TensorBridge.RealApplication
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Contiguous
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 4000000

theorem reference_real {M β : Type} (ops : Ops M ℝ β) (prediction target : Matrix)
    (flag : Bool) (mode : Math.Mode) (memory : M) (r c : Nat) :
    forwardReference ops RealHost.negation Storage.real prediction target flag mode memory r c=
      forwardTarget flag mode prediction.cols (row ops RealHost.negation memory prediction r)
        (row ops RealHost.negation memory target r) c := by
  unfold forwardReference forwardTarget
  split
  · exact RealModel.loss_real flag _ _
  · change Tile.chunkSum RealModel.model flag prediction.cols (Host.blockSize prediction.cols) c _ _=_
    rw [RealModel.chunk_real]
    simp only [VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.chunkLoss,
      VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.paddedLoss_eq,←Math.loss_original]
    rfl

/-- Complete selected forward kernel: independent element losses or individual
chunk sums, successful full DSL execution, all outputs, exact physical frame,
actual Tensor pointers and signed control. No Python post-reduction is claimed. -/
theorem forward {M β : Type} (ops : Ops M ℝ β) (a b o : Tensor)
    (prediction target output : Matrix) (flag : Bool) (mode : Math.Mode)
    (domain : ForwardDomain a b o prediction target output mode) (dispatch : Dispatch prediction.rows)
    (schedule : List Nat) (permutation : schedule.Perm (List.range output.rows)) (initial : M) :
    ∃ final,ForwardPost ops RealHost.negation Storage.real a b o prediction target output flag mode schedule
      (fun r c => forwardTarget flag mode prediction.cols (row ops RealHost.negation initial prediction r)
        (row ops RealHost.negation initial target r) c) initial final := by
  apply completeForward ops RealHost.negation Storage.real a b o prediction target output flag mode
    domain dispatch schedule permutation initial
  intro r c _ _
  exact reference_real ops prediction target flag mode initial r c

/-- Complete selected backward kernel writes the unscaled derivative directly
to the actual supplied destination. Its whole exterior memory is preserved. -/
theorem backward {M β : Type} (ops : Ops M ℝ β) (a b g : Tensor)
    (destination target : Matrix) (gradient : View) (flag : Bool)
    (domain : BackwardDomain a b g destination target gradient) (dispatch : Dispatch destination.rows)
    (schedule : List Nat) (permutation : schedule.Perm (List.range destination.rows)) (initial : M) :
    ∃ final,BackwardPost ops RealHost.negation Storage.real a b destination target flag schedule
      (fun r c => backwardTarget flag (row ops RealHost.negation initial target r c)) initial final := by
  apply completeBackward ops RealHost.negation Storage.real a b g destination target gradient flag
    domain dispatch schedule permutation initial
  intro r c _ _
  exact RealModel.gradient_real flag _

#axiomsClean reference_real
#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.TensorBridge.RealApplication


namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.TensorBridge.NativeApplication
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Contiguous
open SourceNative (Word Byte Primitives Plans adapter)
noncomputable section
set_option autoImplicit false

/-- Every byte of every logical output coordinate, from the same word graph. -/
def OutputBytes {M : Type} (ops : Ops M Word Byte) (output : Matrix)
    (expected : Nat → Nat → Word) (after : M) : Prop :=
  ∀ r c,r<output.rows → c<output.cols → ∀ byte,byte<4 →
    ops.observe after (output.key r c).owner ((output.key r c).address+byte)=
      (expected r c).extractLsb' (8*byte) 8

theorem output_bytes {M : Type} (ops : Ops M Word Byte) (codec : SwigluTriton.Words.Codec ops)
    (output : Matrix) (positive : output.view.negative=Bool.false) (expected : Nat → Nat → Word) (after : M)
    (values : ∀ r c,r<output.rows → c<output.cols →
      row ops SwigluTriton.Words.signNegation after output r c=expected r c) :
    OutputBytes ops output expected after := by
  intro r c hr hc byte hb
  have raw : ops.read after (output.key r c)=expected r c :=
    (row_raw ops SwigluTriton.Words.signNegation after output positive r c hc).trans (values r c hr hc)
  rw [codec.byte _ _ _ hb,raw]

/-- Complete source versus the independent ordered native loss/chunk graph.
Primitive meanings and concrete addition plans are ordinary interpretation
parameters; no numeric transformation or primitive experiment is deferred. -/
theorem forward {M : Type} (ops : Ops M Word Byte) (codec : SwigluTriton.Words.Codec ops)
    (p : Primitives) (plans : Plans) (a b o : Tensor) (prediction target output : Matrix)
    (flag : Bool) (mode : Math.Mode) (domain : ForwardDomain a b o prediction target output mode)
    (dispatch : Dispatch prediction.rows) (schedule : List Nat)
    (permutation : schedule.Perm (List.range output.rows)) (initial : M) :
    ∃ final,ForwardPost ops SwigluTriton.Words.signNegation (adapter p plans) a b o prediction target output
      flag mode schedule (fun r c => SourceNative.forwardTarget p plans flag mode prediction.cols
        (row ops SwigluTriton.Words.signNegation initial prediction r)
        (row ops SwigluTriton.Words.signNegation initial target r) c) initial final ∧
      OutputBytes ops output (fun r c => SourceNative.forwardTarget p plans flag mode prediction.cols
        (row ops SwigluTriton.Words.signNegation initial prediction r)
        (row ops SwigluTriton.Words.signNegation initial target r) c) final := by
  obtain ⟨final,result⟩ := completeForward ops SwigluTriton.Words.signNegation (adapter p plans)
    a b o prediction target output flag mode domain dispatch schedule permutation initial _
    (fun r c _ _ => SourceNative.forward_reference ops SwigluTriton.Words.signNegation p plans
      prediction target flag mode initial r c)
  exact ⟨final,result,output_bytes ops codec output
    (domain.outputBinding.negative.trans domain.outputAdmitted.noNeg) _ final result.values⟩

/-- Source raw-target multiply-by-minus-one and log-target exp then unary
negation remain distinct ordered graphs, with all destination words/bytes and
full exterior frame proved from successful typed source execution. -/
theorem backward {M : Type} (ops : Ops M Word Byte) (codec : SwigluTriton.Words.Codec ops)
    (p : Primitives) (plans : Plans) (a b g : Tensor) (destination target : Matrix) (gradient : View)
    (flag : Bool) (domain : BackwardDomain a b g destination target gradient)
    (dispatch : Dispatch destination.rows) (schedule : List Nat)
    (permutation : schedule.Perm (List.range destination.rows)) (initial : M) :
    ∃ final,BackwardPost ops SwigluTriton.Words.signNegation (adapter p plans) a b destination target flag schedule
      (fun r c => SourceNative.derivative p flag (row ops SwigluTriton.Words.signNegation initial target r c)) initial final ∧
      OutputBytes ops destination
        (fun r c => SourceNative.derivative p flag (row ops SwigluTriton.Words.signNegation initial target r c)) final := by
  obtain ⟨final,result⟩ := completeBackward ops SwigluTriton.Words.signNegation (adapter p plans)
    a b g destination target gradient flag domain dispatch schedule permutation initial _
    (fun _ _ _ _ => SourceNative.derivative_lowering p plans flag _)
  exact ⟨final,result,output_bytes ops codec destination
    (domain.destinationBinding.negative.trans domain.admitted.left.noNeg) _ final result.values⟩

#axiomsClean output_bytes
#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.TensorBridge.NativeApplication
