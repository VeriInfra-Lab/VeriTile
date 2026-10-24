import bench.repairs.tritonbench_g.token_softmax_bloom.Correct
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.KernelSource
import bench.repairs.tritonbench_g.token_softmax_bloom.FPEquiv
import bench.repairs.tritonbench_g.token_softmax_bloom.Misc.TensorBridge
import bench.repairs.tritonbench_g.token_softmax_llama.Correct
import bench.repairs.tritonbench_g.token_softmax_llama.FPEquiv

/-! Optional Python-check correspondence; primary kernel results remain in Correct and FPEquiv. -/
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open TokenSoftmaxBloom TokenSoftmaxBloom.KernelTensor TokenSoftmaxBloom.KernelGrid
noncomputable section

theorem python_checks_iff_preconditions (c : Call) (starts lengths : Nat → Int) :
    checkInput c starts lengths=.ok () ↔ InputContract c starts lengths :=
  KernelAdmission.check_input_iff c starts lengths

#axiomsClean python_checks_iff_preconditions
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.TensorBridge


namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open TokenSoftmaxBloom TokenSoftmaxBloom.KernelTensor TokenSoftmaxBloom.KernelGrid
noncomputable section
set_option maxHeartbeats 5000000

abbrev referenceReal := KernelReal.reference

abbrev executeReal := TokenSoftmaxBloom.TensorBridge.executeReal

/-- All segments and heads, actual successful full-grid source execution,
original logical Tensor values, exact native addresses and complete cell frame.
The mathematical projection accepts arbitrary Real inputs; FP32 words are
handled independently in FPEquiv without a finite-word restriction. -/
theorem token_softmax_kernel_correct (c : TensorCall) (read : Nat → ℝ)
    (metadataRead : Nat → Word) (before : NativeCells KernelReal.Scalar)
    (binding : InputBinding c (KernelReal.reads read) before)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (work : List Grid.PID)
    (order : (Grid.fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work) :
    ∃ after, executeReal c read metadataRead work before=some after ∧
      (VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation → ∃ raw,
        Grid.run KernelReal.model (params c) work (view c (KernelReal.reads read) metadataRead before)=some raw ∧
        after=project c before raw ∧ RawFrame c (KernelReal.reads read) metadataRead before work raw) ∧
      (contentActive c.observation → ∀ b < rows c.start.observation, ∀ head < rows c.input.observation,
        ∀ n < length c metadataRead b,
        WithBot.unbotD 0 ((after (c.output.tensor.storage.address (c.output.tensor.physicalIndex
          (head*cols c.output.observation+start c metadataRead b+n)))).read .fp32)=
            referenceReal c metadataRead read b head n) ∧
      (∀ address,Exterior c metadataRead address → after address=before address) ∧
      (∀ t∈[c.input.observation,c.start.observation,c.length.observation], ∀ i < t.storage.elements,
        after (t.storage.address i)=before (t.storage.address i)) :=
  TokenSoftmaxBloom.TensorBridge.token_softmax_kernel_correct c read metadataRead before binding contract work order

#specNonCircular referenceReal avoiding [Strided.source,executeReal,Grid.run]
#axiomsClean token_softmax_kernel_correct
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.TensorBridge


namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open TokenSoftmaxBloom TokenSoftmaxBloom.KernelTensor TokenSoftmaxBloom.KernelGrid
noncomputable section
set_option maxHeartbeats 5000000

abbrev referenceNative := KernelNative.reference

def executeNative (I : KernelNative.Instructions) (V : Nat) (candidate : Bool) (c : TensorCall)
    (work : List Grid.PID) (before : NativeCells Word) : Option (NativeCells Word) :=
  if VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation then
    let initial := view c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before)
    let run := if candidate then KernelSource.candidateRun (KernelNative.model I V) (params c) work initial
      else Grid.run (KernelNative.model I V) (params c) work initial
    run.map (KernelNative.observe V c before work)
  else some before

/-- Complete typed source/candidate grid success with one common raw state,
actual Tensor words at every output, raw typed and native physical Cell frames,
and preservation of every cell of all read allocations. No scalar numerical law
or whole-row success/result premise is assumed. -/
theorem token_softmax_kernel_fp32 (I : KernelNative.Instructions) (V : Nat)
    (c : TensorCall) (before : NativeCells Word)
    (contract : InputContract c.observation (observedStart c (KernelNative.wordRead before))
      (observedLength c (KernelNative.wordRead before)))
    (work : List Grid.PID)
    (order : (Grid.fullGrid (rows c.start.observation) (rows c.input.observation)).Perm work) :
    ∃ after, executeNative I V Bool.false c work before=some after ∧
      executeNative I V Bool.true c work before=some after ∧
      (VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation → ∃ raw,
        Grid.run (KernelNative.model I V) (params c) work
          (view c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before))=some raw ∧
        KernelSource.candidateRun (KernelNative.model I V) (params c) work
          (view c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before))=some raw ∧
        after=KernelNative.observe V c before work raw ∧
        RawFrame c (KernelNative.nodeRead before) (KernelNative.wordRead before) (KernelNative.lift before) work raw) ∧
      (contentActive c.observation → ∀ b < rows c.start.observation, ∀ head < rows c.input.observation,
        ∀ n < length c (KernelNative.wordRead before) b,
        (after (c.output.tensor.storage.address (c.output.tensor.physicalIndex
          (head*cols c.output.observation+start c (KernelNative.wordRead before) b+n)))).read .fp32=
            referenceNative I V c before b head n) ∧
      (∀ address,Exterior c (KernelNative.wordRead before) address → after address=before address) ∧
      (∀ t∈[c.input.observation,c.start.observation,c.length.observation], ∀ i < t.storage.elements,
        after (t.storage.address i)=before (t.storage.address i)) :=
  TokenSoftmaxBloom.TensorBridge.token_softmax_kernel_fp32 I V c before contract work order

#specNonCircular referenceNative avoiding [Strided.source,KernelSource.indexCommutedSource,executeNative,Grid.run,KernelNative.observe]
#axiomsClean token_softmax_kernel_fp32
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.TensorBridge
