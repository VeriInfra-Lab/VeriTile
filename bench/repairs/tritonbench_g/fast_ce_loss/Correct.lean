import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.LogicalIO
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.FastCeLoss.Correct
open VeriTile.Triton FP.Structural KernelSource
open scoped BigOperators
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/- The .real channel tag is the existing Structural cell representation;
FP32 storage/arithmetic contexts remain fixed by each unchanged typed source. -/

/-- Canonical FP32 source, signed-int64 labels, complete logical logit row,
separately masked target gather, both outputs and no private scratch.
The read/write fields retain the actual row strides and output addresses. -/
def forwardIO (p : Parameters) (row : Nat) :=
  LogicalIO.forwardView KernelReal.algebra RealIO.decode p row 0 Bool.false

/-- Every chunk writes LSE; only chunk zero writes loss or gathers the target. -/
def chunkedIO (p : Parameters) (row chunk : Nat) :=
  LogicalIO.forwardView KernelReal.algebra RealIO.decode p row chunk Bool.true

/-- The saved LSE and active logit tile are read before the in-place gradient
stores. Ignored labels suppress the upstream-gradient read. -/
def backwardIO (p : Parameters) (row chunk : Nat) :=
  LogicalIO.backwardView KernelReal.algebra RealIO.decode p row chunk

/-- The original mathematical label domain; no tensor-finiteness restriction. -/
def inputDomain (p : Parameters) (xs : LogicalIO.ForwardInputs p) : Prop :=
  let target := xs.label
  target = -100 ∨ (0 ≤ target ∧ target<(p.vocab : Int))

/-- Scalar flags in the source agree with these mathematical branches via
KernelReal.Flags. Zero scaling/softcap disables its corresponding transform. -/
def transformed (p : Parameters) (x : ℝ) : ℝ :=
  let scale := if p.scaling=0 then 1 else p.scaling
  if p.softcap=0 then scale*x else p.softcap*Real.tanh (scale*x/p.softcap)

def slope (p : Parameters) (x : ℝ) : ℝ :=
  let scale := if p.scaling=0 then 1 else p.scaling
  scale*(if p.softcap=0 then 1 else 1-(Real.tanh (scale*x/p.softcap))^2)

/-- The full logical vocabulary is summed, rather than the padded tile width.
The two output channels are LSE and unnormalized row loss. -/
def forwardFormula (p : Parameters) (xs : LogicalIO.ForwardInputs p) :
    LogicalIO.ForwardOutputs :=
  let x := LogicalIO.logit xs.logits
  let label := xs.label
  let lse := Real.log (∑ j ∈ Finset.range p.vocab, Real.exp (transformed p (x j)))
  ⟨lse,if label = -100 then 0 else lse-transformed p (x label.toNat)⟩

/-- Each chunk's LSE sums only active logical columns. Only chunk zero's loss
channel is observed by the shared IO; its value is the negative target term. -/
def chunkedFormula (p : Parameters) (chunk : Nat)
    (xs : LogicalIO.ForwardInputs p) : LogicalIO.ForwardOutputs :=
  let x := LogicalIO.logit xs.logits
  let label := xs.label
  ⟨Real.log (∑ j ∈ Finset.range p.block,
      if chunk*p.block+j<p.vocab then Real.exp (transformed p (x (chunk*p.block+j))) else 0),
    if label = -100 then 0 else -(transformed p (x label.toNat))⟩

/-- The gradient uses the supplied saved LSE/upstream, without a hidden
forward computation or host loss-normalization claim. -/
def backwardFormula (p : Parameters) (chunk : Nat)
    (xs : LogicalIO.BackwardInputs p) : Fin p.block → ℝ :=
  let label := xs.label
  let saved : ℝ := xs.savedLSE
  let upstream : ℝ := xs.upstream
  fun j =>
    let col := chunk*p.block+j.val
    let x := LogicalIO.logit xs.logits col
    if label = -100 then (0 : ℝ) else upstream *
      (Real.exp (transformed p x-saved) - if (col : Int)=label then 1 else 0) *
      slope p x

specification forward_kernel (p : Parameters) (row : Nat) (flags : KernelReal.Flags p)
    (positive : 0<p.vocab) (covered : p.vocab ≤ p.block) (distinct : p.loss≠p.lse) :
    Spec.Real ((forwardIO p row) ⊨{inputDomain p} (forwardFormula p)) :=
  (LogicalIO.forward_iff p row 0 Bool.false _ _).mpr (RealIO.forward_correct p row flags positive covered distinct)

specification chunked_kernel (p : Parameters) (row chunk : Nat) (flags : KernelReal.Flags p)
    (positive : 0<p.block) (nonempty : chunk*p.block<p.vocab) (distinct : p.loss≠p.lse) :
    Spec.Real ((chunkedIO p row chunk) ⊨{inputDomain p} (chunkedFormula p chunk)) :=
  (LogicalIO.forward_iff p row chunk Bool.true _ _).mpr (RealIO.chunked_correct p row chunk flags positive nonempty distinct)

specification backward_kernel (p : Parameters) (row chunk : Nat) (flags : KernelReal.Flags p) :
    Spec.Real ((backwardIO p row chunk) ⊨ (backwardFormula p chunk)) :=
  (LogicalIO.backward_iff p row chunk _).mpr (RealIO.backward_correct p row chunk flags)

#stmtSurfaceSubset forward_kernel ⊆
  [LogicalIO.ForwardInputs,LogicalIO.ForwardOutputs,Spec.Real,Parameters,Parameters.vocab,Parameters.block,Parameters.loss,Parameters.lse,
   KernelReal.Flags,RegionName,forwardIO,LogicalProgramIO.ImplementsOn,inputDomain,forwardFormula]
#stmtSurfaceSubset chunked_kernel ⊆
  [LogicalIO.ForwardInputs,LogicalIO.ForwardOutputs,Spec.Real,Parameters,Parameters.vocab,Parameters.block,Parameters.loss,Parameters.lse,
   KernelReal.Flags,RegionName,chunkedIO,LogicalProgramIO.ImplementsOn,inputDomain,chunkedFormula]
#stmtSurfaceSubset backward_kernel ⊆
  [LogicalIO.BackwardInputs,Parameters.block,Spec.Real,Parameters,KernelReal.Flags,backwardIO,LogicalProgramIO.Implements,backwardFormula]
#specNonCircular forwardFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular chunkedFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular backwardFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#axiomsClean forward_kernel
#axiomsClean chunked_kernel
#axiomsClean backward_kernel
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Correct
