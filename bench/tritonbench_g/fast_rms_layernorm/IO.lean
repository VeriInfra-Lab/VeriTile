import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealReference
import bench.tritonbench_g.fast_rms_layernorm.Kernels
import VeriTile.Triton.Float.ProgramIO
import Mathlib.Tactic.FinCases
import bench.tritonbench_g.fast_rms_layernorm.Proofs.Geometry
import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightProgram
import VeriTile.Triton.Memory.PreparedAlgorithmProgramIO
import VeriTile.Triton.Float.Erasure
import bench.tritonbench_g.fast_rms_layernorm.Proofs.EntryInputs
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Memory.LogicalIO

/-! Finite logical channels. Backward DX is in place. The DW input is active
only for an empty batch: the existing host has already zero-initialized it and
launches no kernel. Nonempty kernels impose no initial DW value condition. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Ports
open VeriTile Triton FP.Structural
noncomputable section

def inputRole (i : Fin 5) : RegionName := Fin.cases "x" (Fin.cases "w" (Fin.cases "r" (Fin.cases "dx" (Fin.cases "dw" (fun i=>Fin.elim0 i))))) i
def inputSize (h : Call) (i : Fin 5) : Nat := Fin.cases h.cells (Fin.cases h.cols (Fin.cases h.rows (Fin.cases h.cells (Fin.cases h.cols (fun i=>Fin.elim0 i))))) i
def outputRole (backward : Bool) (o : Fin 2) : RegionName :=
  if backward then Fin.cases "dx" (Fin.cases "dw" (fun i=>Fin.elim0 i)) o else Fin.cases "y" (Fin.cases "r" (fun i=>Fin.elim0 i)) o
def outputSize (h : Call) (backward : Bool) (o : Fin 2) : Nat :=
  if backward then Fin.cases h.cells (Fin.cases h.cols (fun i=>Fin.elim0 i)) o else Fin.cases h.cells (Fin.cases h.rows (fun i=>Fin.elim0 i)) o
def active (h : Call) (backward : Bool) (i : Fin 5) : Prop :=
  i.val<2 ∨ (backward=Bool.true ∧ (i.val<4 ∨ h.rows=0))

def inputs (h : Call) : List IOChannel :=
  [⟨.real,[h.cells],"x"⟩,⟨.real,[h.cols],"w"⟩,⟨.real,[h.rows],"r"⟩,
    ⟨.real,[h.cells],"dx"⟩,⟨.real,[h.cols],"dw"⟩]
def outputs (h : Call) (backward : Bool) : List IOChannel :=
  if backward then [⟨.real,[h.cells],"dx"⟩,⟨.real,[h.cols],"dw"⟩]
  else [⟨.real,[h.cells],"y"⟩,⟨.real,[h.rows],"r"⟩]

abbrev Values (α : Type) (h : Call) := (i : Fin 5) → Fin (inputSize h i) → α
abbrev Results (α : Type) (h : Call) (backward : Bool) :=
  (o : Fin 2) → Fin (outputSize h backward o) → α

def logical {α : Type} [Inhabited α] (h : Call) (vals : Values α h) (r : RegionName) (i : Nat) : α :=
  if r="x" then if hi : i<h.cells then vals 0 ⟨i,hi⟩ else default
  else if r="w" then if hi : i<h.cols then vals 1 ⟨i,hi⟩ else default
  else if r="r" then if hi : i<h.rows then vals 2 ⟨i,hi⟩ else default
  else if r="dx" then if hi : i<h.cells then vals 3 ⟨i,hi⟩ else default
  else if r="dw" then if hi : i<h.cols then vals 4 ⟨i,hi⟩ else default
  else default

def ofNative {α : Type} (h : Call) (vals : ChannelValues α (inputs h)) : Values α h :=
  fun i=>Fin.cases (fun j=>vals ⟨0,by simp [inputs]⟩ (j,()))
    (Fin.cases (fun j=>vals ⟨1,by simp [inputs]⟩ (j,()))
      (Fin.cases (fun j=>vals ⟨2,by simp [inputs]⟩ (j,()))
        (Fin.cases (fun j=>vals ⟨3,by simp [inputs]⟩ (j,()))
          (Fin.cases (fun j=>vals ⟨4,by simp [inputs]⟩ (j,())) (fun i=>Fin.elim0 i))))) i

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.Ports

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealIO
open VeriTile Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def source (h : Call) (backward weight : Bool) : ComputeKernel :=
  (if weight then Kernels.weightBackward h 256 else if backward then Kernels.backward h else Kernels.forward h).eraseDType

def job (h : Call) (backward weight : Bool) (pid count : Nat) : PreparedAlgorithmInvocation where
  source:=⟨source h backward weight,by
    cases weight <;> cases backward <;> cases mode : h.gemma
    all_goals simp only [source,Kernels.forward,mode,Bool.false_eq_true,if_true,if_false]
    all_goals simp [Kernels.regularSource,Kernels.gemmaSource,Kernels.regularRuntimeSource,Kernels.gemmaRuntimeSource,
      ComputeKernel.withScalarArguments,ScalarArgument.fp32,ScalarArgument.statement,Kernels.backward,Kernels.inputGradientSource,
      Kernels.weightBackward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
      ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind]⟩
  launch:={pids:=fun _=>pid,numPids:=some (fun _=>count)}
  clearRegisters:=Bool.true
  zeroUndef:=Bool.true

def schedule (h : Call) (backward : Bool) : List PreparedAlgorithmInvocation :=
  if backward then
    if h.rows=0 then [] else
      ((List.finRange (Geometry.featureCount h)).map (fun i=>job h Bool.true Bool.true i.val (Geometry.featureCount h))) ++
      ((List.finRange h.rows).map (fun i=>job h Bool.true Bool.false i.val h.rows))
  else (List.finRange h.rows).map (fun i=>job h Bool.false Bool.false i.val h.rows)

def memory (h : Call) (backward : Bool) : AlgorithmMemoryIO where
  nIn:=5
  nOut:=2
  nScr:=0
  bufs:=["x","w","r","dx","dw","y"]
  ity:=fun _=>.float
  iarity:=Ports.inputSize h
  ibuf:=Ports.inputRole
  oty:=fun _=>.float
  oarity:=Ports.outputSize h backward
  obuf:=Ports.outputRole backward
  obuf_mem:=by intro o;fin_cases o <;> cases backward <;> simp [Ports.outputRole] <;> decide
  sarity:=fun i=>Fin.elim0 i
  sbuf:=fun i=>Fin.elim0 i
  iwin:=fun _ _ _ _ _ j=>j.val
  imask:=fun i _ _ _ _ _=>Ports.active h backward i
  owin:=fun _ _ _ _ _ j=>j.val
  omask:=fun _ _ _ _ _ _=>True
  swin:=fun i=>Fin.elim0 i
  smask:=fun i=>Fin.elim0 i

def program (h : Call) (backward : Bool) : PreparedAlgorithmProgramIO where
  toAlgorithmMemoryIO:=memory h backward
  schedule:=fun _=>schedule h backward

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealIO

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.NativeIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite NativeGraph
noncomputable section

def forwardJob (M : Algebra Term) (h : Call) (epsilon : Term) (pid : Nat) : Invocation Term where
  kernel:=Kernels.forwardRuntime h
  backend:=engine M
  arguments:=[⟨"eps",.real,[],fun _=>epsilon⟩]
  clearRegisters:=Bool.true
  pids:=fun _=>pid
  numPids:=some (fun _=>h.rows)

def backwardJob (M : Algebra Term) (h : Call) (weight : Bool) (pid count : Nat) : Invocation Term where
  kernel:=if weight then Kernels.weightBackward h 256 else Kernels.backward h
  backend:=engine M
  clearRegisters:=Bool.true
  pids:=fun _=>pid
  numPids:=some (fun _=>count)

def schedule (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term) : List (Invocation Term) :=
  if backward then
    if h.rows=0 then [] else
      ((List.finRange (Geometry.featureCount h)).map (fun i=>backwardJob (algebra R .weight) h Bool.true i.val (Geometry.featureCount h))) ++
      ((List.finRange h.rows).map (fun i=>backwardJob (algebra R .input) h Bool.false i.val h.rows))
  else (List.finRange h.rows).map (fun i=>forwardJob (algebra R .forward) h epsilon i.val)

def program (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term) : ProgramIO Term Term where
  inputs:=Ports.inputs h
  outputs:=Ports.outputs h backward
  buffers:=["x","w","r","dx","dw","y"]
  read:=fun i=>Fin.cases (fun _ j=>j.1.val)
    (Fin.cases (fun _ j=>j.1.val) (Fin.cases (fun _ j=>j.1.val)
      (Fin.cases (fun _ j=>j.1.val) (Fin.cases (fun _ j=>j.1.val) (fun i=>Fin.elim0 i))))) i
  inputMask:=fun i _ _=>Ports.active h backward ⟨i.val,by simpa [Ports.inputs] using i.isLt⟩
  write:=by
    cases backward <;> exact fun o=>Fin.cases (fun _ j=>j.1.val)
      (Fin.cases (fun _ j=>j.1.val) (fun i=>Fin.elim0 i)) o
  outputMask:=fun _ _ _=>True
  scratch:=[]
  scratchWrite:=fun i=>Fin.elim0 i
  scratchMask:=fun i=>Fin.elim0 i
  schedule:=fun _=>schedule R h backward epsilon
  encode:=id
  observe:=fun _ _=>id

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.NativeIO

/-! Candidate native IO; all original ports and backward stages are retained. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.IOPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite NativeGraph
noncomputable section

def forwardJob (M : Algebra Term) (h : Call) (epsilon : Term) (pid : Nat) : Invocation Term :=
  { NativeIO.forwardJob M h epsilon pid with kernel := Kernels.forwardRuntimeIndexCommuted h }
def backwardJob (M : Algebra Term) (h : Call) (weight : Bool) (pid count : Nat) : Invocation Term :=
  { NativeIO.backwardJob M h weight pid count with
    kernel := if weight then Kernels.weightBackward h 256 else Kernels.backwardIndexCommuted h }

def schedule (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term) : List (Invocation Term) :=
  if backward then
    if h.rows=0 then [] else
      ((List.finRange (Geometry.featureCount h)).map (fun i => backwardJob (algebra R .weight) h Bool.true i.val (Geometry.featureCount h))) ++
      ((List.finRange h.rows).map (fun i => backwardJob (algebra R .input) h Bool.false i.val h.rows))
  else (List.finRange h.rows).map (fun i => forwardJob (algebra R .forward) h epsilon i.val)

def variant (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term) :
    ProgramIO.Variant (NativeIO.program R h backward epsilon) where
  schedule := fun _ => schedule R h backward epsilon
  observe := (NativeIO.program R h backward epsilon).observe

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.IOPair

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.LogicalIO
open VeriTile Triton
noncomputable section

/-- Bijective row/feature coordinates, including the empty row grid. -/
def matrix (h : Call) : (Fin h.cells → ℝ) ≃ (Fin h.rows → Fin h.cols → ℝ) where
  toFun v row col := v (finProdFinEquiv (row,col))
  invFun v k := v k.divNat k.modNat
  left_inv := by intro v; funext k; exact congrArg v (finProdFinEquiv.apply_symm_apply k)
  right_inv := by
    intro v; funext row col
    change v ((finProdFinEquiv.symm (finProdFinEquiv (row,col))).1)
      ((finProdFinEquiv.symm (finProdFinEquiv (row,col))).2) = v row col
    rw [finProdFinEquiv.symm_apply_apply]

/-- All five inputs are retained, including inactive forward/backward payloads. -/
structure Inputs (h : Call) where
  x : Fin h.rows → Fin h.cols → ℝ
  weight : Fin h.cols → ℝ
  savedR : Fin h.rows → ℝ
  upstream : Fin h.rows → Fin h.cols → ℝ
  initialDW : Fin h.cols → ℝ
structure ForwardOutputs (h : Call) where
  values : Fin h.rows → Fin h.cols → ℝ
  savedR : Fin h.rows → ℝ
structure BackwardOutputs (h : Call) where
  inputGradient : Fin h.rows → Fin h.cols → ℝ
  weightGradient : Fin h.cols → ℝ

def inputs (h : Call) : Ports.Values ℝ h ≃ Inputs h where
  toFun v := ⟨matrix h (v 0),v 1,v 2,matrix h (v 3),v 4⟩
  invFun v := Fin.cases ((matrix h).symm v.x) (Fin.cases v.weight (Fin.cases v.savedR
    (Fin.cases ((matrix h).symm v.upstream) (Fin.cases v.initialDW (fun i => i.elim0)))))
  left_inv := by
    intro v; funext port j; fin_cases port <;> dsimp <;> try rfl
    all_goals exact congrFun ((matrix h).symm_apply_apply _) j
  right_inv := by
    intro v
    cases v with
    | mk x w r g dw =>
      change Inputs.mk ((matrix h) ((matrix h).symm x)) w r ((matrix h) ((matrix h).symm g)) dw = _
      rw [(matrix h).apply_symm_apply,(matrix h).apply_symm_apply]

def forwardOutputs (h : Call) : Ports.Results ℝ h Bool.false ≃ ForwardOutputs h where
  toFun v := ⟨matrix h (v 0),v 1⟩
  invFun v := Fin.cases ((matrix h).symm v.values) (Fin.cases v.savedR (fun i => i.elim0))
  left_inv := by
    intro v; funext port j; fin_cases port <;> dsimp <;> try rfl
    exact congrFun ((matrix h).symm_apply_apply _) j
  right_inv := by
    intro v
    cases v with
    | mk y r =>
      change ForwardOutputs.mk ((matrix h) ((matrix h).symm y)) r = _
      rw [(matrix h).apply_symm_apply]

def backwardOutputs (h : Call) : Ports.Results ℝ h Bool.true ≃ BackwardOutputs h where
  toFun v := ⟨matrix h (v 0),v 1⟩
  invFun v := Fin.cases ((matrix h).symm v.inputGradient) (Fin.cases v.weightGradient (fun i => i.elim0))
  left_inv := by
    intro v; funext port j; fin_cases port <;> dsimp <;> try rfl
    exact congrFun ((matrix h).symm_apply_apply _) j
  right_inv := by
    intro v
    cases v with
    | mk dx dw =>
      change BackwardOutputs.mk ((matrix h) ((matrix h).symm dx)) dw = _
      rw [(matrix h).apply_symm_apply]

def forwardView (h : Call) : LogicalIO (Ports.Values ℝ h) (Ports.Results ℝ h Bool.false)
    (Inputs h) (ForwardOutputs h) (RealIO.program h Bool.false).Implements where
  inputs := inputs h
  outputs := forwardOutputs h

def backwardView (h : Call) : LogicalDomainIO (Ports.Values ℝ h) (Ports.Results ℝ h Bool.true)
    (Inputs h) (BackwardOutputs h) (RealIO.program h Bool.true).ImplementsOn where
  inputs := inputs h
  outputs := backwardOutputs h
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.LogicalIO

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Reference
open VeriTile Triton FP.Structural NativeGraph
noncomputable section

/-- Mathematical target uses only finite logical channels, never source states. -/
def row {α : Type} (h : Call) (vals : Ports.Values α h) (i : Fin h.rows) (j : Fin h.cols) : α :=
  vals 0 ⟨i.val*h.cols+j.val,by simpa [Call.cells] using Native.offset_bound h i.val j.val i.isLt j.isLt⟩
def upstream {α : Type} (h : Call) (vals : Ports.Values α h) (i : Fin h.rows) (j : Fin h.cols) : α :=
  vals 3 ⟨i.val*h.cols+j.val,by simpa [Call.cells] using Native.offset_bound h i.val j.val i.isLt j.isLt⟩

def realDomain (h : Call) (backward : Bool) (vals : Ports.Values ℝ h) : Prop :=
  backward=Bool.true →
    (∀ i : Fin h.rows,vals 2 i=RealReference.inv h.eps (row h vals i)) ∧
    (h.rows=0 → ∀ j : Fin h.cols,vals 4 j=0)

def real (h : Call) (backward : Bool) (vals : Ports.Values ℝ h) : Ports.Results ℝ h backward := by
  cases backward
  · exact Fin.cases
      (fun j=>RealReference.output h.gemma h.eps
        (row h vals ⟨j.val/h.cols,(Nat.div_lt_iff_lt_mul h.positive).mpr j.isLt⟩) (vals 1)
        ⟨j.val%h.cols,Nat.mod_lt _ h.positive⟩)
      (Fin.cases (fun i=>RealReference.inv h.eps (row h vals i)) (fun i=>Fin.elim0 i))
  · exact Fin.cases
      (fun j=>RealReference.inputGradient h.gemma h.eps
        (row h vals ⟨j.val/h.cols,(Nat.div_lt_iff_lt_mul h.positive).mpr j.isLt⟩) (vals 1)
        (upstream h vals ⟨j.val/h.cols,(Nat.div_lt_iff_lt_mul h.positive).mpr j.isLt⟩)
        ⟨j.val%h.cols,Nat.mod_lt _ h.positive⟩)
      (Fin.cases (RealReference.weightGradient h.eps (row h vals) (upstream h vals)) (fun i=>Fin.elim0 i))

def nativeDomain (h : Call) (backward : Bool) (vals : ChannelValues Term (Ports.inputs h)) : Prop :=
  backward=Bool.true → h.rows=0 → ∀ j : Fin h.cols,Ports.ofNative h vals 4 j=Term.bits 0

/-- Exact independent native expressions retain arbitrary term-valued inputs,
all reduction topology/fusion choices and the selected epsilon ABI word. -/
def native (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term)
    (vals : ChannelValues Term (Ports.inputs h)) : ChannelValues Term (Ports.outputs h backward) := by
  let logical:=Ports.logical h (Ports.ofNative h vals)
  cases backward
  · exact Fin.cases
      (fun j=>(RuntimeForward.Tensor.forwardReference (algebra R .forward) h epsilon logical
        ⟨j.1.val/h.cols,(Nat.div_lt_iff_lt_mul h.positive).mpr j.1.isLt⟩
        ⟨j.1.val%h.cols,Nat.mod_lt _ h.positive⟩).read .real)
      (Fin.cases (fun i=>(RuntimeForward.Tensor.savedReference (algebra R .forward) h epsilon logical i.1).read .real)
        (fun i=>Fin.elim0 i))
  · exact Fin.cases
      (fun j=>(FPTensor.inputReference (algebra R .input) h logical
        ⟨j.1.val/h.cols,(Nat.div_lt_iff_lt_mul h.positive).mpr j.1.isLt⟩
        ⟨j.1.val%h.cols,Nat.mod_lt _ h.positive⟩).read .real)
      (Fin.cases (fun j=>if h.rows=0 then Term.bits 0 else
        (FPTensor.weightReference (algebra R .weight) h logical j.1).read .real) (fun i=>Fin.elim0 i))

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.Reference
