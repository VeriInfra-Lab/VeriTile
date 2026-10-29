import bench.repairs.tritonbench_g.triton_softmax.Correct
import bench.repairs.tritonbench_g.triton_softmax.Misc.NativeCorrect
import bench.repairs.tritonbench_g.triton_softmax.FPEquiv
import bench.repairs.tritonbench_g.triton_softmax.Proofs.KernelBinding

/-! Optional correspondence between Python input checks and the contract.
The complete Tensor-connected kernel results remain the primary entries. -/
namespace VeriTile.Bench.Repairs.TritonSoftmax.TensorBridge
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open Contiguous
noncomputable section

theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInput x=.ok () ↔ AdmittedArgument x := checkInput_iff x

#axiomsClean python_checks_iff_preconditions
end
end VeriTile.Bench.Repairs.TritonSoftmax.TensorBridge

namespace VeriTile.Bench.Repairs.TritonSoftmax.TensorBridge.Real
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open Contiguous
noncomputable section

/-- Independent unshifted softmax over the original logical row. -/
def reference (n : Nat) (xs : Nat → ℝ) (c : Nat) : ℝ :=
  Real.exp (xs c) / ∑ j ∈ Finset.range n, Real.exp (xs j)

/-- Empty columns produce no kernel launch; nonempty rows use the literal
complete source and the Tensor's actual backing allocation. -/
def execute (x y : Object) (p : Nat) (s : BlockState) : Option BlockState :=
  if cols x.metadata=0 then some s else
    RealModel.evaluate (descriptor x) (descriptor y) (blockSize (cols x.metadata)) p s

/-- Every logical output of each valid row, actual source success, the exact
exterior-cell frame, and preservation of the original input allocation. Output
shape, layout and separate storage are ordinary kernel invocation metadata. -/
theorem softmax_kernel_correct (x y : Object)
    (hx : InputContract x.metadata) (hy : InputContract y.metadata)
    (shape : SameShape (descriptor x) (descriptor y)) (separate : x.region≠y.region)
    (p : Nat) (hp : p < rows x.metadata) (read : Nat → ℝ) (s : BlockState)
    (binding : InputMemory RealModel.access read x s) :
    ∃ t, execute x y p s=some t ∧
      (∀ c,c < cols x.metadata →
        t.readMem y.region (y.tensor.physicalIndex (p*cols y.metadata+c)) =
          reference (cols x.metadata) (logicalValue Neg.neg read x p) c) ∧
      FrameRows RealModel.access (descriptor y) [p] s t ∧
      (∀ i,t.mem x.region i=s.mem x.region i) := by
  by_cases empty : cols x.metadata=0
  · refine ⟨s,by simp [execute,empty],?_,fun _ _ _ => rfl,fun _ => rfl⟩
    intro c hc
    omega
  · have input : Inputs RealModel.access (descriptor x) (logicalValue Neg.neg read x) s := by
      intro r c hr hc
      have link := KernelBinding.source_load RealModel.access Neg.neg read x hx s binding r c hr hc
      exact link.1.trans link.2
    obtain ⟨t,run⟩ := RealModel.terminates (descriptor x) (descriptor y)
      (blockSize (cols x.metadata)) p (blockSize_bounds _).1 s
    have result := RealModel.source.row (descriptor x) (descriptor y)
      (blockSize (cols x.metadata)) p s t (logicalValue Neg.neg read x)
      shape (descriptor_injective y) (blockSize_bounds _).1 (blockSize_bounds _).2 hp input run
    refine ⟨t,by simp [execute,empty,run],?_,result.2,?_⟩
    · intro c hc
      have py : p < rows y.metadata := by change p < (descriptor y).rows; rw [shape.rows]; exact hp
      have cy : c < cols y.metadata := by change c < (descriptor y).cols; rw [shape.cols]; exact hc
      rw [KernelBinding.physical_index y hy p c py cy]
      exact result.1 c hc
    · intro i
      exact result.2 x.region i (by intro row hrow c hc equal; exact separate equal.1.symm)

#specNonCircular reference avoiding [softmax_kernel,execute,RealModel.evaluate]
#axiomsClean softmax_kernel_correct
end
end VeriTile.Bench.Repairs.TritonSoftmax.TensorBridge.Real

namespace VeriTile.Bench.Repairs.TritonSoftmax.TensorBridge.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open Contiguous
noncomputable section

/-- The fixed-four-warp native graph, with runtime-stride production layout
V1, at the original Tensor's logical words. The six scalar instruction ports
and compiler graph selection are explicit model boundaries. -/
def reference (inst : NativeGraph.Instructions) := NativeGraph.row inst 1

def negateWord (x : NativeGraph.Word) : NativeGraph.Word := x ^^^ 0x80000000

/-- Empty columns launch no program. The candidate changes exactly one integer
input-row product and retains the complete selected FP32 program. -/
def execute (inst : NativeGraph.Instructions) (candidate : Bool)
    (x y : Object) (p : Nat) (s : State NativeGraph.Word) : Option (State NativeGraph.Word) :=
  if cols x.metadata=0 then some s else
  if candidate then KernelNative.candidate inst 1 (descriptor x) (descriptor y)
      (blockSize (cols x.metadata)) p s
  else NativeGraph.evaluate inst 1 (descriptor x) (descriptor y)
      (blockSize (cols x.metadata)) p s

/-- Actual successful typed source/candidate executions have the same complete
state. Both native runs produce the same full physical state, every logical
output word agrees with the ordered graph, and all exterior cells are retained.
There are no numerical rewrite laws or whole-row success premises. -/
theorem softmax_kernel_fp32 (inst : NativeGraph.Instructions)
    (x y : Object) (hx : InputContract x.metadata) (hy : InputContract y.metadata)
    (shape : SameShape (descriptor x) (descriptor y)) (separate : x.region≠y.region)
    (p : Nat) (hp : p < rows x.metadata) (read : Nat → NativeGraph.Word)
    (s : State NativeGraph.Word) (binding : InputMemory Floating.access read x s) :
    ∃ t, execute inst Bool.false x y p s=some t ∧ execute inst Bool.true x y p s=some t ∧
      (0 < cols x.metadata → ∃ nodes,
        Floating.evaluate (NativeGraph.algebra inst 1) (descriptor x) (descriptor y)
          (blockSize (cols x.metadata)) p (NativeGraph.liftState s)=some nodes ∧
        KernelNative.candidateEvaluate (NativeGraph.algebra inst 1) (descriptor x) (descriptor y)
          (blockSize (cols x.metadata)) p (NativeGraph.liftState s)=some nodes ∧
        t=NativeGraph.projectState 1 (blockSize (cols x.metadata)) (descriptor y) p s nodes ∧
        FrameRows Floating.access (descriptor y) [p] (NativeGraph.liftState s) nodes) ∧
      (∀ c,c < cols x.metadata →
        (t.mem y.region (y.tensor.physicalIndex (p*cols y.metadata+c))).read .real =
          reference inst (cols x.metadata) (blockSize (cols x.metadata))
            (logicalValue negateWord read x p) c) ∧
      FrameRows Floating.access (descriptor y) [p] s t ∧
      (∀ i,t.mem x.region i=s.mem x.region i) := by
  by_cases empty : cols x.metadata=0
  · refine ⟨s,by simp [execute,empty],by simp [execute,empty],?_,?_,fun _ _ _ => rfl,fun _ => rfl⟩
    · intro positive
      omega
    · intro c hc
      omega
  · have input : Inputs Floating.access (descriptor x) (logicalValue negateWord read x) s := by
      intro r c hr hc
      have link := KernelBinding.source_load Floating.access negateWord read x hx s binding r c hr hc
      exact link.1.trans link.2
    obtain ⟨nodes,t,sourceRun,candidateRun,project,nativeRun,nativeCandidateRun⟩ :=
      KernelNative.complete inst 1 (descriptor x) (descriptor y)
        (blockSize (cols x.metadata)) p (blockSize_bounds _).1 s
    have nodeInput : Inputs Floating.access (descriptor x)
        (fun r c => NativeGraph.atom (logicalValue negateWord read x r c))
        (NativeGraph.liftState s) := by
      intro r c hr hc
      change NativeGraph.atom ((s.mem x.region ((descriptor x).index r c)).read .real)=_
      exact congrArg NativeGraph.atom (input r c hr hc)
    have typed := (Floating.source (NativeGraph.algebra inst 1)).row
      (descriptor x) (descriptor y) (blockSize (cols x.metadata)) p
      (NativeGraph.liftState s) nodes
      (fun r c => NativeGraph.atom (logicalValue negateWord read x r c))
      shape (descriptor_injective y) (blockSize_bounds _).1 (blockSize_bounds _).2 hp nodeInput sourceRun
    have result := (NativeGraph.source inst 1).row (descriptor x) (descriptor y)
      (blockSize (cols x.metadata)) p s t (logicalValue negateWord read x)
      shape (descriptor_injective y) (blockSize_bounds _).1 (blockSize_bounds _).2 hp input nativeRun
    refine ⟨t,by simp [execute,empty,nativeRun],by simp [execute,empty,nativeCandidateRun],
      fun _ => ⟨nodes,sourceRun,candidateRun,project,typed.2⟩,?_,result.2,?_⟩
    · intro c hc
      have py : p < rows y.metadata := by change p < (descriptor y).rows; rw [shape.rows]; exact hp
      have cy : c < cols y.metadata := by change c < (descriptor y).cols; rw [shape.cols]; exact hc
      rw [KernelBinding.physical_index y hy p c py cy]
      exact result.1 c hc
    · intro i
      exact result.2 x.region i (by intro row hrow c hc equal; exact separate equal.1.symm)

#specNonCircular reference avoiding [softmax_kernel,indexCommutedKernel,execute,NativeGraph.evaluate]
#axiomsClean softmax_kernel_fp32
end
end VeriTile.Bench.Repairs.TritonSoftmax.TensorBridge.Native
