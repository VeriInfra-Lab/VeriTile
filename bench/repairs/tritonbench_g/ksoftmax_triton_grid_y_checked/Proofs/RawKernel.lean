import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelExecution
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Words

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.RawKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

def run {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (B Q K D : Nat) (s : State α) : Option (State α) :=
  if B*Q*K=0 then some s else KernelExecution.rows model op B Q K D (programs B Q) s

theorem complete {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (B Q K D : Nat) (hD : 0<D) (s : State α) :
    ∃ final, run model op B Q K D s=some final ∧
      (∀ (b : Fin B) (r : Fin Q) (i : Fin D),i.val<K →
        final.mem "dst" (Native.flat Q K b r i.val)=
          gridValue model op Q K b r (fun j => (s.mem "left" j).read .real)
            (fun j => (s.mem "right" j).read .real) i) ∧
      (∀ (region : RegionName) j,region≠"dst" ∨ B*Q*K≤j → final.mem region j=s.mem region j) := by
  by_cases empty : B*Q*K=0
  · refine ⟨s,by simp [run,empty],?_,fun _ _ _ => rfl⟩
    intro b r i hi
    have h := Native.active_bound b.isLt r.isLt hi
    omega
  · refine ⟨runRows model op "dst" "left" "right" B Q K D hD (programs B Q) s,?_,?_,?_⟩
    · simpa only [run,empty,if_false] using KernelExecution.rows_success model op B Q K D hD (programs B Q) s
    · exact fun b r i hi => grid_observe model op "dst" "left" "right" B Q K D hD s b r i hi (by decide) (by decide)
    · exact rows_frame model op "dst" "left" "right" B Q K D hD (programs B Q) s

open KernelCompiler

structure Selected (platform : KernelCompiler.Platform) (op : Operation) (Q K D : Nat)
    (dst left right : BitVec 64) where
  object : Object
  lowering : Lowering
  source : object.sourceHash=sourceSHA256
  decoded : platform.decode object.bytes object.signature object.options object.primitives=some lowering
  operation : lowering.operation=op
  scalarABI : object.signature.integerArguments=(integerArguments op Q K).map (BitVec.ofNat 64)
  scalarWidths : object.signature.integerWidths=List.replicate (integerArguments op Q K).length 64
  pointers : object.signature.pointers=[dst,left,right]
  pointeeWidths : object.signature.pointeeWidths=[32,32,32]
  branch : object.signature.operation=op
  fp32 : object.signature.fp16=Bool.false
  depth : object.signature.depth=D
  depthPowerOfTwo : ∃ exponent : Nat,D=2^exponent
  profile : Graph.Binding lowering.profile K D
  reductions : ValidReductions lowering.profile D
  options : object.options={numWarps := lowering.profile.numWarps,enableFPFusion := Bool.true}
  precision : lowering.integerWidth=64 ∧ lowering.fp32Loads=Bool.true ∧
    lowering.fp32Arithmetic=Bool.true ∧ lowering.fp32Stores=Bool.true
  addresses : lowering.addresses=KernelCompiler.addresses op
  causalRepair : lowering.finalCausalGradientZero=Bool.true

/-- Tensor pointers are erased to raw ABI values; every existing compiler field is retained. -/
def Selected.ofTensor (code : KernelCompiler.Selected platform op Q K D dst left right) :
    Selected platform op Q K D dst.dataPtr left.dataPtr right.dataPtr :=
  ⟨code.object,code.lowering,code.source,code.decoded,code.operation,code.scalarABI,code.scalarWidths,code.pointers,code.pointeeWidths,code.branch,code.fp32,code.depth,code.depthPowerOfTwo,code.profile,code.reductions,code.options,code.precision,code.addresses,code.causalRepair⟩

def InputWords (op : Operation) (B Q K D : Nat)
    (initial : State Graph.Expr) (left right : Nat → Graph.Word) : Prop :=
  (∀ (b : Fin B) (r : Fin Q) (j : Fin D),
    Forward.readActive K r op.causal j.val=Bool.true →
    (initial.mem "left" (Native.flat Q K b r j.val)).read .real=
      .input 0 (Native.flat Q K b r j.val) (left (Native.flat Q K b r j.val))) ∧
  (∀ (b : Fin B) (r : Fin Q) (j : Fin D),
    op.readsRight=Bool.true → Forward.readActive K r op.causal j.val=Bool.true →
    (initial.mem "right" (rightAddress op Q K b r j.val)).read .real=
      .input 1 (rightAddress op Q K b r j.val) (right (rightAddress op Q K b r j.val)))


#axiomsClean complete
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.RawKernel
