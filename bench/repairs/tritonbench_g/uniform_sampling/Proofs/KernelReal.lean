import bench.repairs.tritonbench_g.uniform_sampling.Proofs.KernelMath
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.KernelMemory

namespace VeriTile.Bench.Repairs.UniformSampling.KernelReal
open VeriTile.Triton FP.Structural KernelExecution KernelMath
open VeriTile.Bench.TritonBenchG.UniformSampling
noncomputable section

/-- Successful complete kernel execution in the Real interpretation. The
canonical random decoder is retained. This result is not FP32 exactness to an
ideal uniform fraction; special ABI words remain covered operationally by FP. -/
theorem uniform_sampling_kernel_correctness (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits) (lowerValue upperValue : ℝ)
    (lowerDenotation : Denotes lower lowerValue) (upperDenotation : Denotes upper upperValue)
    (B : Nat) (hB : Random.ArangeSizeDomain B)
    (pid : Nat) (s : State ℝ) (nonempty : 0<N) (extent : N≤2^34) (block : B=512 ∨ B=1024)
    (launched : pid<SourceDomain.grid N B) (seedABI : Random.IntegerType.i64.inRange seed=Bool.true)
    (_offsetABI : Random.IntegerType.i64.inRange offset=Bool.true)
    (capacity : RegionBounds) (bounds : N≤capacity y) :
    Spec.Real (
      (∀ f : Fin 4,∀ i : Fin B,
        (Bounds.counterBits pid B i.val).toInt=((pid*B+i.val : Nat) : Int) ∧
        (Bounds.outputBits pid B i.val f.val).toInt=((KernelOffsets.output pid B f.val i.val : Nat) : Int) ∧
        min (KernelOffsets.output pid B f.val i.val) (N-1)<capacity y) ∧
      ∃ t,FP.Structural.exec realAlgebra (kernel y N seed offset lower upper B hB) (entry pid s)=some t ∧
        (∀ f : Fin 4,∀ i : Fin B,KernelOffsets.output pid B f.val i.val<N →
          t.mem y (KernelOffsets.output pid B f.val i.val)=
            Cell.mk .real (expected seed offset lowerValue upperValue pid B f i)) ∧
        (∀ r a,KernelStores.Outside y N pid B r a → t.mem r a=s.mem r a)) := by
  have positive : 0<B := by rcases block with rfl|rfl <;> decide
  refine ⟨fun f i => ⟨(KernelMemory.native_exact N pid B block extent launched f i).1,
    (KernelMemory.native_exact N pid B block extent launched f i).2,
    (KernelMemory.addresses y N pid B capacity nonempty bounds f i).2.1⟩,?_⟩
  refine ⟨result realAlgebra (fun a b c => a*b+c) y N seed offset lower upper B (entry pid s),
    executes realAlgebra _ rfl Bool.false y N seed offset lower upper B hB (entry pid s) seedABI,?_,?_⟩
  · intro f i active
    have h := readback realAlgebra (fun a b c => a*b+c) y N seed offset lower upper B positive (entry pid s) f i active
    change (result realAlgebra (fun a b c => a*b+c) y N seed offset lower upper B (entry pid s)).mem
      y (KernelOffsets.output pid B f.val i.val) = Cell.mk .real
        (values realAlgebra (fun a b c => a*b+c) seed offset lower upper B (entry pid s) f (i,PUnit.unit)) at h
    rw [scalar_real seed offset lower upper B pid s f i,
      argument_eq lower lowerValue lowerDenotation,argument_eq upper upperValue upperDenotation] at h
    exact h
  · exact fun r a outside => frame realAlgebra _ y N seed offset lower upper B (entry pid s) r a outside

/-- Every requested coordinate has exactly one active program/field/lane
under the declared launch geometry. -/
abbrev output_coverage := @SourceDomain.active_coverage_unique

#specNonCircular expected avoiding [kernel,indexCommutedKernel,result]
#axiomsClean uniform_sampling_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.KernelReal
