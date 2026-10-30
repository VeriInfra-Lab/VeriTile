import bench.tritonbench_g.uniform_sampling.IO
import bench.tritonbench_g.uniform_sampling.Proofs.SourceSafety
import bench.tritonbench_g.uniform_sampling.Proofs.KernelMath
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.UniformSampling.KernelPublic
open VeriTile.Triton FP.Structural KernelExecution KernelMath
open VeriTile.Bench.TritonBenchG.UniformSampling
noncomputable section

def realTarget (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (lowerValue upperValue : ℝ) (B : Nat) (hB : Random.ArangeSizeDomain B) (pid : Nat)
    (_ : (io realAlgebra y N seed offset lower upper B hB pid).Inputs) :
    (io realAlgebra y N seed offset lower upper B hB pid).Outputs :=
  fun o => Fin.cases (fun j => expected seed offset lowerValue upperValue pid B j.1 j.2.1)
    (fun z => Fin.elim0 z) o

private theorem result_frame {α : Type} (M : Algebra α) (fused : α → α → α → α)
    (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) (pid : Nat)
    (vals : (io M y N seed offset lower upper B hB pid).Inputs) (s : State α) :
    (io M y N seed offset lower upper B hB pid).Frame vals s
      (result M fused y N seed offset lower upper B (entry pid s)) := by
  intro r a outside
  apply frame M fused y N seed offset lower upper B (entry pid s) r a
  by_cases eq : r=y
  · right
    intro f i active he
    exact (outside.1 0 (f,i,()) active).elim (fun hn => hn eq) (fun hn => hn he)
  · exact Or.inl eq

theorem implements_real (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (lowerValue upperValue : ℝ)
    (lowerDenotation : Denotes lower lowerValue) (upperDenotation : Denotes upper upperValue)
    (B : Nat) (hB : Random.ArangeSizeDomain B) (pid : Nat) (positive : 0<B)
    (seedABI : Random.IntegerType.i64.inRange seed=Bool.true) :
    (io realAlgebra y N seed offset lower upper B hB pid).Implements
      (realTarget y N seed offset lower upper lowerValue upperValue B hB pid) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb s _
    rw [show (io realAlgebra y N seed offset lower upper B hB pid).schedule vals=[_] from rfl,
      scheduleSafe_singleton]
    exact KernelSafety.trace_safe realAlgebra (fun a b c => a*b+c) rfl Bool.false
      y N seed offset lower upper B hB pid s seedABI bounds
      (fun f i active => hb.2.2.1 0 (f,i,()) active)
  · intro vals s _
    refine ⟨result realAlgebra (fun a b c => a*b+c) y N seed offset lower upper B (entry pid s),?_,?_,
      result_frame realAlgebra (fun a b c => a*b+c) y N seed offset lower upper B hB pid vals s⟩
    · simpa only [io,runSchedule_singleton,Invocation.run,Invocation.initial,
        bindRegisters,List.foldl_nil,Option.getD_none,entry] using
        executes realAlgebra (fun a b c => a*b+c) rfl Bool.false
          y N seed offset lower upper B hB (entry pid s) seedABI
    · intro o
      fin_cases o
      intro j active
      change mapCell id ((result realAlgebra (fun a b c => a*b+c) y N seed offset lower upper B
        (entry pid s)).mem y (KernelOffsets.output pid B j.1.val j.2.1.val)) =
        Cell.mk .real (expected seed offset lowerValue upperValue pid B j.1 j.2.1)
      have read := readback realAlgebra (fun a b c => a*b+c) y N seed offset lower upper B positive
        (entry pid s) j.1 j.2.1 active
      change (result realAlgebra (fun a b c => a*b+c) y N seed offset lower upper B
        (entry pid s)).mem y (KernelOffsets.output pid B j.1.val j.2.1.val) = _ at read
      rw [read]
      change Cell.mk .real (values realAlgebra (fun a b c => a*b+c)
        seed offset lower upper B (entry pid s) j.1 (j.2.1,())) = _
      rw [scalar_real,argument_eq lower lowerValue lowerDenotation,
        argument_eq upper upperValue upperDenotation]

/-- Complete safe execution equivalence uses only the existing integer-product
identity. FP32 FMA availability is an instruction port, not a numerical law. -/
theorem equivalent {α : Type} [Inhabited α] (M : Algebra α)
    (fused : α → α → α → α) (primitive : M.fma .fp32=some fused)
    (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) (pid : Nat)
    (seedABI : Random.IntegerType.i64.inRange seed=Bool.true) :
    (io M y N seed offset lower upper B hB pid).Equivalent
      (variant M y N seed offset lower upper B hB pid) := by
  apply ProgramIO.EquivalentOn.intro
  · intro vals _ bounds hb s _
    rw [show (io M y N seed offset lower upper B hB pid).schedule vals=[_] from rfl,
      scheduleSafe_singleton]
    exact KernelSafety.trace_safe M fused primitive Bool.false
      y N seed offset lower upper B hB pid s seedABI bounds
      (fun f i active => hb.2.2.1 0 (f,i,()) active)
  · intro vals _ bounds hb s _
    rw [show (variant M y N seed offset lower upper B hB pid).schedule vals=[_] from rfl,
      scheduleSafe_singleton]
    exact KernelSafety.trace_safe M fused primitive Bool.true
      y N seed offset lower upper B hB pid s seedABI bounds
      (fun f i active => hb.2.2.1 0 (f,i,()) active)
  · intro vals _ bounds hb s _
    refine ⟨result M fused y N seed offset lower upper B (entry pid s),
      result M fused y N seed offset lower upper B (entry pid s),?_,?_,fun _ _ _ => rfl,
      result_frame M fused y N seed offset lower upper B hB pid vals s,
      result_frame M fused y N seed offset lower upper B hB pid vals s⟩
    · simpa only [io,runSchedule_singleton,Invocation.run,Invocation.initial,
        bindRegisters,List.foldl_nil,Option.getD_none,entry] using
        executes M fused primitive Bool.false y N seed offset lower upper B hB (entry pid s) seedABI
    · simpa only [variant,runSchedule_singleton,Invocation.run,Invocation.initial,
        bindRegisters,List.foldl_nil,Option.getD_none,entry] using
        executes M fused primitive Bool.true y N seed offset lower upper B hB (entry pid s) seedABI

#axiomsClean implements_real
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.UniformSampling.KernelPublic
