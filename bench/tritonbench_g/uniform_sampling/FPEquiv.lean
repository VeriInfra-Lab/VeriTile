import bench.tritonbench_g.uniform_sampling.Proofs.PublicIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.UniformSampling.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.UniformSampling
open scoped VeriTile.Spec
noncomputable section

/-- The existing IO includes all four fields, exact clamped output windows,
tail masks and untouched ABI words. Both actual sources share this interface. -/
abbrev originalIO := @KernelPublic.io
abbrev commutedIO := @KernelPublic.variant

/-- Complete original source schedule, with its existing primitive model. -/
def original {α : Type} (M : Algebra α) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) (pid : Nat) :=
  ProgramIO.Implementation.original (originalIO M y N seed offset lower upper B hB pid) (fun _ => True)

/-- Existing candidate: only the widened integer tuple-index product is
commuted. RNG, subtraction, four FP32 FMAs and four store casts are unchanged. -/
def indexCommuted {α : Type} (M : Algebra α) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) (pid : Nat) :=
  ProgramIO.Implementation.ofVariant (commutedIO M y N seed offset lower upper B hB pid) (fun _ => True)

/-- Both actual sources are safe and terminate, every active output agrees,
and both preserve the complete exterior frame. FMA availability is a primitive
instruction port; no floating rewrite, finite-value or RNG law is assumed. -/
specification index_commuted_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (fused : α → α → α → α) (primitive : M.fma .fp32=some fused)
    (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) (pid : Nat)
    (_nonempty : 0<N) (_block : B=512 ∨ B=1024)
    (_launched : pid<SourceDomain.grid N B)
    (seedABI : Random.IntegerType.i64.inRange seed=Bool.true)
    (_offsetABI : Random.IntegerType.i64.inRange offset=Bool.true) :
    original M y N seed offset lower upper B hB pid ≡[[]]
      indexCommuted M y N seed offset lower upper B hB pid :=
  ProgramIO.Implementation.of_equivalent (commutedIO M y N seed offset lower upper B hB pid)
    (KernelPublic.equivalent M fused primitive y N seed offset lower upper B hB pid seedABI)

#print_fp_assumptions index_commuted_equiv
#axiomsClean index_commuted_equiv
#stmtSurfaceSubset index_commuted_equiv ⊆
  [FP.Structural.Algebra,FP.Structural.Algebra.fma,ComputeDType,ComputeDType.fp32,IndexComputePrecision.fp32,
   RegionName,Float32Bits,Random.ArangeSizeDomain,SourceDomain.grid,
   Random.IntegerType,Random.IntegerType.i64,Random.IntegerType.inRange,
   original,indexCommuted,Spec.Equiv,ProgramIO.Implementation,ProgramIO.Implementation.programSyntax,
   originalIO,KernelPublic.io,ProgramIO.Inputs,ProgramIO.InputBindingMode.decoded,
   Spec.RuleEntry,Spec.ProgramSyntax.Statement]
end
end VeriTile.Bench.Repairs.UniformSampling.FPEquiv
