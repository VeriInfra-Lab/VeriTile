import bench.tritonbench_g.triton_argmax.Proofs.JITProgramPair

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open scoped VeriTile.Spec
noncomputable section

def originalGlobal (casts : Request.Casts) (kind : Request.Kind) (binding : BoundSource.Binding casts kind)
    (rest : Algebra Word) (c : GlobalConfig) (index64 : Bool) :=
  ProgramIO.Implementation.original (JITIO.nativeGlobal casts kind binding rest c index64)
    (JITNativeIO.initialized casts kind binding rest c index64)
def shiftedGlobal (casts : Request.Casts) (kind : Request.Kind) (binding : BoundSource.Binding casts kind)
    (rest : Algebra Word) (c : GlobalConfig) (index64 : Bool) :=
  ProgramIO.Implementation.ofVariant (JITProgramPair.globalVariant casts kind binding rest c index64)
    (JITNativeIO.initialized casts kind binding rest c index64)

/-- Complete first-grid and second-stage source comparison for all thirteen
requested scratch formats. The candidate uses BLOCK_SIZE=1024 shift strength
reduction with the original formula otherwise; conversion, first-NaN selection,
original-value reread, initialized index gather and both frames are retained. -/
specification global_equiv (casts : Request.Casts) (kind : Request.Kind)
    (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (g : Native.GlobalPrograms c) (positive : 0<c.block)
    (midPositive : 0<c.midBlock) (active : ∀ pid,pid<c.blocks → pid*c.block<c.size) (index64 : Bool) :
    originalGlobal casts kind binding rest c index64 ≡[[]] shiftedGlobal casts kind binding rest c index64 :=
  ProgramIO.Implementation.of_equivalentOn (JITProgramPair.globalVariant casts kind binding rest c index64)
    (JITProgramPair.global_equivalent casts kind binding rest c g positive midPositive active index64)

def originalDimension (rest : Algebra Word) (c : DimConfig) (index64 : Bool) :=
  ProgramIO.Implementation.original (JITIO.nativeDimension rest c index64) (fun _=>True)
def shiftedDimension (rest : Algebra Word) (c : DimConfig) (index64 : Bool) :=
  ProgramIO.Implementation.ofVariant (JITProgramPair.dimensionVariant rest c index64) (fun _=>True)

/-- Every dimension-grid CTA and full N loop, using the original 128-row tiles.
The source proof also retains arbitrary block sizes through its exact fallback. -/
specification dimension_equiv (rest : Algebra Word) (c : DimConfig) (nonempty : 0<c.extent) (index64 : Bool) :
    originalDimension rest c index64 ≡[[]] shiftedDimension rest c index64 :=
  ProgramIO.Implementation.of_equivalent (JITProgramPair.dimensionVariant rest c index64)
    (JITProgramPair.dimension_equivalent rest c nonempty index64)

#print_fp_assumptions global_equiv
#print_fp_assumptions dimension_equiv
#axiomsClean global_equiv
#axiomsClean dimension_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.FPEquiv
