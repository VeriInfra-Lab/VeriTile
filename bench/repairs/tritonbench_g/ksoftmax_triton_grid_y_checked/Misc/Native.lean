import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.NativeIO

/-! Auxiliary native-model, empty-grid and compatibility IO results. The existing
FPEquiv namespace preserves their full declaration names for callers. The primary
source-schedule equivalence specification is in FPEquiv.lean. -/
namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

/-- Compatibility with the previously displayed shared IO judgment. -/
theorem reversed_schedule_io_equivalent {A : Type} [Inhabited A] (model : Algebra A) (op : Operation)
    (B Q K D : Nat) (hD : 0<D) (covered : K≤D) :
    (Interface.programIO model op B Q K D (programs B Q)).Equivalent
      (Interface.reversed model op B Q K D) :=
  Interface.reverse_equivalent model op B Q K D hD covered

/-- Full selected grid, all FP32 words, original scalar/pointer ABI and chosen
compiler reduction/native profile. Primitive semantics and compiler selection
remain the existing explicit boundaries; no floating rewrite law is added. -/
abbrev nativeIO := Interface.nativeIO
abbrev nativeFormula := Interface.nativeFormula

theorem kernel_native_raw (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (op : Operation) (B Q K D : Nat) (hD : 0<D) (covered : K≤D)
    (dst leftPtr rightPtr : BitVec 64) (code : RawKernel.Selected platform op Q K D dst leftPtr rightPtr) :
    (nativeIO rest platform op B Q K D dst leftPtr rightPtr code).Implements
      (nativeFormula rest platform op B Q K D dst leftPtr rightPtr code) :=
  Interface.native_implements rest platform op B Q K D hD covered dst leftPtr rightPtr code

/-- Empty selected grids need no padded-depth positivity. The shared IO
judgment includes identity on the complete memory outside the empty outputs. -/
theorem empty_kernel_raw {A : Type} [Inhabited A] (model : Algebra A) (op : Operation)
    (B Q K D : Nat) (empty : B*Q*K=0) :
    (Interface.programIO model op B Q K D (Interface.selectedRows B Q K)).Implements
      (Interface.orderedTarget model op B Q K D (fun _ _ => id) id) :=
  Interface.empty_implements model op B Q K D empty

#axiomsClean kernel_native_raw
#axiomsClean empty_kernel_raw
#axiomsClean reversed_schedule_io_equivalent
#specNonCircular nativeFormula avoiding [KernelExecution.source, KernelExecution.rows, FP.Structural.run]
#stmtSurfaceSubset kernel_native_raw ⊆
  [AddressedProgramIO.Implements, nativeIO, nativeFormula, FP.Structural.Algebra, Graph.Expr, Graph.Word, Graph.instInhabitedExpr,
   KernelCompiler.Platform, Operation, RawKernel.Selected]
#stmtSurfaceSubset empty_kernel_raw ⊆
  [ProgramIO.Implements, Interface.programIO, Interface.selectedRows, Interface.orderedTarget,
   FP.Structural.Algebra, Operation, RegionName]
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.FPEquiv
