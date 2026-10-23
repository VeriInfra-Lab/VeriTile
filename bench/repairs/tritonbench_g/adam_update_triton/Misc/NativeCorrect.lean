import bench.repairs.tritonbench_g.adam_update_triton.Proofs.IONative

/-! Auxiliary single-source native correctness, retained with its historical
full declaration names and statements. This is not a two-program equivalence.
The public FPEquiv entry proves the separate complete-grid program pair. -/

/-! Source-to-native IO relation for the observed FP32 instruction graph.
The target retains FMA order, unordered !=, ordered > and all FP32 bit patterns.
Primitive instructions are model ports; no numerical rewrite is assumed. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term)
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- The same source, runtime scalar registers, input/output windows and mask as
Correct.lionIO. Input words enter as graph leaves; only floating output payloads
are observed, retaining each stored cell's dtype and all discrete payloads. -/
def nativeIO (native : NativeWords.Instructions) (params : Scalars Native.Word)
    (total n B pid : Nat) : ProgramIO Native.Word (Term Native.Word) :=
  Interface.programIO (Native.semantics (NativeWords.ports native)) (Native.arguments params)
    Term.input (Native.interpret (NativeWords.ports native)) total n B pid

/-- Both native output arrays implement the independent ordered recurrence.
The shared judgment includes successful source execution, actual active-access
safety and the exact exterior cell frame. -/
theorem lion_kernel_native_equiv (native : NativeWords.Instructions)
    (params : Scalars Native.Word) (total n B pid : Nat) :
    nativeIO native params total n B pid ⊨ fun values => Interface.result
      (fun j => Native.reference (NativeWords.ports native) params .parameter
        (Interface.parameter values j) (Interface.gradient values j) (Interface.momentum values j))
      (fun j => Native.reference (NativeWords.ports native) params .momentum
        (Interface.parameter values j) (Interface.gradient values j) (Interface.momentum values j)) := by
  exact NativeExecution.io_equiv native params total n B pid

#specNonCircular Native.reference avoiding [Native.interpret, Native.sourceScalar, Tile.scalar, Tile.sourceRun]
#axiomsClean lion_kernel_native_equiv
#stmtSurfaceSubset lion_kernel_native_equiv ⊆
  [nativeIO, ProgramIO.Implements, ProgramIO.Inputs, Scalars, Native.Word,
   Native.reference, NativeWords.Instructions, NativeWords.ports,
   Output.parameter, Output.momentum, Interface.result,
   Interface.parameter, Interface.gradient, Interface.momentum, Term,
   VeriTile.Bench.TritonBenchG.SwigluBackward.Observed.instInhabitedTerm]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.FPEquiv
