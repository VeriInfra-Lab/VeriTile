import bench.repairs.tritonbench_g.chunk_gate_recurrence.Misc.NativeReference
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Misc.NativeIO
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Misc.NativePublicIO

/-! Auxiliary graph observation. The selected-object compiled kernel result is in FPEquiv. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.FPEquiv
open VeriTile.Triton FP.Structural KernelProgram KernelCarrier
noncomputable section

/-- Original FP32 primitive instruction model and complete product plan. Input
nodes remain arbitrary; only the output payload is observed as its 32 bits. -/
def nativeIO (isa : Instructions) (plan : Reduction.ProductPlan 1024) (kind : Kind)
    (c : Shape) (s : SeedStrides) (seed : Bool) (p : Program c) : ProgramIO Node Node :=
  IO.program (model isa plan kind c) kind c s seed p

/-- Source-to-native observation, not a new optimization. FMA order, casts and
all 1024 DG products are retained. Primitive semantics are model data; there
are no transformation-law or primitive-conformance experiment assumptions. -/
theorem chunk_gate_recurrence_graph_equiv (isa : Instructions)
    (plan : Reduction.ProductPlan 1024) (kind : Kind) (c : Shape) (s : SeedStrides)
    (seed : Bool) (p : Program c) (nonempty : 0<c.N) :
    (nativeIO isa plan kind c s seed p).Observes (fun _ _ => Node.value)
      (Reference.native isa plan kind c s seed p) :=
  NativePublicIO.native isa plan kind c s seed p nonempty

#axiomsClean chunk_gate_recurrence_graph_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.FPEquiv
