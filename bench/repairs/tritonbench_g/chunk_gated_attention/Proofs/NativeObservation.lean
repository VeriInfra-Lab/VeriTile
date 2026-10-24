import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeImplements
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeMerge
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.ScalarObservation

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section

def innerObserver (c : HostFP.Inner) (actual : ScalarNative.Primitives) :
    (o : Fin (innerIO c).outputs.length) → TileIndex (innerIO c).outputs[o].shape →
      NativeGraph.Expr → BitVec 32 := fun _ _=>ScalarNative.eval actual
def prefixObserver (c : HostFP.Cumulative) (actual : ScalarNative.Primitives) :
    (o : Fin (prefixIO c).outputs.length) → TileIndex (prefixIO c).outputs[o].shape →
      NativeGraph.Expr → BitVec 32 := fun _ _=>ScalarNative.eval actual

def innerNative (c : HostFP.Inner) (reference : ScalarNative.Primitives)
    (x : (innerIO c).Inputs) : (innerIO c).ObservedOutputs (BitVec 32) :=
  (innerIO c).mapOutputs (fun _ _=>ScalarNative.eval reference) (innerGraph c x)
def prefixNative (c : HostFP.Cumulative) (reference : ScalarNative.Primitives)
    (x : (prefixIO c).Inputs) : (prefixIO c).ObservedOutputs (BitVec 32) :=
  (prefixIO c).mapOutputs (fun _ _=>ScalarNative.eval reference) (prefixGraph c x)

theorem observation_same (actual reference : ScalarNative.Primitives) (laws : ScalarLaws actual reference) :
    ScalarNative.eval actual=ScalarNative.eval reference := funext (scalar_eval_eq actual reference laws)

theorem inner_observes (c : HostFP.Inner) (actual reference : ScalarNative.Primitives)
    (laws : ScalarLaws actual reference) :
    (innerIO c).Observes (fun _ _=>ScalarNative.eval actual) (innerNative c reference) := by
  rw [observation_same actual reference laws]
  exact ProgramIO.Observes.of_implements (inner_implements c) (fun _ _=>ScalarNative.eval reference)

theorem prefix_observes (c : HostFP.Cumulative) (actual reference : ScalarNative.Primitives)
    (laws : ScalarLaws actual reference) :
    (prefixIO c).Observes (fun _ _=>ScalarNative.eval actual) (prefixNative c reference) := by
  rw [observation_same actual reference laws]
  exact ProgramIO.Observes.of_implements (prefix_implements c) (fun _ _=>ScalarNative.eval reference)

#specNonCircular innerNative avoiding [FP.Structural.exec,FP.Structural.run,runSchedule,HostFP.batches]
#specNonCircular prefixNative avoiding [FP.Structural.exec,FP.Structural.run,runSchedule,HostFP.batches]
#axiomsClean inner_observes
#axiomsClean prefix_observes
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
