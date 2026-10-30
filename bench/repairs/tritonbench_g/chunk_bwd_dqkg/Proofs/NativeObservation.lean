import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.NativeImplements
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedWholeSafety
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.NativeTarget

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.NativeObservation
open VeriTile Triton FP.Structural Whole KernelNativeCarrier AddressBounds
noncomputable section

theorem implements (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    (IO.nativeIO isa route scale c order).Implements (NativeIOResults.formula isa route scale c) := by
  apply NativeImplements.implements isa route scale c order ho
  intro s a bounds cap
  change TraceSafe (FP.ProfiledRewrite.engine (Typed.withDot (algebra isa route scale) dotModel))
    bounds (kernel (nativeConfig c)) (Typed.prepare (nativeConfig c) s (KernelNative.nativeCTA a))
  exact TypedWholeSafety.source_safe _ _ _ s _ bounds cap

theorem observes (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    (IO.nativeIO isa route scale c order).Observes (fun _ _=>Node.eval isa route)
      (NativeTarget.words isa route scale c) :=
  ProgramIO.Observes.of_implements (implements isa route scale c order ho) (fun _ _=>Node.eval isa route)

#axiomsClean implements
#axiomsClean observes
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.NativeObservation
