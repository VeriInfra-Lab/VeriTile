import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.NativeObservation

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg
open VeriTile.Triton FP.Structural Whole KernelNativeCarrier
noncomputable section

/-- Pure word observation, shared by every declared output lane. -/
def nativeObserver (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (order : List (CTA c)) :
    (o : Fin (IO.nativeIO isa route scale c order).outputs.length) →
      TileIndex ((IO.nativeIO isa route scale c order).outputs[o]).shape → Node → Word :=
  fun _ _=>Node.eval isa route

/-- Native observation of the same complete FP32 source grid, with the original
runtime scale word, instruction/routing model, arbitrary symbolic input terms,
output dtype tags and exterior frame. This is a source-to-native recurrence
result; no second kernel, floating rewrite or exact native-to-Real equality is
introduced. Instructions and Routing are primitive model data, not deferred
numerical transformation assumptions. -/
theorem kernel_native (isa : Instructions) (route : Routing) (scale : Word)
    (c : Config) (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    (IO.nativeIO isa route scale c order).Observes (nativeObserver isa route scale c order)
      (NativeTarget.words isa route scale c) :=
  NativeObservation.observes isa route scale c order ho

#axiomsClean kernel_native
#stmtSurfaceSubset kernel_native ⊆
  [Instructions,Routing,Word,Config,CTA,allCTAs,List.Perm,
   ProgramIO.Observes,IO.nativeIO,nativeObserver,NativeTarget.words,Node,instInhabitedNode]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg
