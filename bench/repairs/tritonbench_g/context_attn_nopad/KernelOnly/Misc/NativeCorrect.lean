import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.PublicNativeResult

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section

/-- Original typed FP32 source and complete selected schedule. The existing
SM80 TF32 MMA, live seeds, FMA, exp2 and physical reduction plan are retained.
The logical carrier admits every original symbolic input and every word;
Q/K/V/output may alias and the metadata may exceed the launched grid. -/
abbrev nativeAttentionIO (c : Call) (r : Regions) : ProgramIO NativePhysical.Expr NativePhysical.Expr where
  inputs := IOData.inputs c r
  outputs := IOData.outputs c r
  scratch := []
  buffers := [r.query,r.key,r.value,r.start,r.sequence,r.out]
  read := fun i _ => IOData.read c r i
  inputMask := fun i _ => IOData.inputActive c r i
  write := fun i _ => match i with | ⟨0,_⟩ => IOData.address c c.o
  outputMask := fun i _ => match i with | ⟨0,_⟩ => IOData.active c
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => (FPHost.jobs c).map (fun idx => {
    kernel := kernel c r
    backend := FP.ProfiledRewrite.engine (NativePhysical.backend (NativeProfiles.plan c.q.dims)).algebra
    pids := idx.toPids
    numPids := some c.grid.toNumPids })
  encode := id
  observe := fun _ _ => id

/-- Native observation changes output payloads only. -/
def nativeAttentionObserver (c : Call) (r : Regions) (ops : NativeInstructions.ScalarOps) :
    (o : Fin (IOData.outputs c r).length) → TileIndex (IOData.outputs c r)[o].shape →
      NativePhysical.Expr → BitVec 32 := fun _ _ e =>
  NativeInstructions.evaluate ops (NativeProfiles.plan c.q.dims) e (NativeProfiles.ownerOf c.q.dims e)

/-- Independent pure online-softmax recurrence on logical input values. The
pure scatter recurrence retains the original last writer for allowed aliases;
it contains no source executor, state or output-memory proposition. -/
def nativeAttentionExpected (c : Call) (r : Regions) (referenceOps : NativeInstructions.ScalarOps)
    (xs : ChannelValues NativePhysical.Expr (IOData.inputs c r)) :
    ChannelValues (BitVec 32) (IOData.outputs c r) :=
  fun out => match out with
  | ⟨0,_⟩ => fun j =>
      let e := IOLogical.grid c r (NativePhysical.backend (NativeProfiles.plan c.q.dims))
        (FPHost.jobs c) (IOData.initialMemory c r xs) r.out (IOData.address c c.o j)
      NativeInstructions.evaluate referenceOps (NativeProfiles.plan c.q.dims) e
        (NativeProfiles.ownerOf c.q.dims e)

/-- Source-to-native observation under the original nine primitive instruction
laws. This is not an optimization or a native-equals-Real claim. -/
theorem fp_equiv (c : Call) (r : Regions)
    (startLabel : r.out≠r.start) (lengthLabel : r.out≠r.sequence)
    (actual referenceOps : NativeInstructions.ScalarOps)
    (localLaws : NativeInstructions.ScalarLaws actual referenceOps) :
    (nativeAttentionIO c r).ObservesOn (nativeAttentionObserver c r actual)
      (IOData.domain c r) (nativeAttentionExpected c r referenceOps) :=
  IOData.native_complete c r startLabel lengthLabel actual referenceOps localLaws

#axiomsClean fp_equiv
#stmtSurfaceSubset fp_equiv ⊆
  [Call,Regions,RegionName,Regions.out,Regions.start,Regions.sequence,NativeInstructions.ScalarOps,
   NativeInstructions.ScalarLaws,nativeAttentionIO,ProgramIO.ObservesOn,
   nativeAttentionObserver,IOData.domain,nativeAttentionExpected,NativePhysical.Expr,NativePhysical.instInhabitedExpr]
#specNonCircular nativeAttentionExpected avoiding [FPReference.grid,FPReference.runGrid,FP.Structural.exec,kernel]
/- ScalarLaws is the existing primitive-model boundary. Its deferred numerical
validation status is unchanged; no transformation assumption is introduced. -/
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
