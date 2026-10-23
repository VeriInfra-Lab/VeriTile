import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicNativeIO
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
open KernelNative (Term Word Instructions Profile Reductions Dots)
noncomputable section

/-- The complete original source grid, source FP32 X arithmetic, FP16/E5 input
storage and FP16/FP8 output selection. Runtime dimensions, signed start and
rotary flag remain unchanged; arbitrary native terms are admitted. -/
def nativeIO (R : Reductions) (D : Dots) (fp8 : Bool) (B : Nat)
    (p : KernelSource.Parameters) : ProgramIO Term Term where
  inputs := [⟨.real,[B,p.M,p.K],"X"⟩,
    ⟨if fp8 then .int else .fp16,[p.N,p.K],"W"⟩,
    ⟨if fp8 then .int else .fp16,[p.K],"RMS"⟩]
  outputs := [⟨(KernelSource.outputDType fp8).toTileDType,[B,p.M,p.N],"OUT"⟩]
  scratch := []
  buffers := ["X","W","RMS","OUT"]
  read := PublicIO.read fp8 B p
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val*(p.M*p.N)+j.2.1.val*p.N+j.2.2.1.val)
    (fun k => k.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (KernelGrid.programs B p).map (fun q => {
    (PublicIO.invocation (KernelNative.operations R D) fp8 p q) with kernel := KernelSource.kernel fp8 p })
  encode := id
  observe := fun _ _ => id

/-- Signed-byte input representation, without a floating-finiteness guard. -/
def nativeStorageDomain (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :=
  @PublicIO.domain Term fp8 B p

/-- Independent source-ordered scalar/reduction/dot expression interpreted by
its existing instruction and selected compiler profile. -/
def nativeFormula (I : Instructions) (P : Profile) (R : Reductions) (D : Dots)
    (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters)
    (xs : ChannelValues Term (PublicIO.inputs fp8 B p)) : ChannelValues Word (PublicIO.outputs fp8 B p) :=
  fun o j => mapValue (KernelNative.eval I P.schedule) (PublicIO.outputs fp8 B p)[o].dtype
    (PublicIO.formula (KernelNative.operations R D) fp8 B p xs o j)

/-- Source-to-native observation for every output of the complete launch.
This preserves the previous primitive/compiler boundaries and makes no claim
that native arithmetic equals Real or that a second kernel is optimized. -/
theorem rms_rbe_matmul_grid_native
    (I : Instructions) (P : Profile) (R : Reductions) (D : Dots)
    (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :
    (nativeIO R D fp8 B p).ObservesOn (fun _ _ => KernelNative.eval I P.schedule)
      (nativeStorageDomain fp8 B p) (nativeFormula I P R D fp8 B p) :=
  PublicIO.native_observes I P R D fp8 B p

/- I/P/R/D are existing primitive and lowering models. This organization adds
no floating transformation law and no deferred numerical experiment. -/
#stmtSurfaceSubset rms_rbe_matmul_grid_native ⊆
  [Instructions,Profile,Profile.schedule,Reductions,Dots,KernelSource.Parameters,
   nativeIO,ProgramIO.ObservesOn,KernelNative.eval,nativeStorageDomain,nativeFormula,
   Term,Word,KernelNative.instInhabitedTerm,IOChannel,MemoryIO.outputs,ProgramIO.toMemoryIO,TileIndex,IOChannel.shape,
   ProgramIO.mapOutputs._proof_1]
#specNonCircular nativeFormula avoiding [FP.Structural.exec,KernelSource.inputs,
  KernelSource.result,KernelSource.scatter,KernelGrid.launch]
#axiomsClean rms_rbe_matmul_grid_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard
