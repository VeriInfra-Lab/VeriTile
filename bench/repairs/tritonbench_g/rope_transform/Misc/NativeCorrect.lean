import bench.repairs.tritonbench_g.rope_transform.Proofs.IOGrid
import bench.repairs.tritonbench_g.rope_transform.Proofs.RawResults
import VeriTile.Triton.Float.ObservedProgramIO

namespace VeriTile.Bench.Repairs.RopeTransform.FPEquiv
open VeriTile.Triton FP.Structural KernelContract
noncomputable section

/-- Same FP32 source/grid and arbitrary original tagged input payloads. The
input binding is identity; output words are observed separately below. -/
@[reducible] def nativeIO (cfg : FP.Config) (r : Request) : ProgramIO Tagged Tagged :=
  PublicIO.sourceIO (nativeAlgebra cfg) id id r

/-- Independent native word formula, retaining both exact FMA operand orders.
The original load casts clear arbitrary input provenance tags. -/
def formula (cfg : FP.Config) (a b c s : Tagged) (second : Bool) : Word :=
  if second then FP.fma cfg s.word a.word (FP.mul cfg c.word b.word)
  else FP.fma cfg c.word a.word (FP.mul cfg s.word b.word).neg

def target (cfg : FP.Config) (r : Request) (xs : PublicIO.Inputs Tagged r) :
    ChannelValues Word (PublicIO.outputs r) :=
  PublicIO.result (fun key row head col =>
    let lane := if col<r.half then col else col-r.half
    formula cfg (PublicIO.q xs key row head lane)
      (PublicIO.q xs key row head (lane+r.half))
      (PublicIO.coefficient xs Bool.false (row%r.sequence) lane)
      (PublicIO.coefficient xs Bool.true (row%r.sequence) lane) (decide (r.half≤col)))

/-- Complete source-to-native observation preserves the arbitrary Tagged input
domain, all word payloads, source safety, success and the full exterior frame. -/
theorem rope_forward_fp32 (cfg : FP.Config) (r : Request) :
    (nativeIO cfg r).Observes (fun _ _ x => x.word) (target cfg r) := by
  have proved := ProgramIO.Observes.of_implements
    (PublicIO.implements (nativeAlgebra cfg) (id : Tagged → Tagged) id r)
    (fun _ _ x => x.word)
  have same : (fun xs => (nativeIO cfg r).mapOutputs (fun _ _ x => x.word)
      (PublicIO.scalarTarget (nativeAlgebra cfg) id id r xs)) = target cfg r := by
    funext xs o
    fin_cases o <;> funext j
    all_goals exact scalar_native_raw cfg _ _ _ _ _
  rw [same] at proved
  exact proved

#specNonCircular formula avoiding [kernel,scalar,Grid.run,nativeAlgebra]
#specNonCircular target avoiding [kernel,scalar,Grid.run,nativeIO]
#stmtSurfaceSubset rope_forward_fp32 ⊆
  [Request,FP.Config,nativeIO,ProgramIO.Observes,target,
   Word,Tagged,VeriTile.Bench.Repairs.RopeBackwardTransform.Numerics.instInhabitedTagged,
   VeriTile.Bench.Repairs.RopeBackwardTransform.Numerics.Tagged.word,
   VeriTile.Bench.Repairs.RopeBackwardTransform.Numerics.Word,
   IOChannel,MemoryIO.outputs,ProgramIO.toMemoryIO,TileIndex,IOChannel.shape,
   ProgramIO.mapOutputs._proof_1]
-- FP.Config and the selected FMA operand order are primitive model boundaries.
-- No numerical transformation law, second kernel, or input-tag restriction.
#axiomsClean rope_forward_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RopeTransform.FPEquiv
