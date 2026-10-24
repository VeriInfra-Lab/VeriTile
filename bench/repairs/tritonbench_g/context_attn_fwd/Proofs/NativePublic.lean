import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeIOComplete
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.KernelProfile

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativePublic
open VeriTile.Triton FP.Structural NativeGraph NativeFormula
noncomputable section

def io (p : Layout) (r : Regions) (scale : ℝ) (jobs : List Launch) : ArrayProgramIO Expr Expr :=
  NativeIO.io p r (backend (NativeSM90.plan p.BM p.D))
    (dots.support (scalar (NativeSM90.plan p.BM p.D)) p) scale jobs

def observe (p : Layout) (ops : ScalarOps) (owner : Owner) : Expr → Word :=
  fun expression => evaluate ops (NativeSM90.plan p.BM p.D) expression owner

def observer (p : Layout) (r : Regions) (scale : ℝ) (jobs : List Launch)
    (ops : ScalarOps) (owner : Owner) :
    (o : Fin (io p r scale jobs).outputs.length) →
      TileIndex (io p r scale jobs).outputs[o].shape → Expr → Word :=
  fun _ _ => observe p ops owner

/-- Independent ordered array recurrence observed under the same fixed SM90
primitive plan. All FP32 payloads and every native consumer remain admitted. -/
def formula (p : Layout) (r : Regions) (scale : ℝ) (jobs : List Launch)
    (ops : ScalarOps) (owner : Owner) (v : ArrayValues Expr (NativeIO.inputs r)) :
    ChannelValues Word (NativeIO.outputs p r jobs.length) :=
  fun o j => mapValue (observe p ops owner) (NativeIO.outputs p r jobs.length)[o].dtype
    (NativeIO.formula p r (backend (NativeSM90.plan p.BM p.D))
      (dots.support (scalar (NativeSM90.plan p.BM p.D)) p) scale jobs v o j)

theorem complete (p : Layout) (r : Regions) (scale : ℝ) (jobs : List Launch)
    (positive : 0 < p.BN) (ops : ScalarOps) (owner : Owner) :
    (io p r scale jobs).Observes (observer p r scale jobs ops owner)
      (formula p r scale jobs ops owner) :=
  ArrayProgramIO.Observes.of_implements
    (NativeIOComplete.complete p r (backend (NativeSM90.plan p.BM p.D))
      (dots.support (scalar (NativeSM90.plan p.BM p.D)) p) scale jobs positive)
    (fun _ _ => observe p ops owner)

#specNonCircular formula avoiding [FP.Structural.exec,FP.Structural.run,
  NativeArrayBridge.applyWrites,FPReference.inputs,kernel]
#axiomsClean complete
end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativePublic
