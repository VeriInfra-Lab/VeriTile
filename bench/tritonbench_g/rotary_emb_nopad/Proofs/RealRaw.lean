import bench.tritonbench_g.rotary_emb_nopad.Proofs.KernelInterface
import bench.tritonbench_g.rotary_emb_nopad.Kernels
import bench.tritonbench_g.rotary_emb_nopad.Proofs.TypedResult

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Correct
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 2000000

/-- Complete uncached typed execution on arbitrary Real payloads. The
independent split-half formula is a*c-b*s / a*s+b*c, using the entry-memory
coefficient pair. GQA leaders own K; the complete memory frame is retained. -/
theorem rotary_raw_uncached_correct (rest : Algebra ℝ) (g : KernelInterface.Geometry)
    (heads : ∀ key,0<g.request.heads key) (s : State ℝ) :
    Spec.Real (∃ t,exec (Typed.algebra rest realOps) (Kernels.uncached (Typed.uLayout g.request)) s=some t ∧
      t.mem=(Typed.uTarget realOps g.request s).mem ∧
      (∀ key odd i,Typed.ULive g.request s key i →
        t.mem (Typed.uRegion (Typed.uLayout g.request) key) (Typed.uOffset (Typed.uLayout g.request) s key odd i)=
          Typed.uValue realOps (Typed.uLayout g.request) s s key odd i) ∧
      (∀ r o,Typed.UOutside g.request s r o → t.mem r o=s.mem r o)) :=
  Typed.uncached_raw_successful rest realOps g.request heads s

/-- Complete cached typed execution, including padded-head early return,
checked signed page/slot arithmetic, cache stores before Q, and unchanged K.
Metadata observations are separate from unrestricted Real payloads. -/
theorem rotary_raw_cached_correct (rest : Algebra ℝ) (g : KernelInterface.Geometry) (s : State ℝ)
    (token : s.pids 1<g.tokens)
    (past : 0≤Typed.cPast (Typed.cLayout g.request) s)
    (block : 0≤Typed.cBlock (Typed.cLayout g.request) s) :
    Spec.Real (∃ t,exec (Typed.algebra rest realOps) (Kernels.cached (Typed.cLayout g.request)) s=some t ∧
      t.mem=(Typed.cReference realOps (Typed.cLayout g.request) s).mem ∧
      (∀ key odd i,s.pids 0<g.request.qHeads → Typed.cActive (Typed.cLayout g.request) s key i →
        t.mem (Typed.cRegion (Typed.cLayout g.request) key) (Typed.cOffset (Typed.cLayout g.request) s key odd i)=
          Typed.cValue realOps (Typed.cLayout g.request) s key odd i) ∧
      (∀ r o,Typed.COutside g.request s r o → t.mem r o=s.mem r o) ∧
      (∀ o,t.mem "K" o=s.mem "K" o)) :=
  Typed.cached_raw_successful rest realOps g.request s token past block


#specNonCircular Typed.uValue avoiding [Kernels.uncached,Typed.uncachedExecution]
#specNonCircular Typed.cValue avoiding [Kernels.cached,Typed.cachedExecution]
#axiomsClean rotary_raw_uncached_correct
#axiomsClean rotary_raw_cached_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.Correct
