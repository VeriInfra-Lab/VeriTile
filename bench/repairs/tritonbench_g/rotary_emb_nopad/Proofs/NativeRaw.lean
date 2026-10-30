import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.KernelInterface
import bench.repairs.tritonbench_g.rotary_emb_nopad.Kernels
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.TypedNative

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.FPEquiv
open VeriTile.Triton FP.Structural
open Checked
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 2000000

/-- Successful complete-source equivalence to the independent selected native
FP32 graph. Uncached multiplication order and FMA contraction are preserved.
There is no historical optimization pair for this variant. Primitive semantics
are the model boundary, with no numerical obligations or finite-word guards. -/
theorem rotary_raw_uncached_fp32 (rest : Algebra Native.Node)
    (p : Native.Instructions) (g : KernelInterface.Geometry)
    (heads : ∀ key,0<g.request.heads key)
    (s : State Native.Node) :
    ∃ t,exec (Typed.algebra rest (Native.ops p Bool.false)) (Kernels.uncached (Typed.uLayout g.request)) s=some t ∧
      t.mem=(Typed.uGraphTarget p g.request s).mem ∧
      (∀ key odd i,Typed.ULive g.request s key i →
        t.mem (Typed.uRegion (Typed.uLayout g.request) key) (Typed.uOffset (Typed.uLayout g.request) s key odd i)=
          Typed.uGraphValue p g.request s key odd i) ∧
      (∀ r o,Typed.UOutside g.request s r o → t.mem r o=s.mem r o) := by
  simpa only [Typed.u_target_graph,Typed.u_value_graph] using
    Typed.uncached_raw_successful rest (Native.ops p Bool.false) g.request heads s

/-- The cached kernel has a distinct selected native graph and store order.
The statement includes every typed cell, early return, and unchanged input K. -/
theorem rotary_raw_cached_fp32 (rest : Algebra Native.Node)
    (p : Native.Instructions) (g : KernelInterface.Geometry) (s : State Native.Node)
    (token : s.pids 1<g.tokens)
    (past : 0≤Typed.cPast (Typed.cLayout g.request) s)
    (block : 0≤Typed.cBlock (Typed.cLayout g.request) s) :
    ∃ t,exec (Typed.algebra rest (Native.ops p Bool.true)) (Kernels.cached (Typed.cLayout g.request)) s=some t ∧
      t.mem=(Typed.cGraphTarget p g.request s).mem ∧
      (∀ key odd i,s.pids 0<g.request.qHeads → Typed.cActive (Typed.cLayout g.request) s key i →
        t.mem (Typed.cRegion (Typed.cLayout g.request) key) (Typed.cOffset (Typed.cLayout g.request) s key odd i)=
          Typed.cGraphValue p g.request s key odd i) ∧
      (∀ r o,Typed.COutside g.request s r o → t.mem r o=s.mem r o) ∧
      (∀ o,t.mem "K" o=s.mem "K" o) := by
  simpa only [Typed.c_target_graph,Typed.c_value_graph] using
    Typed.cached_raw_successful rest (Native.ops p Bool.true) g.request s token past block


#specNonCircular Typed.uGraphTarget avoiding [Kernels.uncached,Typed.uncachedExecution]
#specNonCircular Typed.cGraphTarget avoiding [Kernels.cached,Typed.cachedExecution]
#axiomsClean rotary_raw_uncached_fp32
#axiomsClean rotary_raw_cached_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.FPEquiv
