import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Runtime

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
set_option autoImplicit false

theorem finallyDevice_state {M A : Type} (rt : Runtime M) (token : L2NormBwd.Host.Context) (body : Result M A) :
    (finallyDevice rt token body).state=(rt.perform (.exit token) body.state).state := by
  cases body with
  | error why s => cases e : rt.perform (.exit token) s <;> simp [finallyDevice,Result.state,Result.bind,e]
  | ok value s => cases e : rt.perform (.exit token) s <;> simp [finallyDevice,Result.state,Result.bind,e]

theorem finallyGrad_state {M A : Type} (rt : Runtime M) (token : Bool) (body : Result M A) :
    (finallyGrad rt token body).state=(rt.perform (.exitNoGrad token) body.state).state := by
  cases body with
  | error why s => cases e : rt.perform (.exitNoGrad token) s <;> simp [finallyGrad,Result.state,Result.bind,e]
  | ok value s => cases e : rt.perform (.exitNoGrad token) s <;> simp [finallyGrad,Result.state,Result.bind,e]

/-- A successful exit restores context even when the body failed. A failed
exit retains that operation's exact state, as stated by finallyDevice_state. -/
theorem device_restored {M α β A : Type} (ops : Ops M α β) (neg : Negation α) (arith : Arithmetic α)
    (rt : Runtime M) (lib : Library ops neg arith rt) (token : L2NormBwd.Host.Context) (body : Result M A)
    (after : World M) (exit : rt.perform (.exit token) body.state=.ok () after) :
    (finallyDevice rt token body).state.device=token.previous := by
  rw [finallyDevice_state,exit]
  exact (lib.exit token body.state after exit).2

theorem grad_restored {M α β A : Type} (ops : Ops M α β) (neg : Negation α) (arith : Arithmetic α)
    (rt : Runtime M) (lib : Library ops neg arith rt) (token : Bool) (body : Result M A)
    (after : World M) (exit : rt.perform (.exitNoGrad token) body.state=.ok () after) :
    (finallyGrad rt token body).state.gradEnabled=token := by
  rw [finallyGrad_state,exit]
  exact (lib.exitNoGrad token body.state after exit).2

#axiomsClean finallyDevice_state
#axiomsClean finallyGrad_state
#axiomsClean device_restored
#axiomsClean grad_restored
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivTriton.Host
