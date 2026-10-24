import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Host

namespace VeriTile.Bench.Repairs.UniformSampling.Protocol
open VeriTile.Triton
open VeriTile.Bench.Repairs.LayernormFwdTriton Memory
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
open Host
noncomputable section
open scoped Classical

def available : Runtime := ⟨none,none,none,none,none,0,0⟩
def completed : Stops := ⟨none,none⟩

theorem succeeds (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (which : Nat) (s : State Store) (hn:0<n)
    (r : Reservation) (g : Generator)
    (hr:reserve v.device (block n) (counterCount n) (s.generators which)=.ok (r,g)) :
    ((entry api M scalars v scratch which available completed).run s).result=.ok v.handle := by
  simp [entry,Nat.ne_of_gt hn,available,completed,runtime,call,guarded,inDevice,device,
    body,reservation,dispatch,native,writeback,Action.bind,Action.pure,Except.mapError,hr]

theorem body_other_generator (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (which other : Nat) (lib : Runtime) (stops : Stops)
    (s : State Store) (hne:other≠which) :
    ((body api M scalars v scratch which lib stops).run s).state.generators other=s.generators other := by
  cases ha:lib.allocate with
  |some e=>simp [body,runtime,call,Action.bind,ha,Except.mapError]
  |none=>
    cases hr:reserve v.device (block n) (counterCount n) (s.generators which) with
    |error e=>simp [body,runtime,call,Action.bind,ha,reservation,hr,Except.mapError]
    |ok pair=>
      rcases pair with ⟨r,g⟩
      cases hp:lib.reserveReturn <;> cases hn:stops.native <;> cases hc:stops.writeback <;>
        simp [body,runtime,call,Action.bind,ha,reservation,hr,hp,dispatch,native,writeback,
          hn,hc,Action.pure,Except.mapError,Function.update_of_ne hne]

theorem all_outcomes_other_generator (api : API α Store Byte) (M : Arithmetic α) (scalars : Scalars)
    (v : View n) (scratch : Scratch n) (which other : Nat) (lib : Runtime) (stops : Stops)
    (s : State Store) (hne:other≠which) :
    ((entry api M scalars v scratch which lib stops).run s).state.generators other=s.generators other := by
  by_cases hz:n=0
  · simp [entry,hz,Action.pure]
  · cases he:lib.extension <;> cases hi:lib.enter <;> cases hl:lib.leave <;>
      simp [entry,hz,runtime,call,guarded,inDevice,device,Action.bind,he,hi,hl,Except.mapError]
    all_goals
      exact body_other_generator api M scalars v scratch which other lib stops
        ⟨s.memory,s.generators,v.device⟩ hne

theorem reservation_return_failure_commits (api : API α Store Byte)
    (v : View n) (scratch : Scratch n) (which code : Nat) (s : State Store)
    (r : Reservation) (g : Generator)
    (hr:reserve v.device (block n) (counterCount n) (s.generators which)=.ok (r,g)) :
    ((reservation api v scratch which (some code)).run s).state.generators=
      Function.update s.generators which g ∧
    ((reservation api v scratch which (some code)).run s).result=.error (.api .admission code) := by
  simp [reservation,hr]

theorem supplied_selected (defaults : Nat→Nat) (owner supplied : Nat) :
    selected defaults owner (some supplied)=supplied := rfl

theorem default_selected (defaults : Nat→Nat) (owner : Nat) :
    selected defaults owner none=defaults owner := rfl

#axiomsClean succeeds
#axiomsClean all_outcomes_other_generator
#axiomsClean reservation_return_failure_commits
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.Protocol
