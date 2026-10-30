import VeriTile.Triton.Float.ArrayProgramIO
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.IntervalCases

/-! A real two-source alias case: an integer store overwrites a Nat metadata
role. Its subsequent Nat load must decode to zero, so the second source gathers
x[0], regardless of the original metadata. Bounds cover only finite accesses.
-/
namespace VeriTile.Tests.FPArrayProgramIO
open Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

def overwrite : ComputeKernel := .mk [] ["index"] [
  .store .int [] (.region "index" (.constNat 0)) (.alg (.constInt 23)) .none]

def gather : ComputeKernel := .mk ["index","x"] ["out"] [
  .assign .nat [] "i" (.alg (.load .nat (.region "index" (.constNat 0)) .none .zero)),
  .store .int [] (.region "out" (.constNat 0))
    (.alg (.load .int (.region "x" (.ref .nat [] "i")) .none .zero)) .none]

def inputPorts : List ArrayIOChannel := [⟨.nat,"index"⟩,⟨.int,"x"⟩]
def outputPorts : List IOChannel := [⟨.int,[],"index"⟩,⟨.int,[],"out"⟩]

def job {α : Type} (M : Algebra α) (k : ComputeKernel) : Invocation α :=
  {kernel:=k,backend:=M,pids:=fun _=>0}

def io {α : Type} (M : Algebra α) : ArrayProgramIO α α where
  inputs := inputPorts
  outputs := outputPorts
  scratch := []
  buffers := ["index","x","out"]
  reads := fun _ _ => [0]
  write := fun _ _ _ => 0
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => [job M overwrite,job M gather]
  encode := id
  observe := fun _ _ => id

def formula {α : Type} (xs : ArrayValues α inputPorts) : ChannelValues α outputPorts
  | ⟨0,_⟩ => fun _ => (23 : Int)
  | ⟨1,_⟩ => fun _ => xs ⟨1,by decide⟩ 0
  | ⟨n+2,h⟩ => False.elim (by simp [outputPorts] at h)

/-- Arbitrary old metadata, including a wrong tag, admits an exact snapshot. -/
theorem all_memories_bind {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    (io M).InputBinding ((io M).snapshot s) s :=
  ArrayProgramIO.snapshot_binding _ rfl _

/-- Finite backing exists even though the logical arrays have domain Nat. -/
theorem finite_bounds {α : Type} (M : Algebra α) (xs : (io M).Inputs) :
    (io M).InBounds (fun r => if r ∈ (["index","x","out"] : List RegionName) then 1 else 0) xs := by
  constructor
  · intro r hr; simp only [io] at hr; simp [hr]
  constructor
  · rintro ⟨i,hi⟩ off hoff
    change i < 2 at hi
    simp only [io, List.mem_singleton] at hoff
    subst off
    interval_cases i <;> simp [io,inputPorts]
  constructor
  · rintro ⟨o,ho⟩ j _
    change o < 2 at ho
    interval_cases o <;> simp [io,outputPorts]
  · intro k; exact k.elim0

theorem alias_gather_runs {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ final, runSchedule ((io M).schedule ((io M).snapshot s)) s = some final ∧
      final.mem "index" 0 = Cell.mk .int 23 ∧
      final.mem "out" 0 = Cell.mk .int ((s.mem "x" 0).read .int) := by
  simp [io,job,overwrite,gather,runSchedule,Invocation.run,Invocation.initial,bindRegisters,
    exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalOp_unfold,store,Option.bind,
    State.setReg,State.write,Cell.read,defaultValue]

/-- Safety uses the changed metadata in the actual predecessor memory. -/
theorem alias_gather_safe {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (bounds : RegionName → Nat) :
    ScheduleSafe bounds ((io M).schedule ((io M).snapshot s)) s ↔
      0 < bounds "index" ∧ 0 < bounds "x" ∧ 0 < bounds "out" := by
  simp [io,job,overwrite,gather,ScheduleSafe,Invocation.Safe,TraceSafe,ComputeKernel.surfaceBody,
    Safety.traceSafeList_cons,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeOp,
    Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Invocation.run,Invocation.initial,bindRegisters,exec,run,step,evalExpr,
    evalOp_unfold,store,Option.bind,State.setReg,State.write,Cell.read,defaultValue]

/-- A missing successful run cannot be hidden by an impossible capacity. -/
theorem unconditional_result {α β : Type} [Inhabited β] {p : ArrayProgramIO α β}
    {domain : p.Inputs → Prop} {f : p.Inputs → p.Outputs}
    (correct : p.ImplementsOn domain f) (xs : p.Inputs) (hd : domain xs)
    (s : State β) (bound : p.InputBinding xs s) :
    ∃ final, runSchedule (p.schedule xs) s = some final ∧
      p.OutputBinding xs (f xs) final ∧ p.Frame xs s final :=
  (correct xs hd s bound).2

#axiomsClean all_memories_bind
#axiomsClean finite_bounds
#axiomsClean alias_gather_runs
#axiomsClean alias_gather_safe
#axiomsClean unconditional_result
end
end VeriTile.Tests.FPArrayProgramIO
