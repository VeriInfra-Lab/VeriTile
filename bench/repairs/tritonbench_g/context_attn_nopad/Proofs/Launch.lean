import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideFP
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Wrapper

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Launch
open VeriTile.Triton FP.Structural
open HostPlan FPHostSource
noncomputable section

abbrev Chunk := Nat × Nat × Nat
def chunk {c : Call} (j : Job c) : Chunk :=
  (j.1.1.val.val,j.2.1.1.val.val,j.2.2.1.val.val)

/-- One source native call's stride/argument setup and compiler/runtime outcome.
Successful setup leaves the tensor/register projection unchanged; actual kernel
execution is performed below, with constructive success proved separately.
Errors retain the runtime's reported state, including possible partial effects.
No successful whole-kernel value or memory law is supplied by this API. -/
structure API (α : Type) where
  prepare : Chunk → Grid → State α → Wrapper.StepResult (State α)
  success : ∀ base grid s t, prepare base grid s = .ok t → t = s

inductive Result (α : Type) where
  | ok (state : State α)
  | failed (chunk : Chunk) (error : String) (state : State α)
  | executionFailure

def run {α : Type} [Inhabited α] (o : FPBackend.Ops α) (api : API α)
    (c : Call) (r : Regions) : List (Job c) → Option Chunk → State α → Result α
  | [],_,s => .ok s
  | j :: rest,previous,s =>
    let prepared := if previous = some (chunk j) then Wrapper.StepResult.ok s
      else api.prepare (chunk j) (nativeGrid c j) s
    match prepared with
    | .failed error t => .failed (chunk j) error t
    | .ok t => match FP.Structural.exec (FP.ProfiledRewrite.engine o.algebra)
        (kernel c r) (atProgram c j t) with
      | none => .executionFailure
      | some next => run o api c r rest (some (chunk j)) next

def Ready {α : Type} (api : API α) (c : Call) (j : Job c)
    (previous : Option Chunk) (s : State α) : Prop :=
  previous = some (chunk j) ∨ api.prepare (chunk j) (nativeGrid c j) s = .ok s

/-- Ordered trace of completed programs and the first failed native call.
It retains both each prior successful execution and the actual error state;
error outcomes do not invent a rollback of earlier output writes. -/
inductive FailedAt {α : Type} [Inhabited α] (o : FPBackend.Ops α) (api : API α)
    (c : Call) (r : Regions) :
    List (Job c) → Option Chunk → State α → Chunk → String → State α → Prop
  | here {j rest previous s error t}
      (different : previous ≠ some (chunk j))
      (failed : api.prepare (chunk j) (nativeGrid c j) s = .failed error t) :
      FailedAt o api c r (j :: rest) previous s (chunk j) error t
  | later {j rest previous s next base error t}
      (ready : Ready api c j previous s)
      (executed : FP.Structural.exec (FP.ProfiledRewrite.engine o.algebra)
        (kernel c r) (atProgram c j s) = some next)
      (tail : FailedAt o api c r rest (some (chunk j)) next base error t) :
      FailedAt o api c r (j :: rest) previous s base error t

def Contract {α : Type} [Inhabited α] (o : FPBackend.Ops α) (api : API α)
    (c : Call) (r : Regions) (jobs : List (Job c)) (previous : Option Chunk) (s : State α) : Prop :=
  match run o api c r jobs previous s with
  | .ok t => runPrograms o.algebra c r jobs s = some t
  | .failed base error t => FailedAt o api c r jobs previous s base error t
  | .executionFailure => False

/-- Total ordinary API outcome classification. The actual typed source cannot
produce executionFailure: its success is constructed, not required by an API.
Successful native setup calls compose to exactly the existing complete stream. -/
theorem run_correct {α : Type} [Inhabited α] (o : FPBackend.Ops α) (api : API α)
    (c : Call) (r : Regions) (jobs : List (Job c)) (previous : Option Chunk) (s : State α) :
    Contract o api c r jobs previous s := by
  induction jobs generalizing previous s with
  | nil => rfl
  | cons j rest ih =>
    obtain ⟨next,hexec,_⟩ := FPSymbolic.complete o c.layout r c.scale
      (by change 0 < (128 : Nat); omega) (atProgram c j s)
    have he : FP.Structural.exec (FP.ProfiledRewrite.engine o.algebra)
        (kernel c r) (atProgram c j s) = some next := hexec
    have good (ready : Ready api c j previous s)
        (hprepared : (if previous = some (chunk j) then Wrapper.StepResult.ok s
          else api.prepare (chunk j) (nativeGrid c j) s) = .ok s) :
        Contract o api c r (j :: rest) previous s := by
      have tail := ih (some (chunk j)) next
      cases ht : run o api c r rest (some (chunk j)) next with
      | executionFailure =>
        simp only [Contract,ht] at tail
      | failed base error t =>
        simp only [Contract,run,hprepared,he,ht]
        exact .later ready he (by simpa only [Contract,ht] using tail)
      | ok t =>
        simp only [Contract,run,hprepared,he,ht]
        simp only [runPrograms,he,Option.bind_some]
        simpa only [Contract,ht] using tail
    by_cases same : previous = some (chunk j)
    · exact good (Or.inl same) (by simp [same])
    · cases hp : api.prepare (chunk j) (nativeGrid c j) s with
      | failed error t =>
        simp only [Contract,run,if_neg same,hp]
        exact .here same hp
      | ok t =>
        have ht := api.success (chunk j) (nativeGrid c j) s t hp
        subst t
        exact good (Or.inr hp) (by simp [same,hp])

theorem successful_stream {α : Type} [Inhabited α] (o : FPBackend.Ops α) (api : API α)
    (c : Call) (r : Regions) (jobs : List (Job c)) (previous : Option Chunk) (s t : State α)
    (h : run o api c r jobs previous s = .ok t) :
    runPrograms o.algebra c r jobs s = some t := by
  simpa only [Contract,h] using run_correct o api c r jobs previous s

#print axioms run_correct
end
end VeriTile.Bench.Repairs.ContextAttnNopad.Launch
