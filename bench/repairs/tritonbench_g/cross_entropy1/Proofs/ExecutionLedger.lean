import bench.repairs.tritonbench_g.cross_entropy1.Proofs.CallCompletion

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open VeriTile.Triton Source Host Physical
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- A finite exact enumeration of source slots, separately for each actual
participant. Completeness excludes dropped calls; lexicographic source slots
fix host order. States are consecutive boundaries of those exact calls. -/
structure HostLedger {σ : Type} (source : SourceLocation → SourceCall σ → Prop) where
  entries : Participant → List (SourceLocation × SourceCall σ)
  state : Participant → Nat → σ
  member : ∀ participant location call, (location,call)∈entries participant →
    location.participant=participant ∧ source location call
  complete : ∀ location call, source location call → (location,call)∈entries location.participant
  ordered : ∀ participant, (entries participant).Pairwise (fun a b => List.Lex (· < ·) a.1.slot b.1.slot)
  boundaries : ∀ participant (i : Fin (entries participant).length),
    ((entries participant)[i]).2.before.native=state participant i.val ∧
    ((entries participant)[i]).2.after.native=state participant (i.val+1)

/-- Each byte event consumes the next actual host slot. No arbitrary call on
the same runtime, stale state or skipped source position can be substituted. -/
inductive LedgerSteps {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (events : ByteEvents σ) (participant : Participant) : Nat → Nat → List ByteAddress → Prop
  | nil (index : Nat) : LedgerSteps objects ledger events participant index index []
  | cons (index last : Nat) (bound : index < (ledger.entries participant).length)
      (site : Occurrence σ) (called : selection.invoked site.call)
      (location : (objects.exact site.call called).location=((ledger.entries participant)[index]).1)
      (record : (objects.exact site.call called).call=((ledger.entries participant)[index]).2)
      (writes rest : List ByteAddress)
      (event : events.step site (ledger.state participant index) (ledger.state participant (index+1)) writes)
      (tail : LedgerSteps objects ledger events participant (index+1) last rest) :
      LedgerSteps objects ledger events participant index last (writes++rest)

theorem LedgerSteps.trace {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger events participant first last writes}
    (steps : LedgerSteps objects ledger events participant first last writes) :
    ByteTrace events (ledger.state participant first) (ledger.state participant last) writes := by
  induction steps with
  | nil index => exact .nil _
  | cons index last bound site called location record writes rest event tail ih =>
    exact .cons site _ _ _ writes rest event ih

end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- The returned coordinate's local store completes in its actual call, then
all remaining host slots execute consecutively to this participant's return. -/
structure ReturnPath {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (participant : Participant)
    {layout : Layout σ} {observe events readWord expression state tensor index}
    (binding : LinkedReturn (selection:=selection) layout observe events objects.registry readWord expression state tensor index) where
  position : Fin (ledger.entries participant).length
  location : (objects.exact binding.site.call binding.active.1).location=((ledger.entries participant)[position]).1
  record : (objects.exact binding.site.call binding.active.1).call=((ledger.entries participant)[position]).2
  returned : state=ledger.state participant (ledger.entries participant).length
  boundaryStable : layout.buffer binding.site.call.after tensor.region=layout.buffer binding.stored tensor.region
  returnStable : layout.buffer state tensor.region=layout.buffer binding.site.call.after tensor.region
  localOutside : ∀ tick,
    (objects.image binding.site.call).tick binding.site.coordinate binding.site.instruction+1≤tick →
    tick < (objects.image binding.site.call).length → ∀ byte : Fin 4,
    ((layout.buffer binding.stored tensor.region).device,
      (layout.buffer binding.stored tensor.region).base+4*tensor.index index+byte.val)∉(objects.image binding.site.call).writes tick
  writes : List ByteAddress
  steps : LedgerSteps objects ledger events participant (position.val+1) (ledger.entries participant).length writes
  outside : ∀ byte : Fin 4,
    ((layout.buffer binding.site.call.after tensor.region).device,
      (layout.buffer binding.site.call.after tensor.region).base+4*tensor.index index+byte.val)∉writes

def ReturnPath.completion {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger participant}
    {layout : Layout σ} {observe events readWord expression state tensor index}
    {binding : LinkedReturn (selection:=selection) layout observe events objects.registry readWord expression state tensor index}
    (path : ReturnPath objects ledger participant binding) : CallCompletion objects binding where
  boundaryStable := path.boundaryStable
  returnStable := path.returnStable
  localOutside := path.localOutside
  writes := path.writes
  tail := by
    have boundary := (objects.exact binding.site.call binding.active.1).matching.after
    rw [path.record] at boundary
    have ends := (ledger.boundaries participant path.position).2
    exact (boundary.trans ends).symm ▸ (path.returned.symm ▸ path.steps.trace)
  outside := path.outside

#axiomsClean LedgerSteps.trace
#axiomsClean ReturnPath.completion
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
