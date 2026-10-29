import VeriTile.Triton.Memory.AccessHelpers

/-! Register-only portions of the actual source require no allocation bounds.
The local certificate is constructed from each concrete AST, then translated
into the framework's trace-safety predicate. Loads and stores have no certificate
constructor and are handled by separate active-address proofs. -/
namespace VeriTile.Triton.Memory.RegisterSafety
open VeriTile.Triton Memory.AccessHelpers

inductive RegisterOnly : Stmt → Prop
  | assign {dt shape name} {e : Op dt shape}
      (safe : ∀ bounds s, e.SafeAt bounds s) : RegisterOnly (.assign dt shape name e)
  | branch {c : Op .bool []} {yes no : List Stmt}
      (safe : ∀ bounds s, c.SafeAt bounds s)
      (hy : ∀ st ∈ yes, RegisterOnly st) (hn : ∀ st ∈ no, RegisterOnly st) :
      RegisterOnly (.ifThenElse c yes no)
  | optional {c : Op .bool []} {body : List Stmt}
      (safe : ∀ bounds s, c.SafeAt bounds s) (hb : ∀ st ∈ body, RegisterOnly st) :
      RegisterOnly (.ifThen c body)

@[simp] theorem assign_iff {dt shape name} {e : Op dt shape} :
    RegisterOnly (.assign dt shape name e) ↔ ∀ bounds s, e.SafeAt bounds s := by
  constructor
  · intro h; cases h with | assign he => exact he
  · exact RegisterOnly.assign

@[simp] theorem branch_iff {c : Op .bool []} {yes no : List Stmt} :
    RegisterOnly (.ifThenElse c yes no) ↔ (∀ bounds s, c.SafeAt bounds s) ∧
      (∀ st ∈ yes, RegisterOnly st) ∧ (∀ st ∈ no, RegisterOnly st) := by
  constructor
  · intro h; cases h with | branch hc hy hn => exact ⟨hc, hy, hn⟩
  · rintro ⟨hc, hy, hn⟩; exact RegisterOnly.branch hc hy hn

@[simp] theorem optional_iff {c : Op .bool []} {body : List Stmt} :
    RegisterOnly (.ifThen c body) ↔ (∀ bounds s, c.SafeAt bounds s) ∧ (∀ st ∈ body, RegisterOnly st) := by
  constructor
  · intro h; cases h with | optional hc hb => exact ⟨hc, hb⟩
  · rintro ⟨hc, hb⟩; exact RegisterOnly.optional hc hb

theorem RegisterOnly.safe {st : Stmt} (h : RegisterOnly st) : ∀ bounds s, Stmt.TraceSafe bounds st s := by
  induction h with
  | assign he => intro bounds s; simpa only [Stmt.TraceSafe] using he bounds s
  | branch hc _ _ ihy ihn =>
    intro bounds s
    exact if_else_safe (hc bounds s)
      (all_safe _ s (fun st hs t => ihy st hs bounds t))
      (all_safe _ s (fun st hs t => ihn st hs bounds t))
  | optional hc _ ih =>
    intro bounds s
    exact if_safe (hc bounds s) (all_safe _ s (fun st hs t => ih st hs bounds t))

theorem list_safe (body : List Stmt) (h : ∀ st ∈ body, RegisterOnly st) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds body s := all_safe body s (fun st hs => (h st hs).safe bounds)

@[simp] theorem dynamic_offsets_safe (bounds : RegionBounds) (s : BlockState) (region : RegionName)
    (base : Op .nat []) (parent shape strides : List Nat) (a b : Op .nat []) :
    (Op.makeBlockPtrDynOffsets region base parent shape strides [a, b]).SafeAt bounds s ↔
      base.SafeAt bounds s ∧ a.SafeAt bounds s ∧ b.SafeAt bounds s := by
  rw [Op.SafeAt.eq_def]
  simp only [List.forall_mem_cons]
  simp

@[simp] theorem dynamic_advance_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (ptr : Op .blockPtr shape) (a b : Op .int []) :
    (Op.advanceBlockPtrDyn ptr [a, b]).SafeAt bounds s ↔
      ptr.SafeAt bounds s ∧ a.SafeAt bounds s ∧ b.SafeAt bounds s := by
  rw [Op.SafeAt.eq_def]
  simp only [List.forall_mem_cons]
  simp

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Triton.Memory.RegisterSafety
