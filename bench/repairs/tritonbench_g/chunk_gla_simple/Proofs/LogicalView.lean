import bench.repairs.tritonbench_g.chunk_gla_simple.Kernels

/-! Coordinate-preserving library boundary. `Prepared.exact` is the ordinary
resolve_neg().clone(memory_format=torch.contiguous_format) value contract, not an arithmetic/compiler axiom.
Views may share storage, have a storage offset, or have repeated addresses.
The candidate never writes a supplied input and returns a fresh O tensor. -/
namespace VeriTile.Bench.Repairs.ChunkGlaSimple.Logical
open VeriTile.Triton

structure View (α : Type) where
  storage : Nat → α
  address : Nat → Nat
  decode : α → α

def View.read {α : Type} (v : View α) (i : Nat) := v.decode (v.storage (v.address i))

structure Inputs (α : Type) where
  history : View α
  query : View α
  key : View α
  value : View α
  gate : View α

/-- Both routines return the same dtype and logical shape, with contiguous
coordinate order and no pending negative bit. Their allocation is ordinary
Tensor-library setup. This interface states only coordinate values. -/
structure Prepared {α : Type} (v : View α) where
  cells : Nat → α
  exact : ∀ i, cells i = v.read i

structure PreparedInputs {α : Type} (a : Inputs α) where
  history : Prepared a.history
  query : Prepared a.query
  key : Prepared a.key
  value : Prepared a.value
  gate : Prepared a.gate

def Inputs.read {α : Type} (a : Inputs α) (region : RegionName) (i : Nat) : Option α :=
  if region = "q" then some (a.query.read i)
  else if region = "k" then some (a.key.read i)
  else if region = "v" then some (a.value.read i)
  else if region = "h" then some (a.history.read i)
  else if region = "g" then some (a.gate.read i)
  else none

def PreparedInputs.read {α : Type} {a : Inputs α} (p : PreparedInputs a)
    (region : RegionName) (i : Nat) : Option α :=
  if region = "q" then some (p.query.cells i)
  else if region = "k" then some (p.key.cells i)
  else if region = "v" then some (p.value.cells i)
  else if region = "h" then some (p.history.cells i)
  else if region = "g" then some (p.gate.cells i)
  else none

theorem prepared_read {α : Type} (a : Inputs α) (p : PreparedInputs a) :
    p.read = a.read := by
  funext r i
  simp [PreparedInputs.read, Inputs.read, p.query.exact, p.key.exact,
    p.value.exact, p.history.exact, p.gate.exact]

/-- Input aliases require no special rule: each logical coordinate is copied
before the launch, and the launch's only output allocation is fresh. -/
theorem same_logical_input {α : Type} (x y : View α) (hx : Prepared x) (hy : Prepared y)
    (hxy : x.read = y.read) : hx.cells = hy.cells := by
  funext i
  rw [hx.exact,hy.exact,hxy]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaSimple.Logical
