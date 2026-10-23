import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostFrame
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SnapshotLayout
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostSnapshots

structure Window where
  (backing offset bytes : Nat)

def Window.contains (w : Window) (backing address : Nat) : Prop :=
  backing = w.backing ∧ w.offset ≤ address ∧ address < w.offset+w.bytes

def Window.Disjoint (left right : Window) : Prop :=
  ∀ backing address, left.contains backing address → ¬right.contains backing address

theorem disjoint_symm (left right : Window) (h : left.Disjoint right) : right.Disjoint left := by
  intro backing address hr hl
  exact h backing address hl hr

/-- Finite logical tensor contents after interpreting the supplied view. The
three metadata roles retain signed values and independent source dtypes. -/
def Payload (α : Host.Role → Type) : Host.Role → Type
  | .input => List (α .input)
  | .weight => List (α .weight)
  | .output => List (α .output)
  | .starts | .lengths | .adapters => List Int

def payloadLength {α : Host.Role → Type} (role : Host.Role) : Payload α role → Nat := by
  cases role <;> exact List.length

structure View (α : Host.Role → Type) (role : Host.Role) where
  values : Payload α role
  dimensions : List Nat
  extent : payloadLength role values = dimensions.prod
  bytesPerElement : Nat
  positiveWidth : 0 < bytesPerElement

/-- Ordinary successful detach/resolve_neg/contiguous-clone contract. Freshness
is byte-window disjointness, so empty allocations need no distinct pointer.
There is no condition that caller tensors have disjoint backing storage. -/
structure CloneResult {α : Host.Role → Type} (role : Host.Role) (entry : View α role) (occupied : List Window) where
  values : Payload α role
  window : Window
  preserved : values = entry.values
  extent : window.bytes = payloadLength role entry.values*entry.bytesPerElement
  fresh : ∀ previous ∈ occupied, window.Disjoint previous
  stride : Nat → Nat
  matrixLayout : ∀ rows columns, entry.dimensions = [rows,columns] →
    GridStride.Contiguous rows columns (stride 0) (stride 1)
  weightLayout : ∀ adapters columns inner, entry.dimensions = [adapters,columns,inner] →
    SnapshotLayout.WeightContiguous adapters columns inner (stride 0) (stride 1) (stride 2)
  vectorLayout : ∀ count, entry.dimensions = [count] → 1 < count → stride 0 = 1

/-- Only the local tensor-library operation is abstract here. It may fail at
any role. Program execution, source indexing, metadata validation and caller
copyback are not fields of this API. -/
structure API (α : Host.Role → Type) where
  clone : (role : Host.Role) → (entry : View α role) → (occupied : List Window) →
    Except Unit (CloneResult role entry occupied)

structure Prepared {α : Host.Role → Type} (entry : (role : Host.Role) → View α role) (caller : List Window) where
  x : CloneResult .input (entry .input) caller
  w : CloneResult .weight (entry .weight) (x.window::caller)
  y : CloneResult .output (entry .output) (w.window::x.window::caller)
  starts : CloneResult .starts (entry .starts) (y.window::w.window::x.window::caller)
  lengths : CloneResult .lengths (entry .lengths) (starts.window::y.window::w.window::x.window::caller)
  adapters : CloneResult .adapters (entry .adapters) (lengths.window::starts.window::y.window::w.window::x.window::caller)

/-- The six calls occur in the source order and all use the same entry view.
No caller write or kernel launch occurs during this preparation. -/
def prepare {α : Host.Role → Type} (api : API α) (entry : (role : Host.Role) → View α role) (caller : List Window) :
    Except Host.Stage (Prepared entry caller) := do
  let x ← (api.clone .input (entry .input) caller).mapError (fun _ => .snapshot .input)
  let w ← (api.clone .weight (entry .weight) (x.window::caller)).mapError (fun _ => .snapshot .weight)
  let y ← (api.clone .output (entry .output) (w.window::x.window::caller)).mapError (fun _ => .snapshot .output)
  let starts ← (api.clone .starts (entry .starts) (y.window::w.window::x.window::caller)).mapError (fun _ => .snapshot .starts)
  let lengths ← (api.clone .lengths (entry .lengths) (starts.window::y.window::w.window::x.window::caller)).mapError (fun _ => .snapshot .lengths)
  let adapters ← (api.clone .adapters (entry .adapters) (lengths.window::starts.window::y.window::w.window::x.window::caller)).mapError (fun _ => .snapshot .adapters)
  pure ⟨x,w,y,starts,lengths,adapters⟩

def privateWindows {α : Host.Role → Type} {entry : (role : Host.Role) → View α role} {caller : List Window}
    (p : Prepared entry caller) : List Window :=
  [p.adapters.window,p.lengths.window,p.starts.window,p.y.window,p.w.window,p.x.window]

theorem add_fresh {α : Host.Role → Type} {role : Host.Role} {entry : View α role}
    (caller current : List Window) (snapshot : CloneResult role entry (current++caller))
    (h : current.Pairwise Window.Disjoint) : (snapshot.window::current).Pairwise Window.Disjoint := by
  apply List.pairwise_cons.mpr
  exact ⟨fun w hw => snapshot.fresh w (List.mem_append_left caller hw),h⟩

theorem private_pairwise {α : Host.Role → Type} {entry : (role : Host.Role) → View α role} {caller : List Window}
    (p : Prepared entry caller) : (privateWindows p).Pairwise Window.Disjoint := by
  exact add_fresh caller _ p.adapters (add_fresh caller _ p.lengths
    (add_fresh caller _ p.starts (add_fresh caller _ p.y
      (add_fresh caller _ p.w (add_fresh caller [] p.x (by simp))))))

theorem separate_from_caller {α : Host.Role → Type} {entry : (role : Host.Role) → View α role} {caller : List Window}
    (p : Prepared entry caller) (window : Window) (hw : window ∈ privateWindows p)
    (original : Window) (ho : original ∈ caller) : window.Disjoint original := by
  simp only [privateWindows,List.mem_cons,List.not_mem_nil,or_false] at hw
  rcases hw with rfl | rfl | rfl | rfl | rfl | rfl
  · exact p.adapters.fresh original (by simp [ho])
  · exact p.lengths.fresh original (by simp [ho])
  · exact p.starts.fresh original (by simp [ho])
  · exact p.y.fresh original (by simp [ho])
  · exact p.w.fresh original (by simp [ho])
  · exact p.x.fresh original ho

theorem same_entry_values {α : Host.Role → Type} {entry : (role : Host.Role) → View α role} {caller : List Window}
    (p : Prepared entry caller) :
    p.x.values = (entry .input).values ∧ p.w.values = (entry .weight).values ∧ p.y.values = (entry .output).values ∧
    p.starts.values = (entry .starts).values ∧ p.lengths.values = (entry .lengths).values ∧ p.adapters.values = (entry .adapters).values :=
  ⟨p.x.preserved,p.w.preserved,p.y.preserved,p.starts.preserved,p.lengths.preserved,p.adapters.preserved⟩

#axiomsClean disjoint_symm
#axiomsClean private_pairwise
#axiomsClean separate_from_caller
#axiomsClean same_entry_values
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostSnapshots
