import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostControl

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostSnapshotWitness
open HostSnapshots

def freshBacking : List Window → Nat
  | [] => 0
  | w::ws => w.backing+1+freshBacking ws

theorem fresh_gt (occupied : List Window) (previous : Window) (member : previous ∈ occupied) :
    previous.backing < freshBacking occupied := by
  induction occupied with
  | nil => simp at member
  | cons w ws ih =>
      rcases List.mem_cons.mp member with rfl | member
      · simp only [freshBacking]; omega
      · have := ih member
        simp only [freshBacking]; omega

/-- Explicit successful local allocator/clone witness for finite tensor views.
It demonstrates consistency of the local contract; no hardware-allocation or
tensor-library implementation conformance is inferred from this witness. -/
def clone {α : Host.Role → Type} (role : Host.Role) (entry : View α role) (occupied : List Window) :
    CloneResult role entry occupied where
  values := entry.values
  window := ⟨freshBacking occupied,0,payloadLength role entry.values*entry.bytesPerElement⟩
  preserved := rfl
  extent := rfl
  fresh := by
    intro previous member backing address current old
    have h := fresh_gt occupied previous member
    have hc := current.1
    have ho := old.1
    simp only at hc
    omega
  stride := fun dim => (entry.dimensions.drop (dim+1)).prod
  matrixLayout := by
    intro rows columns dims
    rw [dims]
    constructor <;> intros <;> simp
  weightLayout := by
    intro adapters columns inner dims
    rw [dims]
    constructor <;> intros <;> simp
  vectorLayout := by
    intro count dims positive
    rw [dims]
    simp

def api (α : Host.Role → Type) : HostControl.API α where
  enter := .ok ()
  snapshots := ⟨fun role entry occupied => .ok (clone role entry occupied)⟩
  lists := ⟨fun _ values => .ok ⟨values,rfl⟩⟩
  launch := fun _ => .ok ()
  leave := .ok ()

theorem snapshots_succeed {α : Host.Role → Type} (entry : (role : Host.Role) → View α role)
    (caller : List Window) : ∃ p, HostSnapshots.prepare (api α).snapshots entry caller = .ok p := by
  exact ⟨_,rfl⟩

#axiomsClean fresh_gt
#axiomsClean clone
#axiomsClean snapshots_succeed
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostSnapshotWitness
