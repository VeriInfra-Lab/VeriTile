import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Results

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Public
open Checked Binding
noncomputable section

def program (c : Shape) (h k v : Nat) : Nat := h*c.tiles+(k/64*c.VT+v/16)
def lane (k v : Nat) : Nat := k%64*16+v%16

theorem complete_coordinates (c : Shape) (h k v : Nat) (hh : h<c.heads) (hk : k<c.K) (hv : v<c.V) :
    program c h k v<c.programs ∧ lane k v<1024 ∧
    (program c h k v)/c.tiles=h ∧
    ((program c h k v)%c.tiles/c.VT)*64+lane k v/16=k ∧
    ((program c h k v)%c.tiles%c.VT)*16+lane k v%16=v := by
  have coord := coordinate_coverage c k v hk hv
  have grid := Safety.program_grid c h (k/64) (v/16) hh coord.1 coord.2.1
  have lanes := Safety.lane_grid (k%64) (v%16) coord.2.2.1 coord.2.2.2.1
  exact ⟨grid.1,lanes.1,grid.2.1,by dsimp only [program,lane]; rw [grid.2.2.1,lanes.2.1]; omega,
    by dsimp only [program,lane]; rw [grid.2.2.2,lanes.2.2]; omega⟩

theorem matrix_key (c : Shape) (seed : Bool) (runtime : Runtime c seed) (r : Region)
    (role : r=.x ∨ r=.output ∨ r=.incoming ∨ r=.dx) (h t k v : Nat)
    (hh : h<c.heads) (hk : k<c.K) (hv : v<c.V) :
    key c seed runtime ⟨r,program c h k v,t,lane k v⟩=
      ⟨(runtime.tensor r).device,(runtime.tensor r).base+4*matrixOffset c h t k v⟩ := by
  have coord := complete_coordinates c h k v hh hk hv
  rcases role with rfl|rfl|rfl|rfl <;>
    simp only [key,Machine.offset,coord.2.2.1,coord.2.2.2.1,coord.2.2.2.2,Int.toNat_natCast]

theorem seed_key (c : Shape) (seed : Bool) (runtime : Runtime c seed) (h k v : Nat)
    (hh : h<c.heads) (hk : k<c.K) (hv : v<c.V) :
    key c seed runtime ⟨.dseed,program c h k v,0,lane k v⟩=
      ⟨(runtime.tensor .dseed).device,(runtime.tensor .dseed).base+4*seedDenseOffset c h k v⟩ := by
  have coord := complete_coordinates c h k v hh hk hv
  simp only [key,Machine.offset,coord.2.2.1,coord.2.2.2.1,coord.2.2.2.2,Int.toNat_natCast]

theorem batch_head (c : Shape) (b h : Nat) (hb : b<c.B) (hh : h<c.H) : b*c.H+h<c.heads :=
  Safety.mixed_bound _ _ _ _ hb hh

#axiomsClean complete_coordinates
#axiomsClean matrix_key
#axiomsClean seed_key
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Public
