import bench.tritonbench_g.chunk_gla_simple.Proofs.Shape
import VeriTile.Triton.Launch.Grid

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
open VeriTile.Triton

def grid (c : Shape) : Grid := ⟨c.grid⟩

theorem block_pos (n : Nat) : 0 < min (nextPowerOfTwo n) 64 := by
  simp only [lt_min_iff,nextPowerOfTwo]
  exact ⟨Nat.pow_pos (by decide), by decide⟩

theorem div_lt_cdiv {x n b : Nat} (hx : x < n) (hb : 0 < b) :
    x / b < (n+b-1)/b := by
  have hm := Nat.mod_lt (n+b-1) hb
  have hd := Nat.mod_add_div (n+b-1) b
  have hs : n+b-1+1 = n+b := by omega
  apply (Nat.div_lt_iff_lt_mul hb).2
  nlinarith

def index (c : Shape) (v : Fin c.NV) (t : Fin c.NT) (h : Fin c.heads) : GridIndex (grid c) :=
  fun a => match a with
    | ⟨0,_⟩ => v
    | ⟨1,_⟩ => t
    | ⟨2,_⟩ => h
    | ⟨n+3,hn⟩ => False.elim (by have : n+3 < 3 := hn; omega)

def owner (c : Shape) (ht : 0 < c.BT) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    GridIndex (grid c) :=
  index c ⟨v.val/c.BV,div_lt_cdiv v.isLt (block_pos c.V)⟩
    ⟨t.val/c.BT,div_lt_cdiv t.isLt ht⟩ h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
