import bench.tritonbench_g.chunk_gated_attention.Sources.HostData

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RawStorage
open VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory

def Outside (base count p : Nat) : Prop := p<base ∨ base+4*count≤p
def Disjoint (a countA b countB : Nat) : Prop := a+4*countA≤b ∨ b+4*countB≤a

theorem stores_read (memory : Memory) (base count : Nat) (values : Nat→Word)
    (i : Nat) (hi : i<count) : read (stores memory base count values) (base+4*i)=values i := by
  rw [stores_render]
  exact render_read _ _ _ _ i hi

theorem stores_frame (memory : Memory) (base count : Nat) (values : Nat→Word)
    (p : Nat) (outside : Outside base count p) : stores memory base count values p=memory p := by
  rw [stores_render]
  exact render_frame _ _ _ _ p outside

theorem stores_read_frame (memory : Memory) (base count : Nat) (values : Nat→Word)
    (address : Nat) (outside : Disjoint base count address 1) :
    read (stores memory base count values) address=read memory address := by
  unfold VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory.read
  congr 1
  funext j
  apply stores_frame
  unfold Outside Disjoint at *
  have:=j.isLt
  omega

theorem disjoint_cell {a countA b countB i : Nat} (separate : Disjoint a countA b countB)
    (hi : i<countB) : Disjoint a countA (b+4*i) 1 := by
  unfold Disjoint at *
  omega

theorem disjoint_byte {a countA b countB j : Nat} (separate : Disjoint a countA b countB)
    (hj : j<4*countB) : Outside a countA (b+j) := by
  unfold Outside Disjoint at *
  omega

/-- An already contiguous shared view performs no byte mutation. -/
theorem stores_identity (memory : Memory) (base count : Nat) (values : Nat→Word)
    (same : ∀ i,i<count → values i=read memory (base+4*i)) : stores memory base count values=memory := by
  rw [stores_render]
  funext address
  unfold render
  split
  · rename_i inside
    have bound : (address-base)/4<count:=by omega
    rw [same _ bound]
    let j : Fin 4:=⟨(address-base)%4,Nat.mod_lt _ (by decide)⟩
    have byte:=encode_decode (fun k : Fin 4=>memory (base+4*((address-base)/4)+k.val)) j
    apply byte.trans
    congr 1
    dsimp [j]
    omega
  · rfl

#axiomsClean stores_read
#axiomsClean stores_read_frame
#axiomsClean stores_identity
end VeriTile.Bench.Repairs.ChunkGatedAttention.RawStorage
