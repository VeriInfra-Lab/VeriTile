import bench.repairs.tritonbench_g.parallel_retention_attention.Contract

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton
noncomputable section

theorem forward_address_bound (s : Shape) (tile : Tile) (valid : ProgramDomain s 128 tile)
    (i : TileIndex [128,s.BV]) (hi : activeTile s.T s.DV (tile.chunk*128) (tile.value*s.BV) i) :
    forwardAddress s tile i<OutputKind.forward.count s := by
  have hb := dense_bound s.T s.DV (s.B*s.H) tile.bh (tile.chunk*128+i.1.val)
    (tile.value*s.BV+i.2.1.val) valid.bh hi.1 hi.2
  have hk := Nat.mul_le_mul_right (s.B*s.H*s.T*s.DV) (Nat.succ_le_of_lt valid.key)
  have eq : forwardAddress s tile i=tile.key*(s.B*s.H*s.T*s.DV)+
      dense s.T s.DV tile.bh (tile.chunk*128+i.1.val) (tile.value*s.BV+i.2.1.val) := by
    unfold forwardAddress tileOffset dense; ring
  rw [eq]
  change _<blocks s.DK s.BK*(s.B*s.H*s.T*s.DV)
  nlinarith

theorem key_address_bound (s : Shape) (tile : Tile) (valid : ProgramDomain s 64 tile)
    (i : TileIndex [64,s.BK]) (hi : activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i) :
    keyAddress s tile i<OutputKind.key.count s := by
  have hb := dense_bound s.T s.DK (s.B*s.H) tile.bh (tile.chunk*64+i.1.val)
    (tile.key*s.BK+i.2.1.val) valid.bh hi.1 hi.2
  have hk := Nat.mul_le_mul_right (s.B*s.H*s.T*s.DK) (Nat.succ_le_of_lt valid.value)
  have eq : keyAddress s tile i=tile.value*(s.B*s.H*s.T*s.DK)+
      dense s.T s.DK tile.bh (tile.chunk*64+i.1.val) (tile.key*s.BK+i.2.1.val) := by
    unfold keyAddress tileOffset dense; ring
  rw [eq]
  change _<blocks s.DV s.BV*(s.B*s.H*s.T*s.DK)
  nlinarith

theorem value_address_bound (s : Shape) (tile : Tile) (valid : ProgramDomain s 64 tile)
    (i : TileIndex [64,s.BV]) (hi : activeTile s.T s.DV (tile.chunk*64) (tile.value*s.BV) i) :
    valueAddress s tile i<OutputKind.value.count s := by
  have hb := dense_bound s.T s.DV (s.B*s.H) tile.bh (tile.chunk*64+i.1.val)
    (tile.value*s.BV+i.2.1.val) valid.bh hi.1 hi.2
  have hk := Nat.mul_le_mul_right (s.B*s.H*s.T*s.DV) (Nat.succ_le_of_lt valid.key)
  have eq : valueAddress s tile i=tile.key*(s.B*s.H*s.T*s.DV)+
      dense s.T s.DV tile.bh (tile.chunk*64+i.1.val) (tile.value*s.BV+i.2.1.val) := by
    unfold valueAddress tileOffset dense; ring
  rw [eq]
  change _<blocks s.DK s.BK*(s.B*s.H*s.T*s.DV)
  nlinarith

theorem scratch_byte_bound (s : Shape) (kind : OutputKind) (base device : Nat)
    (tensor : ScratchTensor s kind base device) (offset : Nat) (bound : offset<kind.count s) (j : Fin 4) :
    tensor.metadata.dataPtr.toNat+4*offset+j.val<2^64 := by
  rw [tensor.pointer]
  have room := tensor.room
  have hj := j.isLt
  omega

#axiomsClean forward_address_bound
#axiomsClean key_address_bound
#axiomsClean value_address_bound
#axiomsClean scratch_byte_bound
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
