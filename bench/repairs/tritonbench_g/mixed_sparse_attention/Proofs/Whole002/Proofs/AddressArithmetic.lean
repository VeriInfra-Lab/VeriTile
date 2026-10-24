import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Domain
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Whole
open VeriTile Triton FP.Structural Source
noncomputable section

theorem dense_bound (heads length dim hz row channel : Nat)
    (hh : hz<heads) (hr : row<length) (hd : channel<dim) :
    (hz*length+row)*dim+channel<heads*length*dim := by
  have hp:=Nat.mul_le_mul_right length (show hz+1≤heads by omega)
  have hp' : hz*length+row+1≤heads*length := by
    rw [Nat.add_mul,Nat.one_mul] at hp
    omega
  have hq:=Nat.mul_le_mul_right dim hp'
  rw [Nat.add_mul,Nat.one_mul] at hq
  omega

theorem metadata_bound (heads rows hz block : Nat) (hh : hz<heads) (hb : block<rows) :
    hz*rows+block<heads*rows := by
  have hp:=Nat.mul_le_mul_right rows (show hz+1≤heads by omega)
  rw [Nat.add_mul,Nat.one_mul] at hp
  omega

theorem selected_metadata_bound (heads rows capacity hz block lane count : Nat)
    (hh : hz<heads) (hb : block<rows) (hc : count≤capacity) (hl : lane<count) :
    (hz*rows+block)*capacity+lane<heads*rows*capacity := by
  have hm:=metadata_bound heads rows hz block hh hb
  have hp:=Nat.mul_le_mul_right capacity (show hz*rows+block+1≤heads*rows by omega)
  rw [Nat.add_mul,Nat.one_mul] at hp
  omega

/-- Descriptor/count facts for the original contiguous metadata arrays. -/
structure Metadata {α : Type} [Inhabited α] (c : Config) (s : State α) : Prop where
  sequence : ∀ b<c.Z,(s.mem (Region.cast c.seq) b).read .nat≤c.T
  blockCount : ∀ hz<c.Z*c.H,∀ row<c.NR,
    (s.mem (Region.cast c.blocks) (hz*c.NR+row)).read .nat≤c.NS
  columnCount : ∀ hz<c.Z*c.H,∀ row<c.NR,
    (s.mem (Region.cast c.counts) (hz*c.NR+row)).read .nat≤c.NV
  blockOffset : ∀ hz<c.Z*c.H,∀ row<c.NR,∀ b,
    b<(s.mem (Region.cast c.blocks) (hz*c.NR+row)).read .nat →
    (s.mem (Region.cast c.offsets) ((hz*c.NR+row)*c.NS+b)).read .nat<
      (s.mem (Region.cast c.seq) (hz/c.H)).read .nat
  columnIndex : ∀ hz<c.Z*c.H,∀ row<c.NR,∀ j,
    j<(s.mem (Region.cast c.counts) (hz*c.NR+row)).read .nat →
    (s.mem (Region.cast c.columns) ((hz*c.NR+row)*c.NV+j)).read .nat<
      (s.mem (Region.cast c.seq) (hz/c.H)).read .nat

#axiomsClean dense_bound
#axiomsClean metadata_bound
#axiomsClean selected_metadata_bound
#specNonCircular Metadata avoiding [FP.Structural.exec,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Whole
