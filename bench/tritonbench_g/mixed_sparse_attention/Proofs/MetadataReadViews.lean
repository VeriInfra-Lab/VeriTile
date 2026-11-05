import bench.tritonbench_g.mixed_sparse_attention.Proofs.Support.HostContract
import VeriTile.Triton.DSL.NonnegativeReadViews

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Checked
open VeriTile Triton FP.Structural Whole002 Source
noncomputable section
variable {α : Type} [Inhabited α]

/-- Physical signed32 decoding agrees with the existing mathematical metadata
read view at this actual cell. The mathematical source theorem itself retains
its original natural domain; this is the checked native boundary. -/
def SignedRead (c : Config) (values : Observed) (s : State α) (role : Role) (address : Nat) : Prop :=
  (BitVec.ofInt 32 (values role address)).toInt=values role address ∧
  (s.mem (role.region c) address).read .nat=(BitVec.ofInt 32 (values role address)).toNat

theorem signedRead (c : Config) (values : Observed) (s : State α)
    (binding : MetadataBinding c values s) (role : Role)
    (hr : role∈[Role.sequence,.blocks,.offsets,.counts,.columns]) (address : Nat)
    (lo : 0≤values role address) (hi : values role address<2147483648) :
    SignedRead c values s role address := by
  have decoded := DSL.NonnegativeReadViews.signed32_read (values role address) lo hi
  exact ⟨decoded.1,(binding.read role hr address).trans decoded.2.symm⟩

/-- Every metadata read selected by the existing checked wrapper has the same
signed word and natural value. This derives all bounds from Admission and its
original metadata guards, with no new input or successful-execution premise. -/
structure MetadataReads (c : Config) (values : Observed) (s : State α) : Prop where
  sequence : ∀ b<c.Z,SignedRead c values s .sequence b
  blocks : ∀ hz<c.Z*c.H,∀ row<c.NR,SignedRead c values s .blocks (hz*c.NR+row)
  counts : ∀ hz<c.Z*c.H,∀ row<c.NR,SignedRead c values s .counts (hz*c.NR+row)
  offsets : ∀ hz<c.Z*c.H,∀ row<c.NR,∀ j,
    j<(values .blocks (hz*c.NR+row)).toNat → SignedRead c values s .offsets ((hz*c.NR+row)*c.NS+j)
  columns : ∀ hz<c.Z*c.H,∀ row<c.NR,∀ j,
    j<(values .counts (hz*c.NR+row)).toNat → SignedRead c values s .columns ((hz*c.NR+row)*c.NV+j)

theorem admitted_metadata_reads (c : Config) (request : Request) (values : Observed)
    (s : State α) (accepted : Admission c request values) (binding : MetadataBinding c values s) :
    MetadataReads c values s := by
  have bounds := accepted.schema.dimensions
  have hT : c.T<2147483648 := bounds c.T (by simp)
  have hS : c.NS<2147483648 := bounds c.NS (by simp)
  have hV : c.NV<2147483648 := bounds c.NV (by simp)
  constructor
  · intro b hb
    have h := accepted.metadata.sequence b hb
    apply signedRead c values s binding .sequence (by simp) b h.1
    omega
  · intro hz hh row hr
    have h := accepted.metadata.blocks hz hh row hr
    apply signedRead c values s binding .blocks (by simp) _ h.1
    omega
  · intro hz hh row hr
    have h := accepted.metadata.counts hz hh row hr
    apply signedRead c values s binding .counts (by simp) _ h.1
    omega
  · intro hz hh row hr j hj
    have h := accepted.metadata.offsets hz hh row hr j hj
    have hH := accepted.schema.heads
    have hb : hz/c.H<c.Z := (Nat.div_lt_iff_lt_mul hH).mpr (by simpa [Nat.mul_comm] using hh)
    have hs := accepted.metadata.sequence (hz/c.H) hb
    apply signedRead c values s binding .offsets (by simp) _ h.1
    omega
  · intro hz hh row hr j hj
    have h := accepted.metadata.columns hz hh row hr j hj
    have hH := accepted.schema.heads
    have hb : hz/c.H<c.Z := (Nat.div_lt_iff_lt_mul hH).mpr (by simpa [Nat.mul_comm] using hh)
    have hs := accepted.metadata.sequence (hz/c.H) hb
    apply signedRead c values s binding .columns (by simp) _ h.1
    omega

#axiomsClean signedRead
#axiomsClean admitted_metadata_reads
#specNonCircular MetadataReads avoiding [FP.Structural.exec,Whole.invoke,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Checked
