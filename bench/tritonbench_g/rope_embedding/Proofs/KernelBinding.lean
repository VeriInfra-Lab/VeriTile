import bench.tritonbench_g.rope_embedding.Proofs.KernelFP
namespace VeriTile.Bench.Repairs.RopeEmbedding.KernelBinding
open VeriTile.Triton FP.Structural
open VeriTile.Torch
open Numerics (Word)
noncomputable section
set_option maxHeartbeats 3000000

/-- The three actual tensors passed to the kernel. Shared Torch laws bind
metadata to physical storage; no output or kernel execution is a premise. -/
structure Inputs where
  (q cosine sine : Torch.FP32.Tensor)
  qLaws : Torch.FP32.TensorLaws q
  cosineLaws : Torch.FP32.TensorLaws cosine
  sineLaws : Torch.FP32.TensorLaws sine
  (cosineRows sineRows : Nat)

def metadata (i : Inputs) (r : Request) : KernelMetadata :=
  ⟨r,i.q.metadata,i.cosine.metadata,i.sine.metadata,i.cosineRows,i.sineRows⟩

def memory (read : Nat→Word) (i : Inputs) (other : RegionName→Nat→Word)
    (region : RegionName) (o : Nat) : Word :=
  if region="Q" then read (i.q.dataPtr.toNat+4*o)
  else if region="C" then read (i.cosine.dataPtr.toNat+4*o)
  else if region="S" then read (i.sine.dataPtr.toNat+4*o)
  else other region o

theorem payload_facts (t : TensorMetadata) (h : payloadCheck t=Bool.true) :
    t.isContiguous=Bool.true ∧ t.isNeg=Bool.false := by
  simp only [payloadCheck,Bool.and_eq_true] at h
  refine ⟨h.1.2,?_⟩
  cases hn : t.isNeg  <;> simp_all

theorem admitted_payload (i : Inputs) (r : Request) (h : (metadata i r).check=Bool.true) :
    payloadCheck i.q.metadata=Bool.true ∧ payloadCheck i.cosine.metadata=Bool.true ∧
      payloadCheck i.sine.metadata=Bool.true := by
  simp only [KernelMetadata.check,metadata,Bool.and_eq_true,decide_eq_true_eq,
    List.all_cons,List.all_nil,Bool.and_true] at h
  tauto

theorem admitted_shapes (i : Inputs) (r : Request) (h : (metadata i r).check=Bool.true) :
    i.q.metadata.shape=[r.rows,r.heads*r.dimension] ∧
      tableShape r.half i.cosineRows i.cosine.metadata.shape ∧
      tableShape r.half i.sineRows i.sine.metadata.shape ∧
      r.sequence ≤ i.cosineRows ∧ r.sequence ≤ i.sineRows := by
  simp only [KernelMetadata.check,metadata,Bool.and_eq_true,decide_eq_true_eq] at h
  tauto

theorem table_numel (half rows : Nat) (t : TensorMetadata) (h : tableShape half rows t.shape) :
    t.numel=rows*half := by
  rcases h with h | ⟨hh,h⟩
  · simp [TensorMetadata.numel,h]
  · simp [TensorMetadata.numel,h,hh]

/-- Actual data_ptr plus an element offset denotes the logical input value.
This derives the relationship from shared contiguity/storage laws. -/
theorem actual_read (read : Nat→Word) (t : Torch.FP32.Tensor)
    (laws : Torch.FP32.TensorLaws t) (payload : payloadCheck t.metadata=Bool.true)
    (k : Nat) (hk : k < t.metadata.numel) :
    read (t.dataPtr.toNat+4*k)=Torch.FP32.logicalValue read t laws k := by
  obtain ⟨hc,hn⟩ := payload_facts t.metadata payload
  have direct := Torch.FP32.direct_read_eq_logical read t laws hc hn k hk
  have ptr := congrArg BitVec.toNat (Torch.FP32.actual_pointer t laws hc (by omega))
  have span := (Torch.FP32.dataView t laws hc).spanFits
  have bound : (Torch.FP32.dataView t laws hc).base < 2^64 := by
    change (Torch.FP32.dataView t laws hc).base+4*t.metadata.numel ≤ 2^64 at span
    omega
  simp only [Torch.FP32.DataView.address,Nat.mul_zero,Nat.add_zero,
    BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] at ptr
  rw [←ptr]
  exact direct

theorem input_binding (read : Nat→Word) (i : Inputs) (r : Request)
    (other : RegionName→Nat→Word) (h : (metadata i r).check=Bool.true)
    (row head col : Nat) (hr : row < r.rows) (hh : head < r.heads) (hc : col < r.dimension) :
    memory read i other "Q" (r.offset row head col)=
      Torch.FP32.logicalValue read i.q i.qLaws (r.offset row head col) := by
  have shape := (admitted_shapes i r h).1
  have size : i.q.metadata.numel=r.size := by
    simp [TensorMetadata.numel,shape,Request.size,Nat.mul_assoc]
  simp only [memory,↓reduceIte]
  exact actual_read read i.q i.qLaws (admitted_payload i r h).1 _
    (by rw [size]; exact offset_lt r row head col hr hh hc)

theorem coefficient_bound (r : Request) (row lane : Nat) (hr : row < r.rows) (hl : lane < r.half) :
    row%r.sequence*r.half+lane < r.sequence*r.half := by
  have hs : 0 < r.sequence := by
    by_contra h
    have hz : r.sequence=0 := by omega
    simp [Request.rows,hz] at hr
  have hm := Nat.mod_lt row hs
  have hp := Nat.mul_le_mul_right r.half (show row%r.sequence+1 ≤ r.sequence by omega)
  rw [Nat.add_mul,Nat.one_mul] at hp
  omega

theorem coefficient_binding (read : Nat→Word) (i : Inputs) (r : Request)
    (other : RegionName→Nat→Word) (h : (metadata i r).check=Bool.true)
    (row lane : Nat) (hr : row < r.rows) (hl : lane < r.half) :
    memory read i other "C" (row%r.sequence*r.half+lane)=
      Torch.FP32.logicalValue read i.cosine i.cosineLaws (row%r.sequence*r.half+lane) ∧
    memory read i other "S" (row%r.sequence*r.half+lane)=
      Torch.FP32.logicalValue read i.sine i.sineLaws (row%r.sequence*r.half+lane) := by
  obtain ⟨_,cs,ss,cl,sl⟩ := admitted_shapes i r h
  have cp := Nat.mul_le_mul_right r.half cl
  have sp := Nat.mul_le_mul_right r.half sl
  have bound := coefficient_bound r row lane hr hl
  have cb : row%r.sequence*r.half+lane < i.cosine.metadata.numel := by
    rw [table_numel _ _ _ cs]; omega
  have sb : row%r.sequence*r.half+lane < i.sine.metadata.numel := by
    rw [table_numel _ _ _ ss]; omega
  constructor
  · simp only [memory,↓reduceIte]
    exact actual_read read i.cosine i.cosineLaws (admitted_payload i r h).2.1 _ cb
  · simp only [memory,↓reduceIte]
    exact actual_read read i.sine i.sineLaws (admitted_payload i r h).2.2 _ sb

#axiomsClean actual_read
#axiomsClean input_binding
#axiomsClean coefficient_binding
end
end VeriTile.Bench.Repairs.RopeEmbedding.KernelBinding
