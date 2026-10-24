import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.WholeFlat

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
open VeriTile.Triton
noncomputable section

def byteOf (w : BitVec 32) (i : Nat) : BitVec 8 := BitVec.ofNat 8 (w.toNat/256^(i%4))

/-- Encoding is solely a cell representation parameter. It cannot change source
execution or any specified result. Native bit cells use the identity encoding. -/
def materializeBytes (A : FlatAlloc) (encode : MemCell → BitVec 32)
    (raw : Nat → BitVec 8) (s : BlockState) (i : Nat) : BitVec 8 :=
  if (A.decode (i/4)).isSome then byteOf (encode (s.mem A.flat (i/4))) (i%4) else raw i

def byteWrites (c : Config) (A : FlatAlloc) (i : Nat) : Prop :=
  flatWrites c A (A.flat,i/4)

/-- The initial flat byte image agrees with allocated FP32 cells. Bytes outside
the allocation table remain arbitrary and are retained verbatim. -/
def ByteBinding (A : FlatAlloc) (encode : MemCell → BitVec 32)
    (raw : Nat → BitVec 8) (s : BlockState) : Prop :=
  ∀ i,(A.decode (i/4)).isSome=Bool.true → raw i=byteOf (encode (s.mem A.flat (i/4))) (i%4)

theorem byte_frame (c : Config) (A : FlatAlloc) (encode : MemCell → BitVec 32)
    (raw : Nat → BitVec 8) (s t : BlockState) (hb : ByteBinding A encode raw s)
    (hf : BlockState.WriteWithin (flatWrites c A) s t) (i : Nat) (hi : ¬byteWrites c A i) :
    materializeBytes A encode raw t i=raw i := by
  unfold materializeBytes
  split_ifs with hd
  · rw [← hf A.flat (i/4) hi]
    exact (hb i hd).symm
  · rfl

theorem byte_readback (A : FlatAlloc) (ha : A.Disjoint) (encode : MemCell → BitVec 32)
    (raw : Nat → BitVec 8) (s : BlockState) (region : RegionName) (off k : Nat)
    (hr : region∈A.regions) (ho : off<A.extent region) (hk : k<4) :
    materializeBytes A encode raw s (4*A.addr region off+k)=
      byteOf (encode (s.mem A.flat (A.addr region off))) k := by
  have hd : (4*A.addr region off+k)/4=A.addr region off := by omega
  have hm : (4*A.addr region off+k)%4=k := by omega
  simp only [materializeBytes,hd,hm,A.decode_addr ha hr ho,Option.isSome_some,ite_true]

/-- Complete sequential flat execution induces an exact byte frame, including
arbitrary allocation gaps and byte contents outside all active output cells. -/
theorem whole_byte_frame (c : Config) (A : FlatAlloc) (ha : DenseAllocation c A)
    (s : BlockState) (order : List (CTA c)) (ho : order.Perm (allCTAs c))
    (encode : MemCell → BitVec 32) (raw : Nat → BitVec 8)
    (hb : ByteBinding A encode raw (A.flattenState s)) :
    ∃ final,runFlat c A order (A.flattenState s)=some final ∧
      ∀ i,¬byteWrites c A i → materializeBytes A encode raw final i=raw i := by
  obtain ⟨final,he,_,_,_,hf⟩ := whole_flat_result c A ha s order ho
  exact ⟨final,he,fun i hi => byte_frame c A encode raw (A.flattenState s) final hb hf i hi⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Whole
