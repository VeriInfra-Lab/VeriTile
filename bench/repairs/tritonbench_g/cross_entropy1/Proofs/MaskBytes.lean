import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TypedReads

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- The byte produced by one ordinary int64 equality operation. No kernel,
wrapper, or expected loss is mentioned. Bool tensors pack one byte per item. -/
structure EqualityBytes {α σ : Type} [Inhabited α] (encode : α → Word)
    (obs : Observations α σ) (rt : Runtime α σ) : Prop where
  compare : ∀ input target before output after, before.live input.region → input.dtype="torch.int64" →
    rt.equalInt input target before=.ok output after → ∀ i, i < input.numel →
      byteOf (rawWord encode after output.region (i/4)) ⟨i%4,by omega⟩=
        if obs.integer before input i=target then 1 else 0

theorem equality_bool_coupling {α σ : Type} [Inhabited α] (encode : α → Word)
    (obs : Observations α σ) (A : TensorArithmetic α) (rt : Runtime α σ)
    (ops : TensorLibrary obs A rt) (physical : EqualityBytes encode obs rt)
    (input : Tensor) (target : Int) (before : World α σ) (mask : Tensor) (compared : World α σ)
    (live : before.live input.region) (dtype : input.dtype="torch.int64")
    (call : rt.equalInt input target before=.ok mask compared) (i : Nat) (inside : i < input.numel) :
    BoolRead encode compared mask.region i := by
  have logical := ops.equalInt input target before mask compared live dtype call
  have raw := physical.compare input target before mask compared live dtype call i inside
  unfold BoolRead
  rw [logical.2.2 i inside,raw]
  by_cases equal : obs.integer before input i=target <;> simp [equal]

/-- This is the actual byte read by masked_fill after equality, including
the final partial word of a bool allocation. -/
theorem equality_native_byte {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (obs : Observations α σ) (rt : Runtime α σ) (physical : EqualityBytes encode obs rt)
    (input : Tensor) (target : Int) (before : World α σ) (mask : Tensor) (compared : World α σ)
    (live : before.live input.region) (dtype : input.dtype="torch.int64")
    (call : rt.equalInt input target before=.ok mask compared) (i : Nat) (inside : i < input.numel)
    (active : layout.active compared.native mask.region) (backed : i < (layout.buffer compared.native mask.region).capacity) :
    bytes layout encode compared (layout.buffer compared.native mask.region).device
      ((layout.buffer compared.native mask.region).base+i)=
        if obs.integer before input i=target then 1 else 0 := by
  have coord : 4*(i/4)+i%4=i := by omega
  have raw := raw_bytes layout encode compared mask.region (i/4) ⟨i%4,by omega⟩ active (by simpa only [coord] using backed)
  simpa only [Nat.add_assoc,coord] using raw.trans (physical.compare input target before mask compared live dtype call i inside)

#axiomsClean equality_bool_coupling
#axiomsClean equality_native_byte
#specNonCircular EqualityBytes avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
