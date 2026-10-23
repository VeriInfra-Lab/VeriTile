import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Host
import bench.tritonbench_g.kldiv_ops.Proofs.Math

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open KldivTriton.Host (World ForwardCall BackwardCall Result)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 6000000

/-- Ordinary successful-dispatch PID representation. It states no source
success, output value, arithmetic, or frame property. -/
structure LaunchContract {M : Type} (rt : Runtime M) : Prop where
  forward : ∀ call out before after, rt.setupForward call out before = .ok () after →
    call.prediction.rows ≤ 2^31
  backward : ∀ call before after, rt.setupBackward call before = .ok () after →
    call.input.rows ≤ 2^31

theorem input_after_prefix {M α β ι : Type} (ops : Ops M α β) (neg : Negation α)
    (input output : View) (ib : input.backed) (ob : output.backed)
    (separate : input.storage.separate output.storage) (index : ι → Nat)
    (values : ι → α) (done : List ι) (valid : ∀ j ∈ done, index j < output.numel)
    (initial : M) (i : Nat) (hi : i < input.numel) :
    logical ops neg (stores ops (fun j => output.key (index j)) values done initial) input i =
      logical ops neg initial input i := by
  unfold logical
  congr 1
  apply read_frame
  intro j hj
  exact input.storage.spans_separate output.storage separate _ _ (ib i hi) (ob _ (valid j hj))

/-- Every possible prefix of direct backward stores leaves every subsequent
target load unchanged. The checked guard supplies separation, including the
vacuous empty-destination case; no snapshot Tensor is introduced. -/
theorem backward_prefix {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (d t g : Tensor) (admit : BackwardContract d t g) (destination target : View)
    (db : Binding d destination) (tb : Binding t target) (values : Nat → α)
    (done : List Nat) (valid : ∀ j ∈ done, j < destination.numel)
    (initial : M) (i : Nat) (hi : i < target.numel) :
    logical ops neg (stores ops destination.key values done initial) target i =
      logical ops neg initial target i := by
  have size : target.numel = destination.numel := by simp [View.numel,tb.shape,db.shape,admit.shape]
  have nonempty : d.metadata.numel ≠ 0 := by rw [←db.toViewBinding.numel];omega
  exact input_after_prefix ops neg target destination tb.backed db.backed
    (KldivTriton.Host.separate_symm (db.toViewBinding.separate tb.toViewBinding (admit.separate nonempty).1))
    id values done valid initial i hi

/-- Active flat source offsets are exactly the supplied Tensor's physical
keys, with alignment and all four native bytes proved from its real backing. -/
theorem source_pointer {t : Tensor} {v : View} (binding : Binding t v)
    (admit : TensorContract t) (i : Nat) (hi : i < v.numel) :
    (t.metadata.dataPtr + BitVec.ofNat 64 i * BitVec.ofNat 64 4).toNat = (v.key i).address ∧
    (v.key i).address % 4 = 0 ∧
    ∀ byte, byte < 4 → (v.key i).address+byte < v.storage.baseByte+v.storage.bytes ∧
      (v.key i).address+byte < 2^64 := by
  have nonempty : t.metadata.numel ≠ 0 := by rw [←binding.toViewBinding.numel];omega
  have ptr := binding.pointer nonempty
  have key := binding.contiguous admit.contiguous i hi
  have backed := binding.backed i hi
  have capacity := v.storage.representable
  have align := admit.alignment nonempty
  rw [ptr] at align
  have addr : (v.key i).address = v.storage.baseByte+4*v.storageOffset+4*i := by
    rw [key];simp [Allocation.key];ring
  have endBound : (v.key i).address+4 ≤ v.storage.baseByte+v.storage.bytes := backed.2.2
  refine ⟨?_,?_,?_⟩
  · rw [BitVec.toNat_add,BitVec.toNat_mul,ptr]
    simp only [BitVec.toNat_ofNat]
    have ibound : i < 2^64 := by rw [addr] at endBound;omega
    have product : i*4 < 2^64 := by rw [addr] at endBound;omega
    rw [Nat.mod_eq_of_lt ibound,show 4 % 2^64 = 4 from rfl,Nat.mod_eq_of_lt product]
    rw [Nat.mod_eq_of_lt (by rw [addr] at endBound;omega)]
    omega
  · rw [addr];omega
  · intro byte hb;constructor <;> omega

theorem dense_size_bound (v : View) (dense : Dense v) (backed : v.backed) : v.numel ≤ 2^62 := by
  by_cases empty : v.numel = 0
  · omega
  have last := backed (v.numel-1) (by omega)
  rw [dense _ (by omega)] at last
  have capacity := v.storage.representable
  simp only [Allocation.contains,Allocation.key] at last
  omega

/-- Fresh allocator outputs use their actual dense view pointer. This covers
both the original rank-two element output and original rank-one row output. -/
theorem allocated_pointer (v : View) (dense : Dense v) (backed : v.backed)
    (aligned : v.storage.baseByte % 4 = 0) (i : Nat) (hi : i < v.numel) :
    (BitVec.ofNat 64 (v.storage.baseByte+4*v.storageOffset) +
      BitVec.ofNat 64 i * BitVec.ofNat 64 4).toNat = (v.key i).address ∧
    (v.key i).address % 4 = 0 ∧
    ∀ byte, byte < 4 → (v.key i).address+byte < v.storage.baseByte+v.storage.bytes ∧
      (v.key i).address+byte < 2^64 := by
  have key := dense i hi
  have inside := (backed i hi).2.2
  have capacity := v.storage.representable
  have addr : (v.key i).address = v.storage.baseByte+4*v.storageOffset+i*4 := by
    rw [key];simp [Allocation.key];ring
  refine ⟨?_,?_,?_⟩
  · rw [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat,←addr,Nat.mod_eq_of_lt (by omega)]
  · rw [addr];omega
  · intro byte hb;constructor <;> omega

/-- Literal source stride V is exact even when Torch retains arbitrary
unused strides on empty or singleton dimensions. -/
theorem matrix_pointer {t : Tensor} {v : View} (binding : Binding t v)
    (admit : TensorContract t) (rows cols : Nat) (shape : v.shape = [rows,cols])
    (r c : Nat) (hr : r < rows) (hc : c < cols) :
    (t.metadata.dataPtr + BitVec.ofNat 64 r * BitVec.ofNat 64 cols * BitVec.ofNat 64 4 +
      BitVec.ofNat 64 c * BitVec.ofNat 64 4).toNat = (v.key (r*cols+c)).address := by
  have active : r*cols+c < v.numel := by simp only [View.numel,shape,List.prod_cons,List.prod_nil,Nat.mul_one];nlinarith
  have h := (source_pointer binding admit (r*cols+c) active).1
  simpa only [BitVec.ofNat_add,BitVec.ofNat_mul,BitVec.add_mul,BitVec.add_assoc] using h

theorem checked_indices {t : Tensor} {v : View} (binding : Binding t v)
    (admit : TensorContract t) (rows cols : Nat) (shape : v.shape = [rows,cols])
    (hasRow : 0 < rows) (k lane : Nat)
    (hk : k < VeriTile.Bench.TritonBenchG.KldivOps.FullMath.rounds cols (Host.blockSize cols))
    (hl : lane < Host.blockSize cols) :
    (BitVec.ofNat 64 (k*Host.blockSize cols)+BitVec.ofNat 64 lane).slt (BitVec.ofNat 64 cols) =
      decide (k*Host.blockSize cols+lane < cols) ∧
    (BitVec.ofNat 64 (k*Host.blockSize cols)+BitVec.ofNat 64 (Host.blockSize cols)).toInt =
      (((k+1)*Host.blockSize cols : Nat) : Int) := by
  have count := dense_size_bound v (binding.contiguous admit.contiguous) binding.backed
  have columns : cols ≤ 2^62 := by simp only [View.numel,shape,List.prod_cons,List.prod_nil,Nat.mul_one] at count;nlinarith
  exact ⟨KldivTriton.Index.loop_mask_exact cols (Host.blockSize cols) k lane
    (Host.block_positive _) (Host.block_capped _) columns hk hl,
    KldivTriton.Index.loop_increment_exact cols (Host.blockSize cols) k
    (Host.block_positive _) (Host.block_capped _) columns hk⟩

theorem pid_exact (rows r : Nat) (grid : rows ≤ 2^31) (hr : r < rows) :
    (BitVec.ofNat 32 r).toInt = (r : Int) :=
  SwigluTriton.Index.signed_ofNat 32 r (by decide) (by omega)

#axiomsClean backward_prefix
#axiomsClean source_pointer
#axiomsClean matrix_pointer
#axiomsClean allocated_pointer
#axiomsClean checked_indices
#axiomsClean pid_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous
