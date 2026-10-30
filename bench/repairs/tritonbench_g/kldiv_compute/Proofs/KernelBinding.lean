import bench.repairs.tritonbench_g.kldiv_compute.Contract
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Checked.Pointers

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Contiguous
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 4000000

theorem direct_key (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous = Bool.true) (i : Nat) (hi : i < t.metadata.numel) :
    directKey t i = key t i := by
  simp only [directKey,key,laws.contiguous contiguous i hi]

theorem key_backed (t : Tensor) (laws : TensorLaws t) (i : Nat) (hi : i < t.metadata.numel) :
    (allocation t).contains (key t i) := by
  have inside := laws.inBounds i hi
  simp only [allocation,key,Allocation.contains,Allocation.key]
  constructor
  · trivial
  constructor <;> omega

theorem direct_backed (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous = Bool.true) (i : Nat) (hi : i < t.metadata.numel) :
    (allocation t).contains (directKey t i) := by
  rw [direct_key t laws contiguous i hi]
  exact key_backed t laws i hi

theorem direct_logical {M α β : Type} (ops : Ops M α β) (negate : α → α)
    (memory : M) (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous = Bool.true) (notNeg : t.metadata.isNeg = Bool.false)
    (i : Nat) (hi : i < t.metadata.numel) :
    ops.read memory (directKey t i) = logical ops negate memory t i := by
  rw [direct_key t laws contiguous i hi]
  simp [logical,notNeg]

theorem direct_disjoint (t : Tensor) (i j : Nat) (different : i ≠ j) :
    (directKey t i).disjoint (directKey t j) :=
  Allocation.indices_disjoint (allocation t) _ _ (by omega)

theorem output_count {x y output : Tensor} (d : Domain x y output) :
    output.metadata.numel = x.metadata.numel := congrArg List.prod d.outputShape

/-- Exact four-byte pointer, alignment and backing bounds on every logical
coordinate. The same keys are used in the executable shared-memory model. -/
def Addresses (t : Tensor) : Prop :=
  ∀ i, i < t.metadata.numel →
    (t.dataPtr + BitVec.ofNat 64 i * BitVec.ofNat 64 4).toNat = (key t i).address ∧
    (key t i).address % 4 = 0 ∧
    ∀ byte, byte < 4 → (key t i).address+byte < t.storage.base+4*t.storage.elements ∧
      (key t i).address+byte < 2^64

theorem addresses (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous = Bool.true)
    (aligned : 0 < t.metadata.numel → t.dataPtr.toNat % 4 = 0) : Addresses t := by
  intro i hi
  have positive : 0 < t.metadata.numel := by omega
  have pointer := laws.pointer positive
  have align := aligned positive
  have backing := (key_backed t laws i hi).2.2
  have span := t.storage.spanFits
  have addr : (key t i).address = t.storage.base+4*t.storageOffset+4*i := by
    simp only [key,allocation,Allocation.key,laws.contiguous contiguous i hi]
    omega
  change (key t i).address+4 ≤ t.storage.base+4*t.storage.elements at backing
  have baseBound : t.storage.address t.storageOffset < 2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    rw [addr] at backing
    omega
  rw [pointer,BitVec.toNat_ofNat,Nat.mod_eq_of_lt baseBound] at align
  refine ⟨?_,?_,?_⟩
  · rw [pointer,←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
    have exact : t.storage.address t.storageOffset+i*4 = (key t i).address := by
      simp only [VeriTile.Torch.FP32.Storage.address,addr]
      omega
    rw [exact,Nat.mod_eq_of_lt (by omega)]
  · rw [addr]
    unfold VeriTile.Torch.FP32.Storage.address at align
    omega
  · intro byte hb
    constructor <;> omega

open VeriTile.Bench.TritonBenchG.VectorAddition.Finite

def Control (n : Nat) : Prop :=
  ∀ pid lane, pid < gridCount n → lane < 1024 →
    (BitVec.ofNat 32 pid).toInt = (pid : Int) ∧
    (BitVec.ofNat 32 lane).toInt = (lane : Int) ∧
    (BitVec.ofNat 32 n).toInt = (n : Int) ∧
    (start32 pid).toInt = (pid*1024 : Nat) ∧
    (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane).toInt =
      (pid*1024+lane : Nat) ∧
    VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32 n pid lane =
      decide (pid*1024+lane < n)

theorem control {x y output : Tensor} (d : Domain x y output) : Control x.metadata.numel := by
  intro pid lane hp hl
  have arithmetic := source_index_arithmetic x.metadata.numel pid lane d.admitted.extent hp hl
  have bound := launched_index_i32 x.metadata.numel pid lane d.admitted.extent hp hl
  exact ⟨nat32_signed pid (by omega),nat32_signed lane (by omega),
    nat32_signed _ d.admitted.extent,arithmetic⟩

/-- Actual source sign extension, byte stride and supplied data_ptr compose
with the Tensor's logical coordinate address. -/
theorem native_pointer (t : Tensor) (bound : t.metadata.numel < 2^31)
    (access : Addresses t) (pid lane : Nat) (hp : pid < gridCount t.metadata.numel)
    (hl : lane < 1024) (active : pid*1024+lane < t.metadata.numel) :
    (t.dataPtr + (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane).signExtend 64 *
      BitVec.ofNat 64 4).toNat = (key t (pid*1024+lane)).address := by
  rw [Native.offset_word t.metadata.numel pid ⟨bound,hp⟩ lane hl]
  exact (access _ active).1

def SourcePointers (x y output : Tensor) : Prop :=
  ∀ pid lane, pid < gridCount x.metadata.numel → lane < 1024 →
    pid*1024+lane < x.metadata.numel →
    (∀ t, t=x ∨ t=y ∨ t=output →
      (t.dataPtr + (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane).signExtend 64 *
        BitVec.ofNat 64 4).toNat = (key t (pid*1024+lane)).address)

theorem source_pointers {x y output : Tensor} (d : Domain x y output)
    (xa : Addresses x) (ya : Addresses y) (oa : Addresses output) : SourcePointers x y output := by
  intro pid lane hp hl active t role
  rcases role with hx | hy | ho
  · subst t
    exact native_pointer x d.admitted.extent xa pid lane hp hl active
  · subst t
    have count := same_numel x.metadata y.metadata d.admitted
    exact native_pointer y (by rw [←count];exact d.admitted.extent) ya pid lane
      (by rw [←count];exact hp) hl (by rw [←count];exact active)
  · subst t
    have count := output_count d
    exact native_pointer output (by rw [count];exact d.admitted.extent) oa pid lane
      (by rw [count];exact hp) hl (by rw [count];exact active)

#axiomsClean source_pointers
#axiomsClean direct_logical
#axiomsClean addresses
#axiomsClean native_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Kernel
