import bench.repairs.tritonbench_g.dropout_triton.Contract
import bench.repairs.tritonbench_g.dropout_triton.Kernels
import VeriTile.Triton.Semantics.LaunchIndex

namespace VeriTile.Bench.Repairs.DropoutTriton

/-- Named fields describe exactly the concrete supported metadata conditions. -/
theorem tensor_contract_iff (x y : TensorMetadata) :
    TensorContract x y ↔
      (x.layout = .strided ∧ y.layout = .strided) ∧
      (x.dtype = .floating .fp32 ∧ y.dtype = .other "int32") ∧
      (x.device.kind = "cuda" ∧ y.device.kind = "cuda") ∧
      x.device = y.device ∧ x.shape = y.shape ∧
      (x.isContiguous = Bool.true ∧ y.isContiguous = Bool.true) ∧
      (x.isNeg = Bool.false ∧ y.isNeg = Bool.false) ∧ x.numel < 2^31 ∧
      (0 < x.numel → x.dataPtr.toNat % 4 = 0 ∧ y.dataPtr.toNat % 4 = 0) := by
  constructor
  · intro h
    exact ⟨h.layout, h.dtype, h.cuda, h.sameDevice, h.sameShape, h.contiguous, h.noNeg,
      h.extent, h.alignment⟩
  · rintro ⟨hl, hd, hc, hv, hs, ht, hn, he, ha⟩
    exact ⟨hl, hd, hc, hv, hs, ht, hn, he, ha⟩

/-- The attribute checks accept exactly the named input contract. -/
theorem checkTensorInputs_iff (x y : TensorMetadata) :
    checkTensorInputs x y = .ok () ↔ TensorContract x y := by
  rw [tensor_contract_iff]
  simp [checkTensorInputs, VeriTile.Torch.TensorMetadata.isCuda, ite_eq_iff]

theorem checkScalar_iff (p : ScalarArgument) :
    checkScalar p = .ok () ↔ ScalarContract p := by
  classical
  cases p with
  | other => simp [checkScalar, ScalarContract]
  | «float» value =>
    cases value with
    | none => simp [checkScalar, ScalarContract]
    | some q => simp [checkScalar, ScalarContract]

theorem checkInputs_iff (x y : ArgumentMetadata) (p : ScalarArgument) :
    checkInputs x y p = .ok () ↔ AdmittedArguments x y p := by
  cases x <;> cases y
  all_goals try simp [checkInputs, AdmittedArguments]
  rename_i x y
  change (checkTensorInputs x y >>= fun _ => checkScalar p) = .ok () ↔ InputContract x y p
  cases h : checkTensorInputs x y with
  | error e =>
    simp only [Except.bind, bind, reduceCtorEq, false_iff]
    intro accepted
    have := (checkTensorInputs_iff x y).mpr accepted.tensors
    rw [h] at this
    contradiction
  | ok u =>
    cases u
    have ht := (checkTensorInputs_iff x y).mp h
    simp only [Except.bind, bind, checkScalar_iff]
    exact ⟨fun hp => ⟨ht,hp⟩, fun hp => hp.scalar⟩

theorem same_numel (x y : TensorMetadata) (h : TensorContract x y) :
    x.numel = y.numel := congrArg List.prod h.sameShape

def offset32 (pid lane : Nat) : BitVec 32 :=
  BitVec.ofNat 32 pid * 1024 + BitVec.ofNat 32 lane

def mask32 (n pid lane : Nat) : Bool :=
  (offset32 pid lane).slt (BitVec.ofNat 32 n)

theorem nat32_signed (n : Nat) (bound : n < 2^31) :
    (BitVec.ofNat 32 n).toInt = (n : Int) :=
  BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n : Int))
    (by decide) (by omega) (by norm_num; omega)

theorem native_indices (x y : TensorMetadata) (h : TensorContract x y)
    (pid lane : Nat) (hp : pid < gridCount x.numel) (hl : lane < 1024) :
    (offset32 pid lane).toInt = (pid * 1024 + lane : Nat) ∧
    mask32 x.numel pid lane = decide (pid * 1024 + lane < x.numel) := by
  have bound : pid * 1024 + lane < 2^31 :=
    VeriTile.Triton.LaunchIndex.launched_index_i32
      x.numel pid lane h.extent hp hl
  have word : offset32 pid lane = BitVec.ofNat 32 (pid * 1024 + lane) := by
    simp only [offset32, show (1024 : BitVec 32) = BitVec.ofNat 32 1024 from rfl,
      BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
  have signed : (offset32 pid lane).toInt = (pid * 1024 + lane : Nat) := by
    rw [word]; exact nat32_signed _ bound
  refine ⟨signed, ?_⟩
  rw [mask32, BitVec.slt_eq_decide, signed, nat32_signed _ h.extent]
  simp only [Int.ofNat_lt]

/-- Concrete original-source index overflow beyond this supported contract. -/
theorem first_wrapped_offset : (offset32 (2^21) 0).toInt = -2147483648 := by
  decide +kernel

#axiomsClean first_wrapped_offset
#axiomsClean checkInputs_iff
#axiomsClean native_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DropoutTriton
