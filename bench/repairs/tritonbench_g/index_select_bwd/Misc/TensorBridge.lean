import bench.repairs.tritonbench_g.index_select_bwd.Correct
import bench.repairs.tritonbench_g.index_select_bwd.FPEquiv
import bench.repairs.tritonbench_g.index_select_bwd.Proofs.KernelBinding

/-! Optional Tensor applications and Python-check correspondence. -/
namespace VeriTile.Bench.Repairs.IndexSelectBwd.TensorBridge
open VeriTile.Triton FP.Structural Physical
noncomputable section

local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

export Native (all_indices source_lane_cast slice_address)

/-- Fixed production block/stride, using precisely the asserted input domain.
Empty work is a wrapper branch; this theorem also covers count zero directly. -/
theorem index_select_bwd_checked_kernel_correctness (rest : Algebra ℝ)
    (r : Request) (checked : checkInputs r=Bool.true) (k : Nat)
    (s : State ℝ) (current incoming : Nat → ℝ)
    (inputs : KernelProof.InputValues (Program.layout r.C k) s current incoming) :
    Spec.Real (InputContract r ∧ ∃ t,
      exec (Source.backend (KernelProof.arithmetic rest (·+·))) (kernel (Program.layout r.C k)) s=some t ∧
      t.mem=KernelProof.reference (·+·) (Program.layout r.C k) s current incoming ∧
      (∀ i,i< min 512 (r.C-k*512) → t.mem Program.destinationRegion i=
        .mk .real (sliceTarget (·+·) current incoming i)) ∧
      (∀ region offset,region≠Program.destinationRegion ∨ min 512 (r.C-k*512)≤offset →
        t.mem region offset=s.mem region offset)) :=
  ⟨(Native.checkInputs_iff r).mp checked,
    KernelProof.successful rest (·+·) (Program.layout r.C k) s current incoming rfl
      (Nat.min_le_left _ _) inputs⟩

theorem python_checks_iff_preconditions (a : Arguments) :
    checkArguments a=Bool.true ↔ AdmittedArguments a := KernelProof.argument_checks_iff a

/-- Public caller-Tensor binding for one actual launch. Both inputs come from
one current byte-memory map, including any preceding duplicate-row updates;
no payload snapshot or assumed final output appears in the premises. -/
theorem index_select_bwd_checked_fp32_equiv (rest : Algebra Word)
    (nativeAdd : Word → Word → Word) (r : Request) (checked : checkInputs r=Bool.true)
    (objects : Objects r) (m : Memory) (storage : StorageBinding r)
    (observed : IndexBinding r m) (i : Fin r.N) (k : Nat)
    (launched : k<(r.C+511)/512) (surrounding : State Word) :
    let h := (Native.checkInputs_iff r).mp checked
    let p := Program.layout r.C k
    let s := KernelBinding.state r h m i k surrounding
    let current := KernelBinding.current r h objects m i k
    let incoming := KernelBinding.incoming r objects m i k
    KernelBinding.CheckedCorrespondence r h m i k ∧ ∃ t,exec (Source.backend (KernelProof.arithmetic rest nativeAdd)) (kernel p) s=some t ∧
      t.mem=KernelProof.reference nativeAdd p s current incoming ∧
      (∀ lane,lane<p.count → t.mem p.destination lane=
        .mk .real (sliceTarget nativeAdd current incoming lane)) ∧
      (∀ region offset,region≠p.destination ∨ p.count≤offset →
        t.mem region offset=s.mem region offset) := by
  dsimp only
  refine ⟨KernelBinding.checked_correspondence r _ m storage observed i k launched,?_⟩
  exact KernelProof.successful rest nativeAdd _ _ _ _ rfl (Nat.min_le_left _ _)
    (KernelBinding.inputs r _ objects m i k surrounding)

open KernelBinding in
/-- Checked tensor extents bound every clamped lane address and the native
int64 lane conversion. Physical storage observations rule out pointer wrap. -/
theorem accesses (r : Request) (h : InputContract r) (storage : StorageBinding r)
    (i : Fin r.N) (k lane : Nat) (hk : k<(r.C+511)/512) (hl : lane<512) :
    let safe := if lane < (min 512 (r.C-k*512)) then lane else 0
    destinationOffset r h i k safe<r.R*r.C ∧ incomingOffset r i k safe<r.N*r.C ∧
    (BitVec.ofInt 64 (lane : Int)).toInt=(lane : Int) ∧
    (BitVec.ofNat 64 (Native.address r.destination 4 (destinationOffset r h i k safe))).toNat=
      Native.address r.destination 4 (destinationOffset r h i k safe) ∧
    (BitVec.ofNat 64 (Native.address r.incoming 4 (incomingOffset r i k safe))).toNat=
      Native.address r.incoming 4 (incomingOffset r i k safe) :=
  KernelBinding.accesses r h storage i k lane hk hl
#axiomsClean index_select_bwd_checked_kernel_correctness
#axiomsClean python_checks_iff_preconditions
#axiomsClean index_select_bwd_checked_fp32_equiv
#axiomsClean accesses
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectBwd.TensorBridge
