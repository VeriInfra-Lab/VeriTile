import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelOutputs
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelInputs

namespace VeriTile.Bench.Repairs.FastLayernorm.KernelBound
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelInputs
noncomputable section
variable {α : Type} [Inhabited α] {read : Nat → α}

def Pointers (views : RegionName → Option (ReadView α read)) : Prop :=
  ∀ r v,views r=some v → 0<v.tensor.metadata.numel →
    BitVec.ofNat 64 ((VeriTile.Torch.FP32.dataView v.tensor v.laws (v.checked.dense rfl)).address 0)=v.tensor.dataPtr
theorem pointers (views : RegionName → Option (ReadView α read)) : Pointers views := actual_pointers views

theorem raw_logical (views : RegionName → Option (ReadView α read)) (u : State α) (r : RegionName) (stride B : Nat) :
    KernelRows.raw (logical views u) r stride B=fun j => value views u r (u.pids 0*stride+j.val) := by
  funext j
  exact logical_read views u r _

theorem forward (M : Algebra α) (N B : Nat) (eps : ℝ) (bias : Bool)
    (views : RegionName → Option (ReadView α read)) (u : State α) :
    Pointers views ∧ ∃ v,
      FP.Structural.exec (engine M) (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) (physical views u)=some v ∧
      v.mem=KernelRows.forwardHeap M N B eps bias (logical views u) ∧
      (∀ j : Fin B,j.val<N → v.mem "y" (u.pids 0*N+j.val)=
        KernelReference.output M N eps bias (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "w" k.val) (fun k => value views u "b" k.val) j) ∧
      v.mem "r" (u.pids 0)=KernelReference.savedInverse M N eps (fun k : Fin B => value views u "x" (u.pids 0*N+k.val)) ∧
      v.mem "mu" (u.pids 0)=KernelReference.savedMean M N (fun k : Fin B => value views u "x" (u.pids 0*N+k.val)) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) := by
  refine ⟨pointers views,?_⟩
  rw [physical_eq_logical]
  obtain ⟨v,he,hm,hy,hr,hmu,hf⟩ := KernelOutputs.forward_complete M N B eps bias (logical views u)
  refine ⟨v,he,hm,?_,?_,?_,hf⟩
  · simpa only [raw_logical,Nat.mul_zero,Nat.zero_add] using hy
  · simpa only [raw_logical] using hr
  · simpa only [raw_logical] using hmu

theorem backward (M : Algebra α) (N B : Nat)
    (views : RegionName → Option (ReadView α read)) (u : State α) :
    Pointers views ∧ ∃ v,
      FP.Structural.exec (engine M) (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) (physical views u)=some v ∧
      v.mem=KernelRows.backwardHeap M N B (logical views u) ∧
      (∀ j : Fin B,j.val<N → v.mem "dx" (u.pids 0*N+j.val)=
        KernelReference.gradient M N (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "dy" (u.pids 0*N+k.val)) (fun k => value views u "w" k.val)
          (value views u "mu" (u.pids 0)) (value views u "r" (u.pids 0)) j) ∧
      (∀ q o,¬KernelOutputs.backwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) := by
  refine ⟨pointers views,?_⟩
  rw [physical_eq_logical]
  obtain ⟨v,he,hm,hy,hf⟩ := KernelOutputs.backward_complete M N B (logical views u)
  refine ⟨v,he,hm,?_,hf⟩
  simpa only [raw_logical,logical_read,Nat.mul_zero,Nat.zero_add] using hy

def affineValues (M : Algebra α) (rows N : Nat) (views : RegionName → Option (ReadView α read))
    (u : State α) (j : Nat) : α × α :=
  KernelReference.affineCarry M (decide (j<N))
    (fun row => value views u "x" (row*N+(if j<N then j else 0)))
    (fun row => value views u "dy" (row*N+(if j<N then j else 0)))
    (value views u "mu") (value views u "r") rows

theorem saved_logical (views : RegionName → Option (ReadView α read)) (u : State α) (r : RegionName) :
    KernelAffine.saved (logical views u) r=value views u r := by
  funext row
  exact logical_read views u r row

theorem carry_logical (M : Algebra α) (rows N : Nat) (views : RegionName → Option (ReadView α read))
    (u : State α) (j : TileIndex [128]) :
    KernelAffine.carry M N (u.pids 0) (logical views u) rows j=affineValues M rows N views u (u.pids 0*128+j.1.val) := by
  simp only [KernelAffine.carry,KernelAffine.raw,saved_logical,logical_read,affineValues]

theorem affine (M : Algebra α) (rows N wc bc : Nat) (bias : Bool)
    (views : RegionName → Option (ReadView α read)) (u : State α) :
    Pointers views ∧ ∃ v,
      FP.Structural.exec (engine M) (KernelAffine.kernel rows N wc bc bias) (physical views u)=some v ∧
      v.mem=KernelAffine.heap M rows N wc bc (u.pids 0) bias (logical views u) ∧
      (∀ j : Fin 128,u.pids 0*128+j.val<wc → v.mem "dw" (u.pids 0*128+j.val)=
        Cell.mk .real (KernelReference.cast M (affineValues M rows N views u (u.pids 0*128+j.val)).1)) ∧
      (bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → v.mem "db" (u.pids 0*128+j.val)=
        Cell.mk .real (KernelReference.cast M (affineValues M rows N views u (u.pids 0*128+j.val)).2)) ∧
      (∀ q o,¬KernelOutputs.affineFootprint wc bc (u.pids 0) bias q o → v.mem q o=(physical views u).mem q o) := by
  refine ⟨pointers views,?_⟩
  rw [physical_eq_logical]
  obtain ⟨v,he,hm,hw,hb,hf⟩ := KernelOutputs.affine_complete M rows N wc bc bias (logical views u)
  refine ⟨v,he,hm,?_,?_,hf⟩
  · simpa only [show (logical views u).pids=u.pids from rfl,carry_logical] using hw
  · simpa only [show (logical views u).pids=u.pids from rfl,carry_logical] using hb

#axiomsClean forward
#axiomsClean backward
#axiomsClean affine
end
end VeriTile.Bench.Repairs.FastLayernorm.KernelBound
