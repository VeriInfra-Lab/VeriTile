import bench.tritonbench_g.fast_rms_layernorm.Proofs.KernelTensor
import bench.tritonbench_g.fast_rms_layernorm.Proofs.FPSequence

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.KernelTensor
open VeriTile.Triton KernelStorage
noncomputable section

def Frame {γ : Type} (h : Call) (phase : Phase) (initial final : RegionName × Nat → γ) : Prop :=
  ∀ role off,¬(role∈outputs phase ∧ off<cells h role) → final (role,off)=initial (role,off)

def zeroWeight {γ : Type} (h : Call) (zero : γ) (initial : RegionName × Nat → γ) : RegionName × Nat → γ :=
  fun a => if a.1="dw" ∧ a.2<h.cols then zero else initial a

theorem zeroWeight_frame {γ : Type} (h : Call) (zero : γ) (initial : RegionName × Nat → γ) :
    Frame h .backward initial (zeroWeight h zero initial) := by
  intro role off outside
  simp only [zeroWeight]
  split
  · rename_i active
    exact False.elim (outside (by rcases active with ⟨rfl,hi⟩;simpa [outputs,cells] using hi))
  · rfl

theorem forward_frame {γ : Type} (h : Call) (initial final : RegionName × Nat → γ)
    (frame : ∀ addr,(∀ row : Fin h.rows,¬FPGridForward.writes h row addr) → final addr=initial addr) :
    Frame h .forward initial final := by
  intro q o outside
  apply frame
  intro row
  rintro (⟨j,eq⟩|eq)
  · have qr:=congrArg Prod.fst eq
    have oo:=congrArg Prod.snd eq
    simp only [Prod.fst,Prod.snd] at qr oo
    subst q;subst o
    exact outside ⟨by simp [outputs],by simpa [cells] using Native.offset_bound h row.val j.val row.isLt j.isLt⟩
  · have qr:=congrArg Prod.fst eq
    have oo:=congrArg Prod.snd eq
    simp only [Prod.fst,Prod.snd] at qr oo
    subst q;subst o
    exact outside ⟨by simp [outputs],by simpa [cells] using row.isLt⟩

theorem backward_frame {γ : Type} (h : Call) (initial final : RegionName × Nat → γ)
    (frame : ∀ addr,(∀ row : Fin h.rows,¬FPGridBackward.writes h row addr) →
      (∀ pid : Fin (Geometry.featureCount h),¬FPGridWeight.writes h pid addr) → final addr=initial addr) :
    Frame h .backward initial final := by
  intro q o outside
  apply frame
  · intro row
    rintro ⟨j,eq⟩
    have qr:=congrArg Prod.fst eq
    have oo:=congrArg Prod.snd eq
    simp only [FPGridBackward.address,Prod.fst,Prod.snd] at qr oo
    subst q;subst o
    exact outside ⟨by simp [outputs],by simpa [cells] using Native.offset_bound h row.val j.val row.isLt j.isLt⟩
  · intro pid
    rintro ⟨j,eq⟩
    have qr:=congrArg Prod.fst eq
    have oo:=congrArg Prod.snd eq
    simp only [FPGridWeight.address,Prod.fst,Prod.snd] at qr oo
    subst q;subst o
    exact outside ⟨by simp [outputs],by simpa [cells] using j.property⟩

#axiomsClean zeroWeight_frame
#axiomsClean forward_frame
#axiomsClean backward_frame
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.KernelTensor
