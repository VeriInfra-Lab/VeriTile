import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.NativeIO
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GridSafety
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GraphLocalityGrid
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.OutputTags

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
open VeriTile Triton FP.Structural Native
open SoftmaxFlaggemsFiniteSeed
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem forward_native_pins (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (forwardNativeIO ops scalar R S X Y d c).Inputs) (s : State Term)
    (h : (forwardNativeIO ops scalar R S X Y d c).InputBinding v s) :
    Locality.Agree [X,Y] (count d) s (entry (forwardWords ops scalar R S X Y d c v)) := by
  intro r member k hk
  by_cases hx : r=X
  · subst r
    have pin := h (0 : Fin 2) (⟨k,hk⟩,PUnit.unit) trivial
    simpa [forwardNativeIO,layout,channels,index,AddressedProgramIO.InputBinding,mapValue,
      FPReference.read,entry,forwardWords,hk] using pin
  · have hy : r=Y := by simpa [hx] using member
    subst r
    have pin := h (1 : Fin 2) (⟨k,hk⟩,PUnit.unit) trivial
    simpa [forwardNativeIO,layout,channels,index,AddressedProgramIO.InputBinding,mapValue,
      FPReference.read,entry,forwardWords,hk,hx] using pin

theorem backward_native_pins (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (backwardNativeIO ops scalar R S P G Y d c).Inputs) (s : State Term)
    (h : (backwardNativeIO ops scalar R S P G Y d c).InputBinding v s) :
    Locality.Agree [P,G,Y] (count d) s (entry (backwardWords ops scalar R S P G Y d c v)) := by
  intro r member k hk
  by_cases hp : r=P
  · subst r
    have pin := h (0 : Fin 3) (⟨k,hk⟩,PUnit.unit) trivial
    simpa [backwardNativeIO,layout,channels,index,AddressedProgramIO.InputBinding,mapValue,
      FPReference.read,entry,backwardWords,hk] using pin
  · by_cases hg : r=G
    · subst r
      have pin := h (1 : Fin 3) (⟨k,hk⟩,PUnit.unit) trivial
      simpa [backwardNativeIO,layout,channels,index,AddressedProgramIO.InputBinding,mapValue,
        FPReference.read,entry,backwardWords,hk,hp] using pin
    · have hy : r=Y := by simpa [hp,hg] using member
      subst r
      have pin := h (2 : Fin 3) (⟨k,hk⟩,PUnit.unit) trivial
      simpa [backwardNativeIO,layout,channels,index,AddressedProgramIO.InputBinding,mapValue,
        FPReference.read,entry,backwardWords,hk,hp,hg] using pin

theorem forward_native (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B) :
    (forwardNativeIO ops scalar R S X Y d c).Implements
      (forwardNativeFormula ops scalar R S X Y d c) := by
  apply AddressedProgramIO.Implements.intro
  · intro v bounds backed s _
    have hx : ∀ k,k<count d → k<bounds X := by
      intro k hk
      exact backed.2.1 (0 : Fin 2) (⟨k,hk⟩,PUnit.unit) trivial
    have hy : ∀ k,k<count d → k<bounds Y := by
      intro k hk
      exact backed.2.2.1 (0 : Fin 1) (⟨k,hk⟩,PUnit.unit) trivial
    exact forward_safe (algebra R) X Y d c hN hK hB hone bounds hx hy s
  · intro v s inputs
    obtain ⟨final,ran,memory⟩ := FPGrid.forward_whole (algebra R) X Y d c hB s s rfl
    have pins := forward_native_pins ops scalar R S X Y d c v s inputs
    have same := Locality.forward (algebra R) [X,Y] X Y d c hK hB hone (by simp) _ _ pins
    refine ⟨final,(forwardJobs_run (algebra R) X Y d c s).trans ran,?_,?_⟩
    · intro o j _
      obtain ⟨o,ho⟩ := o
      have he : o=0 := Nat.lt_one_iff.mp ho
      subst o
      have sameY := same Y (by simp) j.1.val j.1.isLt
      obtain ⟨value,tag⟩ := OutputTags.forward (algebra R) X Y d c hN hK hB hC hone s j.1.val j.1.isLt
      change mapCell (fun t : Term => Scalar.eval (Scalar.instructions ops scalar)
        (S (Y,j.1.val)) (t (Y,j.1.val))) (final.mem Y j.1.val)=Cell.mk .real _
      rw [memory,tag]
      apply congrArg (Cell.mk .real)
      have value_eq : value=FPReference.read
          (Kernel.forwardTarget R X Y d c (forwardWords ops scalar R S X Y d c v)) Y j.1.val := by
        simpa [FPReference.read,tag,Kernel.forwardTarget] using sameY
      change Scalar.eval (Scalar.instructions ops scalar) (S (Y,j.1.val)) (value (Y,j.1.val))=_
      rw [value_eq]
      exact NativeProfile.lower_exact ops scalar S (Y,j.1.val) _
    · intro r k outside
      rw [memory]
      apply FPTrace.forward_frame (algebra R) X Y d c hN hK hB hC hone s r k
      rintro ⟨rfl,hk⟩
      exact (outside.1 (0 : Fin 1) (⟨k,hk⟩,PUnit.unit) trivial).elim (fun h => h rfl) (fun h => h rfl)

theorem backward_native (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B) :
    (backwardNativeIO ops scalar R S P G Y d c).Implements
      (backwardNativeFormula ops scalar R S P G Y d c) := by
  apply AddressedProgramIO.Implements.intro
  · intro v bounds backed s _
    have hp : ∀ k,k<count d → k<bounds P := by
      intro k hk
      exact backed.2.1 (0 : Fin 3) (⟨k,hk⟩,PUnit.unit) trivial
    have hg : ∀ k,k<count d → k<bounds G := by
      intro k hk
      exact backed.2.1 (1 : Fin 3) (⟨k,hk⟩,PUnit.unit) trivial
    have hy : ∀ k,k<count d → k<bounds Y := by
      intro k hk
      exact backed.2.2.1 (0 : Fin 1) (⟨k,hk⟩,PUnit.unit) trivial
    exact backward_safe (algebra R) P G Y d c hK bounds hp hg hy s
  · intro v s inputs
    obtain ⟨final,ran,memory⟩ := FPGrid.backward_whole (algebra R) P G Y d c s s rfl
    have pins := backward_native_pins ops scalar R S P G Y d c v s inputs
    have same := Locality.backward (algebra R) [P,G,Y] P G Y d c hK (by simp) (by simp) _ _ pins
    refine ⟨final,(backwardJobs_run (algebra R) P G Y d c s).trans ran,?_,?_⟩
    · intro o j _
      obtain ⟨o,ho⟩ := o
      have he : o=0 := Nat.lt_one_iff.mp ho
      subst o
      have sameY := same Y (by simp) j.1.val j.1.isLt
      obtain ⟨value,tag⟩ := OutputTags.backward (algebra R) P G Y d c hN hK hB hC hone s j.1.val j.1.isLt
      change mapCell (fun t : Term => Scalar.eval (Scalar.instructions ops scalar)
        (S (Y,j.1.val)) (t (Y,j.1.val))) (final.mem Y j.1.val)=Cell.mk .real _
      rw [memory,tag]
      apply congrArg (Cell.mk .real)
      have value_eq : value=FPReference.read
          (Kernel.backwardTarget R P G Y d c (backwardWords ops scalar R S P G Y d c v)) Y j.1.val := by
        simpa [FPReference.read,tag,Kernel.backwardTarget] using sameY
      change Scalar.eval (Scalar.instructions ops scalar) (S (Y,j.1.val)) (value (Y,j.1.val))=_
      rw [value_eq]
      exact NativeProfile.lower_exact ops scalar S (Y,j.1.val) _
    · intro r k outside
      rw [memory]
      apply FPTrace.backward_frame (algebra R) P G Y d c hN hK hB hC hone s r k
      rintro ⟨rfl,hk⟩
      exact (outside.1 (0 : Fin 1) (⟨k,hk⟩,PUnit.unit) trivial).elim (fun h => h rfl) (fun h => h rfl)

#axiomsClean forward_native
#axiomsClean backward_native
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
