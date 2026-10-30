import bench.tritonbench_g.cross_entropy2.Proofs.Math
import bench.tritonbench_g.cross_entropy1.Proofs.Real

namespace VeriTile.Bench.Repairs.CrossEntropy2.RealSource
open VeriTile.Triton Source
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000

abbrev algebra := VeriTile.Bench.Repairs.CrossEntropy1.RealSource.algebra
abbrev Input := VeriTile.Bench.Repairs.CrossEntropy1.RealSource.Input

def scaledMemory (a : ℝ) (m : Memory (WithBot ℝ)) : Memory (WithBot ℝ) :=
  fun r i => VeriTile.Bench.Repairs.CrossEntropy1.Source.Cell.mk .real (binary algebra .mul ((m r i).read .real) (fp algebra a))

theorem loaded_transfer (p : Parameters) (m : Memory (WithBot ℝ)) (row block : Nat) :
    VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits (scaled algebra p.logitScale) p.toParameters m row block=
      VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits algebra p.toParameters (scaledMemory p.logitScale m) row block := by
  funext lane
  unfold VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits
  split
  · simp [VeriTile.Bench.Repairs.CrossEntropy1.Source.load,scaled,scaledMemory,algebra,Source.binary,Source.fp,
      VeriTile.Bench.Repairs.CrossEntropy1.Source.binary,VeriTile.Bench.Repairs.CrossEntropy1.Source.fp,
      VeriTile.Bench.Repairs.CrossEntropy1.RealSource.algebra]
  · rfl

theorem values_transfer (p : Parameters) (m : Memory (WithBot ℝ)) (row block : Nat) :
    values algebra p m row block=
      VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues algebra p.toParameters
        (scaledMemory p.logitScale m) row block := by
  unfold values VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues
  rw [loaded_transfer]
  rfl

theorem scaled_input (p : Parameters) (m : Memory (WithBot ℝ)) (row : Nat) (x : Nat → ℝ)
    (input : Input p.toParameters m row x) :
    Input p.toParameters (scaledMemory p.logitScale m) row (Math.scaled p.logitScale x) := by
  intro col active
  simp only [scaledMemory,VeriTile.Bench.Repairs.CrossEntropy1.Source.Cell.read_same,input col active]
  rfl

theorem block_lse (p : Parameters) (m : Memory (WithBot ℝ)) (row block : Nat) (x : Nat → ℝ)
    (input : Input p.toParameters m row x) (positive : 0<p.block) (start : block*p.block<p.cols) :
    (values algebra p m row block).lse=
      ↑(VeriTile.Bench.Repairs.CrossEntropy1.Math.blockLSE p.cols p.block block (Math.scaled p.logitScale x)) := by
  rw [values_transfer]
  exact VeriTile.Bench.Repairs.CrossEntropy1.RealSource.forward_lse _ _ _ _ _
    (scaled_input p m row x input) positive start

theorem block_sum (p : Parameters) (m : Memory (WithBot ℝ)) (row block : Nat) (x : Nat → ℝ)
    (input : Input p.toParameters m row x) (smooth : p.hasSmoothing=Bool.true) :
    (values algebra p m row block).sumLogits=
      some (↑(VeriTile.Bench.Repairs.CrossEntropy1.Math.blockSum p.cols p.block block (Math.scaled p.logitScale x))) := by
  rw [values_transfer]
  exact VeriTile.Bench.Repairs.CrossEntropy1.RealSource.forward_sum _ _ _ _ _
    (scaled_input p m row x input) smooth

/-- All branches of the actual scalar registers, including the repaired
non-owner unsplit branch and the explicitly stored split z-loss. -/
theorem scalar_outputs (p : Parameters) (m : Memory (WithBot ℝ)) (row block : Nat) (x : Nat → ℝ)
    (input : Input p.toParameters m row x) (target : Int) (positive : 0<p.block)
    (start : block*p.block<p.cols) :
    let L := VeriTile.Bench.Repairs.CrossEntropy1.Math.blockLSE p.cols p.block block (Math.scaled p.logitScale x)
    let S := VeriTile.Bench.Repairs.CrossEntropy1.Math.blockSum p.cols p.block block (Math.scaled p.logitScale x)
    let h := Math.smoothing p.smoothing p.hasSmoothing
    lossZ algebra p m row block target (values algebra p m row block)=
      (↑(if target=p.ignored then 0 else
         (if p.split then 0 else L)-h*S/p.totalClasses-
          (if blockOwns p target block then (1-h)*Math.scaled p.logitScale x (target-p.classStart).toNat else 0)+
          (if p.split then 0 else p.squareScale*L*L)),
       ↑(if target=p.ignored then 0 else if p.split then 0 else p.squareScale*L*L)) := by
  dsimp only
  have lseEq := block_lse p m row block x input positive start
  by_cases ignored : target=p.ignored
  · simp [lossZ,ignored,fp,algebra,VeriTile.Bench.Repairs.CrossEntropy1.Source.fp,
      VeriTile.Bench.Repairs.CrossEntropy1.RealSource.algebra]
  · by_cases inside : blockOwns p target block
    · have active : (target-p.classStart).toNat<p.cols := by
        unfold blockOwns at inside
        omega
      have selected : load algebra m p.logits
          (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row (target-p.classStart).toNat)=
            ↑(x (target-p.classStart).toNat) := by exact input _ active
      cases hs : p.hasSmoothing <;> cases sp : p.split
      all_goals
        try have sums := block_sum p m row block x input hs
        simp only [lossZ,ignored,inside,hs,sp,Bool.false_eq_true,↓reduceIte,lseEq,Math.smoothing]
        try rw [sums]
        simp only [Option.getD_some,selected]
        simp only [Source.fp,Source.nat,Source.binary,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.fp_coe,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.nat_coe,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.add_coe,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.sub_coe,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.mul_coe,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.div_coe,Nat.cast_one]
        apply Prod.ext
        · apply congrArg (fun v : ℝ => (↑v : WithBot ℝ)) <;> dsimp [Math.scaled] <;> ring
        · rfl
    · cases hs : p.hasSmoothing <;> cases sp : p.split
      all_goals
        try have sums := block_sum p m row block x input hs
        simp only [lossZ,ignored,inside,hs,sp,Bool.false_eq_true,↓reduceIte,lseEq,Math.smoothing]
        try rw [sums]
        simp only [Option.getD_some,Source.fp,Source.nat,Source.binary,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.fp_coe,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.nat_coe,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.add_coe,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.sub_coe,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.mul_coe,
          VeriTile.Bench.Repairs.CrossEntropy1.RealSource.div_coe,Nat.cast_one]
        apply Prod.ext
        · apply congrArg (fun v : ℝ => (↑v : WithBot ℝ)) <;> ring
        · rfl

#axiomsClean block_lse
#axiomsClean scalar_outputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.RealSource
