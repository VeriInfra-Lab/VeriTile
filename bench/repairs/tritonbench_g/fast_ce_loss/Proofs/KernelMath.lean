import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelReal
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelMath
open VeriTile.Triton FP.Structural KernelSource KernelModel
noncomputable section
set_option maxHeartbeats 4000000
namespace CE
abbrev blockLSE := VeriTile.Bench.Repairs.CrossEntropy1.Math.blockLSE
abbrev lse := VeriTile.Bench.Repairs.CrossEntropy1.Math.lse
abbrev blocks := VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks
end CE

def blockLSE (p : Parameters) (block : Nat) (x : Nat → ℝ) : ℝ :=
  CE.blockLSE p.vocab p.block block (fun j => Math.transformed p.softcap p.scaling (x j))
def fullLSE (p : Parameters) (x : Nat → ℝ) : ℝ :=
  CE.lse p.vocab (fun j => Math.transformed p.softcap p.scaling (x j))
def rowLoss (p : Parameters) (x : Nat → ℝ) (target : Int) : ℝ :=
  if target = -100 then 0 else fullLSE p x-Math.transformed p.softcap p.scaling (x target.toNat)
def partialLoss (p : Parameters) (x : Nat → ℝ) (target : Int) : ℝ :=
  if target = -100 then 0 else -(Math.transformed p.softcap p.scaling (x target.toNat))

 theorem lse_block (p : Parameters) (initial : State (WithBot ℝ))
    (flags : KernelReal.Flags p) (x : Nat → ℝ) (chunk : Nat)
    (positive : 0<p.block) (nonempty : chunk*p.block<p.vocab)
    (inputs : ∀ col<p.vocab,
      (initial.mem p.logits (initial.pids 0*p.logitsStride+col)).read .real=some (x col)) :
    lseValue KernelReal.algebra p initial chunk=some (blockLSE p chunk x) := by
  rw [KernelReal.lse_finite p initial flags x chunk positive nonempty inputs]
  congr 2
  rw [Finset.sum_filter]
  exact Fin.sum_univ_eq_sum_range
    (fun i => if chunk*p.block+i<p.vocab then
      Real.exp (Math.transformed p.softcap p.scaling (x (chunk*p.block+i))) else 0) p.block

 theorem lse_full (p : Parameters) (initial : State (WithBot ℝ))
    (flags : KernelReal.Flags p) (x : Nat → ℝ) (positive : 0<p.vocab) (covered : p.vocab≤p.block)
    (inputs : ∀ col<p.vocab,
      (initial.mem p.logits (initial.pids 0*p.logitsStride+col)).read .real=some (x col)) :
    lseValue KernelReal.algebra p initial 0=some (fullLSE p x) := by
  rw [lse_block p initial flags x 0 (by omega) (by simpa using positive) inputs]
  exact congrArg some (VeriTile.Bench.Repairs.CrossEntropy1.Math.block_zero_lse _ _ covered _)

 theorem loss_full (p : Parameters) (initial : State (WithBot ℝ))
    (flags : KernelReal.Flags p) (x : Nat → ℝ) (positive : 0<p.vocab) (covered : p.vocab≤p.block)
    (inputs : ∀ col<p.vocab,
      (initial.mem p.logits (initial.pids 0*p.logitsStride+col)).read .real=some (x col))
    (valid : label p initial = -100 ∨ (0≤label p initial ∧ label p initial<(p.vocab : Int))) :
    forwardLoss KernelReal.algebra p initial=some (rowLoss p x (label p initial)) := by
  by_cases ignored : label p initial = -100
  · simp [forwardLoss,rowLoss,ignored,literal,KernelReal.algebra]
  · have active : (label p initial).toNat<p.vocab := by
      have range := valid.resolve_left ignored
      omega
    have gather : gathered KernelReal.algebra p initial=some (x (label p initial).toNat) := by
      simp only [gathered,inputs _ active,KernelReal.algebra,id_eq]
    simp only [forwardLoss,rowLoss,ignored,if_false,
      lse_full p initial flags x positive covered inputs,gather,
      KernelReal.transform_finite p flags]
    rfl

 theorem loss_partial (p : Parameters) (initial : State (WithBot ℝ))
    (flags : KernelReal.Flags p) (x : Nat → ℝ)
    (inputs : ∀ col<p.vocab,
      (initial.mem p.logits (initial.pids 0*p.logitsStride+col)).read .real=some (x col))
    (valid : label p initial = -100 ∨ (0≤label p initial ∧ label p initial<(p.vocab : Int))) :
    chunkLoss KernelReal.algebra p initial=some (partialLoss p x (label p initial)) := by
  by_cases ignored : label p initial = -100
  · simp [chunkLoss,partialLoss,ignored,literal,KernelReal.algebra]
  · have active : (label p initial).toNat<p.vocab := by
      have range := valid.resolve_left ignored
      omega
    have gather : chunkTarget KernelReal.algebra p initial=some (x (label p initial).toNat) := by
      simp only [chunkTarget,inputs _ active,KernelModel.cast,KernelReal.algebra,id_eq]
    simp only [chunkLoss,partialLoss,ignored,if_false,gather,KernelReal.transform_finite p flags]
    simp [KernelModel.cast,binary,literal,KernelReal.algebra,RealProof.arithmetic]

 theorem chunk_start (p : Parameters) (positive : 0<p.block) (chunk : Nat)
    (launched : chunk<CE.blocks p.vocab p.block) : chunk*p.block<p.vocab :=
  VeriTile.Bench.Repairs.CrossEntropy1.Math.block_starts _ _ _ positive launched

#axiomsClean lse_block
#axiomsClean lse_full
#axiomsClean loss_full
#axiomsClean loss_partial
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelMath
