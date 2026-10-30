import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelResult
import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelNativeProfile

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelNativeResult
open VeriTile.Triton FP.Structural KernelSource KernelNativeProfile
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section
set_option maxHeartbeats 1500000

def tile (p : Parameters) (values : Nat → Nat → Word)
    (i : TileIndex [p.height, p.width]) : Word :=
  if i.1.val < p.heads ∧ i.2.1.val < p.columns then values i.1.val i.2.1.val else zero

def maximum (a : Arithmetic) (p : Parameters) (values : Nat → Nat → Word) :
    Values Word .real [p.height] :=
  KernelNativeProfile.reduction a ⟨1, by simp⟩ Bool.false (fun i => a.absolute (tile p values i))

def scale (a : Arithmetic) (p : Parameters) (values : Nat → Nat → Word) (h : Fin p.height) : Word :=
  a.divFull (maximum a p values (h, PUnit.unit)) levels

/-- Independent native formula over caller Tensor words. Complete reduction
shape/order and all native callbacks are preserved, including both FP32 divisions and total selected PTX byte conversion. -/
def code (a : Arithmetic) (p : Parameters) (values : Nat → Nat → Word)
    (i : TileIndex [p.height, p.width]) : Int :=
  PtxSigned.signedByte (a.divFull (tile p values i)
    (if a.equal (maximum a p values (i.1, PUnit.unit)) zero then one
      else scale a p values i.1))

theorem loaded (a : Arithmetic) (rest : Algebra Word) (p : Parameters) (s : State Word)
    (values : Nat → Nat → Word)
    (reads : ∀ h c, h < p.heads → c < p.columns →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h c)).read .real = values h c) :
    KernelResult.loaded (algebra a rest) p s = tile p values := by
  funext i
  by_cases hi : i.1.val < p.heads ∧ i.2.1.val < p.columns
  · simp [KernelResult.loaded, KernelResult.active, algebra, tile, hi, reads _ _ hi.1 hi.2]
  · simp [KernelResult.loaded, KernelResult.active, algebra, tile, hi]

theorem maximum_eq (a : Arithmetic) (rest : Algebra Word) (p : Parameters) (s : State Word)
    (values : Nat → Nat → Word)
    (reads : ∀ h c, h < p.heads → c < p.columns →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h c)).read .real = values h c) :
    KernelResult.maximum (algebra a rest) p s = maximum a p values := by
  simp only [KernelResult.maximum, loaded a rest p s values reads]
  simp [algebra, maximum]

theorem scale_eq (a : Arithmetic) (rest : Algebra Word) (p : Parameters) (s : State Word)
    (values : Nat → Nat → Word)
    (reads : ∀ h c, h < p.heads → c < p.columns →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h c)).read .real = values h c)
    (h : Fin p.height) :
    KernelResult.scale (algebra a rest) p s h = scale a p values h := by
  unfold KernelResult.scale
  rw [maximum_eq a rest p s values reads]
  simp [algebra, scale]

theorem code_eq (a : Arithmetic) (rest : Algebra Word) (p : Parameters) (s : State Word)
    (values : Nat → Nat → Word)
    (reads : ∀ h c, h < p.heads → c < p.columns →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h c)).read .real = values h c)
    (i : TileIndex [p.height, p.width]) :
    KernelResult.code (algebra a rest) PtxSigned.signedByte a.equal p s i = code a p values i := by
  unfold KernelResult.code KernelResult.denominator
  rw [loaded a rest p s values reads, maximum_eq a rest p s values reads,
    scale_eq a rest p s values reads]
  simp [algebra, code]

#axiomsClean loaded
#axiomsClean maximum_eq
#axiomsClean scale_eq
#axiomsClean code_eq
#specNonCircular code avoiding [KernelSource.kernel, FP.Structural.exec, KernelExecution.execution]
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelNativeResult
