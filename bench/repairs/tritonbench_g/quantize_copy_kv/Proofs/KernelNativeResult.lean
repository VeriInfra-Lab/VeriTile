import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelResult
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelNativeProfile

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelNativeResult
open VeriTile.Triton FP.Structural KernelSource KernelNativeProfile
open VeriTile.Bench.TritonBenchG.QuantizeKvTransform
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section
set_option maxHeartbeats 1500000

def tile (p : Parameters) (values : Nat → Nat → Word)
    (i : TileIndex [p.height, p.width]) : Word :=
  if i.1.val < p.heads then values i.1.val i.2.1.val else zero

def maximum (a : Arithmetic) (p : Parameters) (values : Nat → Nat → Word) :
    Values Word .real [p.height] :=
  a.reduceMax (some .fp32) ⟨1, by simp⟩ Bool.false (fun i => a.absolute (tile p values i))

def scale (a : Arithmetic) (p : Parameters) (values : Nat → Nat → Word) (h : Fin p.height) : Half :=
  a.roundHalf (a.divFull (maximum a p values (h, PUnit.unit)) levels)

/-- Independent native formula over caller Tensor words. Complete reduction
shape/order and all native callbacks are preserved, including FP16 rounding
before promotion and total selected PTX byte conversion. -/
def code (a : Arithmetic) (p : Parameters) (values : Nat → Nat → Word)
    (i : TileIndex [p.height, p.width]) : Int :=
  PtxSigned.signedByte (a.divFull (tile p values i)
    (if a.equal (maximum a p values (i.1, PUnit.unit)) zero then one
      else a.promoteHalf (scale a p values i.1)))

theorem loaded (a : Arithmetic) (rest : Algebra Word) (p : Parameters) (s : State Word)
    (values : Nat → Nat → Word)
    (reads : ∀ h c, h < p.heads → c < p.width →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h c)).read .real = values h c) :
    KernelResult.loaded (algebra a rest) p s = tile p values := by
  funext i
  by_cases hi : i.1.val < p.heads
  · simp [KernelResult.loaded, algebra, tile, hi, reads _ _ hi i.2.1.isLt]
  · simp [KernelResult.loaded, algebra, tile, hi, FP.Software.fp32Context]

theorem maximum_eq (a : Arithmetic) (rest : Algebra Word) (p : Parameters) (s : State Word)
    (values : Nat → Nat → Word)
    (reads : ∀ h c, h < p.heads → c < p.width →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h c)).read .real = values h c) :
    KernelResult.maximum (algebra a rest) p s = maximum a p values := by
  simp only [KernelResult.maximum, loaded a rest p s values reads]
  simp [algebra, maximum, FP.Software.fp32Context]

theorem scale_eq (a : Arithmetic) (rest : Algebra Word) (p : Parameters) (s : State Word)
    (values : Nat → Nat → Word)
    (reads : ∀ h c, h < p.heads → c < p.width →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h c)).read .real = values h c)
    (h : Fin p.height) :
    KernelResult.scale (algebra a rest) p s h = MixedBits.embed (scale a p values h) := by
  unfold KernelResult.scale
  rw [maximum_eq a rest p s values reads]
  simp [algebra, scale, FP.Software.fp32Context]

theorem code_eq (a : Arithmetic) (rest : Algebra Word) (p : Parameters) (s : State Word)
    (values : Nat → Nat → Word)
    (reads : ∀ h c, h < p.heads → c < p.width →
      (s.mem "input" (KernelResult.inputOffset p (s.pids 0) h c)).read .real = values h c)
    (i : TileIndex [p.height, p.width]) :
    KernelResult.code (algebra a rest) PtxSigned.signedByte a.equal p s i = code a p values i := by
  unfold KernelResult.code KernelResult.denominator
  rw [loaded a rest p s values reads, maximum_eq a rest p s values reads,
    scale_eq a rest p s values reads]
  simp [algebra, code, FP.Software.fp32Context, MixedBits.half_embedding]

#axiomsClean loaded
#axiomsClean maximum_eq
#axiomsClean scale_eq
#axiomsClean code_eq
#specNonCircular code avoiding [KernelSource.kernel, FP.Structural.exec, KernelExecution.execution]
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelNativeResult
