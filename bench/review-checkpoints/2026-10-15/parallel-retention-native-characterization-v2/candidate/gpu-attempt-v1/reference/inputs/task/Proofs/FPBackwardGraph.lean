import bench.tritonbench_g.parallel_retention_attention.Proofs.FPForwardGraph
import bench.tritonbench_g.parallel_attention.Proofs.FPBackwardGraph
namespace VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.FPStages
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig BackwardAccumulator matrixRead matrixTranspose backwardKey backwardValue descendingCount descendingStart foldRange backwardOutputMemory)
noncomputable section

/-- The shared stage record's Z/DZ fields are unused by retention. -/
def backwardKernel (c : BackwardConfig) : ComputeKernel :=
  backwardSource c.q c.k c.v c.do_ c.dk c.dv c.i_bh c.i_c c.i_k c.i_v c.i_h
    c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.scale
    c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV

def decayedKey {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig) (s : State α) :
    TileIndex [c.BTL,c.BK] → α := fun i =>
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul (backwardKey M c s i)
    (rowDecay M (logDecay M c.i_h) (c.BTL-i.1.val)))

/-- Complete value update for each original DK/DV loop. The diagonal signed
difference, cast of b_b, dot operands, multiply order and accumulation order
are literal finite-value primitive calls rather than real-algebra identities. -/
def backwardUpdate {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) (diagonal : Bool) (start : Nat)
    (previous : BackwardAccumulator α c.BTL c.BK c.BV) : BackwardAccumulator α c.BTL c.BK c.BV :=
  let b := logDecay M c.i_h
  let query := matrixRead M s c.q (c.i_bh*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
    (c.i_k*c.BK) start c.BK c.BTS
  let doRaw := matrixRead M s c.do_ (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
    (c.i_v*c.BV) start c.BV c.BTS
  let outGrad := if diagonal then doRaw else fun i => M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .mul (doRaw i) (rowDecay M b i.2.1.val))
  let key := if diagonal then backwardKey M c s
    else fun i => M.cast (some .fp32) .real .real (decayedKey M c s i)
  let scoreRaw := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := []) key query
  let dsRaw := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := []) (backwardValue M c s) outGrad
  let decay := fun (i : TileIndex [c.BTL,c.BTS]) =>
    let masked := if i.1.val ≤ start-c.i_c*c.BTL+i.2.1.val then
      M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul
        (M.fromInt (some .fp32) (-(i.1.val : Int) + (start-c.i_c*c.BTL+i.2.1.val : Nat)))
        (M.cast (some .fp32) .real .real b))
      else M.literal (some .fp32) .real 0.0
    M.binary (some .fp32) .real .mul masked (M.literal (some .fp32) .real c.scale)
  let score := fun i => if diagonal then M.binary (some .fp32) .real .mul (scoreRaw i) (decay i) else scoreRaw i
  let ds := fun i => if diagonal then M.binary (some .fp32) .real .mul (dsRaw i) (decay i) else dsRaw i
  let dv := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (fun i => M.cast (some .fp32) .real .real (score i)) (matrixTranspose outGrad)
  let dk := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (fun i => M.cast (some .fp32) .real .real (ds i)) (matrixTranspose query)
  { key := fun i => M.binary (some .fp32) .real .add
      (if diagonal then previous.key i else M.binary (some .fp32) .real .mul (previous.key i) (blockDecay M b c.BTS)) (dk i)
    value := fun i => M.binary (some .fp32) .real .add
      (if diagonal then previous.value i else M.binary (some .fp32) .real .mul (previous.value i) (blockDecay M b c.BTS)) (dv i) }

def backwardBetweenValues {α : Type} (M : Algebra α) (c : BackwardConfig)
    (previous : BackwardAccumulator α c.BTL c.BK c.BV) : BackwardAccumulator α c.BTL c.BK c.BV :=
  { key := fun i => M.binary (some .fp32) .real .mul (previous.key i)
      (M.binary (some .fp32) .real .mul (rowDecay M (logDecay M c.i_h) (c.BTL-i.1.val)) (M.literal (some .fp32) .real c.scale))
    value := fun i => M.binary (some .fp32) .real .mul (previous.value i) (M.literal (some .fp32) .real c.scale) }

def backwardValues {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) : BackwardAccumulator α c.BTL c.BK c.BV :=
  let initial : BackwardAccumulator α c.BTL c.BK c.BV :=
    ⟨fun _ => M.literal (some .fp32) .real 0, fun _ => M.literal (some .fp32) .real 0⟩
  let before := foldRange (fun j => backwardUpdate M D c s Bool.false (descendingStart c j))
    0 (descendingCount c) 1 initial
  foldRange (backwardUpdate M D c s Bool.true) (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) c.BTS
    (backwardBetweenValues M c before)
end
end VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.FPStages
