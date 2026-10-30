/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelQuerySource

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig DotPrimitive withDot foldRange)
open FPStages (logDecay rowDecay blockDecay)
noncomputable section

def outputGradient {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s : State α) : TileIndex [c.BTL,c.BV] → α :=
  matrixRead M s c.do_ (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
    (c.i_c*c.BTL) (c.i_v*c.BV) c.BTL c.BV

/-- Independent DQ recurrence from initial memory. The first loop's decay
and the diagonal loop's left-associated score/decay/scale products deliberately
remain different. No distributivity, FMA contraction, or exp identity is used. -/
def update {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) (diagonal : Bool) (start : Nat)
    (previous : TileIndex [c.BTL,c.BK] → α) : TileIndex [c.BTL,c.BK] → α :=
  let b := logDecay M c.i_h
  let key := matrixRead M s c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
    start (c.i_k*c.BK) c.BTS c.BK
  let value := matrixRead M s c.v (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
    (c.i_v*c.BV) start c.BV c.BTS
  let raw := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (outputGradient M c s) value
  let decay := fun (i : TileIndex [c.BTL,c.BTS]) =>
    if diagonal then
      if start-c.i_c*c.BTL+i.2.1.val ≤ i.1.val then
        M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul
          (M.fromInt (some .fp32) ((i.1.val : Int) - (start-c.i_c*c.BTL+i.2.1.val : Nat))) b)
      else M.literal (some .fp32) .real 0.0
    else rowDecay M b (c.BTS-i.2.1.val)
  let score := fun i =>
    let weighted := M.binary (some .fp32) .real .mul (raw i) (decay i)
    if diagonal then M.binary (some .fp32) .real .mul weighted (M.literal (some .fp32) .real c.scale)
    else weighted
  let contribution := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (fun i => M.cast (some .fp32) .real .real (score i)) key
  fun i => M.binary (some .fp32) .real .add
    (if diagonal then previous i else M.binary (some .fp32) .real .mul (previous i) (blockDecay M b c.BTS))
    (contribution i)

def betweenValues {α : Type} (M : Algebra α) (c : BackwardConfig)
    (previous : TileIndex [c.BTL,c.BK] → α) : TileIndex [c.BTL,c.BK] → α :=
  fun i => M.binary (some .fp32) .real .mul (previous i)
    (M.binary (some .fp32) .real .mul (rowDecay M (logDecay M c.i_h) i.1.val)
      (M.literal (some .fp32) .real c.scale))

def values {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) : TileIndex [c.BTL,c.BK] → α :=
  let initial := fun _ => M.literal (some .fp32) .real 0
  let before := foldRange (update M D c s Bool.false) 0 (c.i_c*c.BTL) c.BTS initial
  foldRange (update M D c s Bool.true) (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) c.BTS
    (betweenValues M c before)

/-- Exact masked store fold, including the frame and overlapping addresses.
Values depend on initial memory; the register interpreter is not a reference. -/
def outputMemory {α : Type} (M : Algebra α) (c : BackwardConfig) (dq : RegionName)
    (s : State α) (result : TileIndex [c.BTL,c.BK] → α) : State α :=
  (TileShape.allIndices [c.BTL,c.BK]).foldl (fun t i =>
    let row := c.i_c*c.BTL+i.1.val
    let col := c.i_k*c.BK+i.2.1.val
    if row<c.T ∧ col<c.K then t.write dq
      ((c.i_bh+c.B*c.H*c.i_v)*c.s_qk_h + row*c.s_qk_t + col*c.s_qk_d)
      (.mk .real (M.cast (some .fp32) .real .real (result i))) else t) s

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
