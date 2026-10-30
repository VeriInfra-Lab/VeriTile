import bench.tritonbench_g.parallel_attention.FPBackwardGraph
import bench.tritonbench_g.parallel_attention.QuerySyntax

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
noncomputable section

structure QueryConfig extends BackwardConfig where
  dq : RegionName

def queryKernel (c : QueryConfig) : ComputeKernel :=
  querySource c.q c.k c.v c.do_ c.dz c.dq c.i_bh c.i_c c.i_k c.i_v
    c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.scale
    c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV

def queryInput {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig) (s : State α) :=
  fun i => M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
    (matrixRead M s c.q (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
      (c.i_c*c.BTL) (c.i_k*c.BK) c.BTL c.BK i) (M.literal (some .fp32) .real c.scale))

def queryOutGrad {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig) (s : State α) :=
  fun i => M.cast (some .fp32) .real .real
    (rawMatrixRead M s c.do_ (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
      (c.i_c*c.BTL) (c.i_v*c.BV) c.BTL c.BV i)

def queryNormGrad {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig) (s : State α) :=
  fun (i : TileIndex [c.BTL]) => M.fp32Load
    (if c.i_c*c.BTL+i.1.val<c.T then (s.mem c.dz (c.i_bh*c.T+c.i_c*c.BTL+i.1.val)).read .real
      else M.literal (some .fp32) .real 0.0)

/-- Independent tile recurrence: no source registers or execution results. -/
def queryUpdate {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s : State α) (diagonal : Bool) (start : Nat)
    (previous : TileIndex [c.BTL,c.BK] → α) : TileIndex [c.BTL,c.BK] → α :=
  let key := matrixRead M s c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
    start (c.i_k*c.BK) c.BTS c.BK
  let value := matrixRead M s c.v (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
    (c.i_v*c.BV) start c.BV c.BTS
  let rawDS := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (queryOutGrad M c s) value
  let mask := fun (i : TileIndex [c.BTL,c.BTS]) => start-c.i_c*c.BTL+i.2.1.val ≤ i.1.val
  let ds := fun i =>
    let raw := if c.i_v=0 then M.binary (some .fp32) .real .add (rawDS i)
      (queryNormGrad M c s ⟨i.1,()⟩) else rawDS i
    if diagonal then M.binary (some .fp32) .real .mul
      (if mask i then raw else M.literal (some .fp32) .real 0.0) (M.literal (some .fp32) .real c.scale)
    else raw
  let rawScore := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (queryInput M c s) (matrixTranspose key)
  let score := fun i => if diagonal then
    if mask i then rawScore i else M.literal (some .fp32) .real 0.0 else rawScore i
  let contribution := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (fun i => M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real 2.0) (ds i)) (score i))) key
  fun i => M.binary (some .fp32) .real .add (previous i) (contribution i)

def queryScale {α : Type} (M : Algebra α) (c : QueryConfig)
    (a : TileIndex [c.BTL,c.BK] → α) :=
  fun i => M.binary (some .fp32) .real .mul (a i) (M.literal (some .fp32) .real c.scale)

def queryValues {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s : State α) :=
  let prior := foldRange (queryUpdate M D c s Bool.false) 0 (c.i_c*c.BTL) c.BTS
    (fun _ => M.literal (some .fp32) .real 0)
  foldRange (queryUpdate M D c s Bool.true) (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) c.BTS
    (queryScale M c prior)

def queryOutputMemory {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig)
    (s : State α) (values : TileIndex [c.BTL,c.BK] → α) : State α :=
  (TileShape.allIndices [c.BTL,c.BK]).foldl (fun t i =>
    let row := c.i_c*c.BTL+i.1.val
    let col := c.i_k*c.BK+i.2.1.val
    if row<c.T ∧ col<c.K then t.write c.dq
      ((c.i_bh+c.B*c.H*c.i_v)*c.s_qk_h + row*c.s_qk_t + col*c.s_qk_d)
      (.mk .real (M.cast (some .fp32) .real .real (values i))) else t) s

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
