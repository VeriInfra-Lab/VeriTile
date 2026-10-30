import bench.tritonbench_g.parallel_attention.FPForwardGraph
namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
noncomputable section

/-- Parameters of the accepted DK/DV source stage, with its caller-computed
program indices explicit. The complete descending and diagonal loops remain. -/
structure BackwardConfig extends ForwardConfig where
  do_ : RegionName
  dz : RegionName
  dk : RegionName
  dv : RegionName
  i_bh : Nat
  i_c : Nat
  i_k : Nat
  i_v : Nat
  i_h : Nat

def backwardKernel (c : BackwardConfig) : ComputeKernel :=
  backwardSource c.q c.k c.v c.do_ c.dz c.dk c.dv c.i_bh c.i_c c.i_k c.i_v c.i_h
    c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.scale
    c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV

def backwardKey {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig) (s : State α) :=
  matrixRead M s c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
    (c.i_c*c.BTL) (c.i_k*c.BK) c.BTL c.BK

def backwardValue {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig) (s : State α) :=
  matrixRead M s c.v (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
    (c.i_c*c.BTL) (c.i_v*c.BV) c.BTL c.BV

structure BackwardAccumulator (α : Type) (BTL BK BV : Nat) where
  key : TileIndex [BTL,BK] → α
  value : TileIndex [BTL,BV] → α

/-- A value-level transpose. No floating operation or change of precision. -/
def matrixTranspose {α : Type} {m n : Nat} (a : TileIndex [m,n] → α) : TileIndex [n,m] → α :=
  fun i => a ⟨i.2.1,i.1,()⟩

/-- Raw storage payload for an immediately cast load; the cast itself is
retained separately in the source graph. -/
def rawMatrixRead {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (R : RegionName)
    (base rows cols rowStride colStride rowOffset colOffset m n : Nat) : TileIndex [m,n] → α := fun i =>
  let row := rowOffset+i.1.val
  let col := colOffset+i.2.1.val
  let address := base+row*rowStride+col*colStride
  if row<rows ∧ col<cols then (s.mem R address).read .real else M.literal (some .fp32) .real 0

/-- The exact primitive order differs between the two source loops: the first
scales the dot and DZ contribution separately, while the diagonal loop masks
their sum and then scales it. No reassociation is assumed. -/
def backwardUpdate {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) (diagonal : Bool) (start : Nat)
    (previous : BackwardAccumulator α c.BTL c.BK c.BV) : BackwardAccumulator α c.BTL c.BK c.BV :=
  let query := matrixRead M s c.q (c.i_bh*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
    (c.i_k*c.BK) start c.BK c.BTS
  let outGrad := fun i => M.cast (some .fp32) .real .real
    (rawMatrixRead M s c.do_ (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
      (c.i_v*c.BV) start c.BV c.BTS i)
  let normGrad := fun (i : Fin c.BTS) =>
    let address := c.i_bh*c.T + start + i.val
    M.fp32Load (if start+i.val<c.T then (s.mem c.dz address).read .real
      else M.literal (some .fp32) .real 0.0)
  let key := if diagonal then backwardKey M c s
    else fun i => M.cast (some .fp32) .real .real (backwardKey M c s i)
  let scoreRaw := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := []) key query
  let scaled : TileIndex [c.BTL,c.BTS] → α := fun i =>
    M.binary (some .fp32) .real .mul (scoreRaw i) (M.literal (some .fp32) .real c.scale)
  let valid := fun (i : TileIndex [c.BTL,c.BTS]) => start+i.2.1.val<c.T
  let mask := fun (i : TileIndex [c.BTL,c.BTS]) => i.1.val ≤ start-c.i_c*c.BTL+i.2.1.val ∧ valid i
  let score := fun i => if diagonal then
      if mask i then scaled i else M.literal (some .fp32) .real 0.0
    else if valid i then scaled i else M.literal (some .fp32) .real 0.0
  let square := fun i =>
    let v := M.binary (some .fp32) .real .mul (scaled i) (scaled i)
    if diagonal then if mask i then v else M.literal (some .fp32) .real 0.0
    else M.binary (some .fp32) .real .mul (score i) (score i)
  let dsRaw := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (backwardValue M c s) outGrad
  let ds := fun i =>
    if diagonal then
      let raw := if c.i_v=0 then M.binary (some .fp32) .real .add (dsRaw i) (normGrad i.2.1) else dsRaw i
      let masked := if mask i then raw else M.literal (some .fp32) .real 0.0
      M.binary (some .fp32) .real .mul masked (M.literal (some .fp32) .real c.scale)
    else
      let scaled := M.binary (some .fp32) .real .mul (dsRaw i) (M.literal (some .fp32) .real c.scale)
      let raw := if c.i_v=0 then M.binary (some .fp32) .real .add scaled
        (M.binary (some .fp32) .real .mul (normGrad i.2.1) (M.literal (some .fp32) .real c.scale)) else scaled
      if valid i then raw else M.literal (some .fp32) .real 0.0
  let dv := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (fun i => M.cast (some .fp32) .real .real (square i)) (matrixTranspose outGrad)
  let dk := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (fun i => M.cast (some .fp32) .real .real
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real 2.0) (ds i)) (score i)))
    (matrixTranspose query)
  { key := fun i => M.binary (some .fp32) .real .add (previous.key i) (dk i)
    value := fun i => M.binary (some .fp32) .real .add (previous.value i) (dv i) }

def descendingCount (c : BackwardConfig) : Nat :=
  (((c.T+c.BTS-1)/c.BTS*c.BTS - (c.i_c+1)*c.BTL)+c.BTS-1)/c.BTS

def descendingStart (c : BackwardConfig) (j : Nat) : Nat :=
  (c.T+c.BTS-1)/c.BTS*c.BTS-c.BTS-j*c.BTS

def backwardValues {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) : BackwardAccumulator α c.BTL c.BK c.BV :=
  let initial : BackwardAccumulator α c.BTL c.BK c.BV :=
    ⟨fun _ => M.literal (some .fp32) .real 0, fun _ => M.literal (some .fp32) .real 0⟩
  let before := foldRange (fun j => backwardUpdate M D c s Bool.false (descendingStart c j))
    0 (descendingCount c) 1 initial
  foldRange (backwardUpdate M D c s Bool.true) (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) c.BTS before

def backwardOutputMemory {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig)
    (s : State α) (values : BackwardAccumulator α c.BTL c.BK c.BV) : State α :=
  let afterKey := (TileShape.allIndices [c.BTL,c.BK]).foldl (fun t i =>
    let row := c.i_c*c.BTL+i.1.val
    let col := c.i_k*c.BK+i.2.1.val
    if row<c.T ∧ col<c.K then t.write c.dk
      ((c.i_bh+c.B*c.H*c.i_v)*c.s_qk_h + row*c.s_qk_t + col*c.s_qk_d)
      (.mk .real (M.cast (some .fp32) .real .real (values.key i))) else t) s
  (TileShape.allIndices [c.BTL,c.BV]).foldl (fun t i =>
    let row := c.i_c*c.BTL+i.1.val
    let col := c.i_v*c.BV+i.2.1.val
    if row<c.T ∧ col<c.V then t.write c.dv
      ((c.i_bh+c.B*c.H*c.i_k)*c.s_vo_h + row*c.s_vo_t + col*c.s_vo_d)
      (.mk .real (M.cast (some .fp32) .real .real (values.value i))) else t) afterKey

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
