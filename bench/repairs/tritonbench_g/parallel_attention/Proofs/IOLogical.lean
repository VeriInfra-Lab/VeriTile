import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.QueryGraph

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOLogical
open VeriTile.Triton FP.Structural
noncomputable section

/-- Pure scalar input tables. No state, register, source executor or output
memory is part of the independent numerical recurrence. -/
abbrev InputValues (α : Type) := RegionName → Nat → α

def matrixRead {α : Type} [Inhabited α] (M : Algebra α) (read : InputValues α)
    (R : RegionName) (base rows cols rowStride colStride rowOffset colOffset m n : Nat) :
    TileIndex [m,n] → α := fun i =>
  let row := rowOffset + i.1.val
  let col := colOffset + i.2.1.val
  let address := base + row * rowStride + col * colStride
  M.fp32Load (if row < rows ∧ col < cols then read R address
    else M.literal (some .fp32) .real 0)

def rawMatrixRead {α : Type} [Inhabited α] (M : Algebra α) (read : InputValues α) (R : RegionName)
    (base rows cols rowStride colStride rowOffset colOffset m n : Nat) : TileIndex [m,n] → α := fun i =>
  let row := rowOffset+i.1.val
  let col := colOffset+i.2.1.val
  let address := base+row*rowStride+col*colStride
  if row<rows ∧ col<cols then read R address else M.literal (some .fp32) .real 0

def forwardQuery {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (pid : Nat → Nat) (read : InputValues α) : TileIndex [c.BTL,c.BK] → α := fun i =>
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
    (matrixRead M read c.q (pid 2 * c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
      ((pid 1*c.BTL)) ((pid 0/((c.V+c.BV-1)/c.BV)*c.BK)) c.BTL c.BK i)
    (M.literal (some .fp32) .real c.scale))

def forwardUpdate {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (pid : Nat → Nat) (read : InputValues α) (diagonal : Bool) (start : Nat)
    (previous : ForwardAccumulator α c.BTL c.BV) : ForwardAccumulator α c.BTL c.BV :=
  let key := matrixRead M read c.k (pid 2 * c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
    ((pid 0/((c.V+c.BV-1)/c.BV)*c.BK)) start c.BK c.BTS
  let value := matrixRead M read c.v (pid 2 * c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
    start ((pid 0%((c.V+c.BV-1)/c.BV)*c.BV)) c.BTS c.BV
  let raw := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (forwardQuery M c pid read) key
  let score : TileIndex [c.BTL,c.BTS] → α := fun i =>
    let squared := M.binary (some .fp32) .real .mul (raw i) (raw i)
    if diagonal then
      if start - (pid 1*c.BTL) + i.2.1.val ≤ i.1.val then squared
      else M.literal (some .fp32) .real 0.0
    else squared
  let total := M.reduceSum (some .fp32) (shape := [c.BTL,c.BTS]) ⟨1, by simp⟩ Bool.false score
  let contribution := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (fun i => M.cast (some .fp32) .real .real (score i)) value
  { output := fun i => M.binary (some .fp32) .real .add (previous.output i) (contribution i)
    normalizer := fun i => M.binary (some .fp32) .real .add (previous.normalizer i) (total i) }

def forwardValues {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (pid : Nat → Nat) (read : InputValues α) : ForwardAccumulator α c.BTL c.BV :=
  let zero := M.literal (some .fp32) .real 0
  let initial : ForwardAccumulator α c.BTL c.BV := ⟨fun _ => zero, fun _ => zero⟩
  let previous := foldRange (forwardUpdate M D c pid read Bool.false) 0 ((pid 1*c.BTL)) c.BTS initial
  foldRange (forwardUpdate M D c pid read Bool.true) ((pid 1*c.BTL))
    ((pid 1 + 1) * c.BTL) c.BTS previous

def backwardKey {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig) (read : InputValues α) :=
  matrixRead M read c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
    (c.i_c*c.BTL) (c.i_k*c.BK) c.BTL c.BK

def backwardValue {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig) (read : InputValues α) :=
  matrixRead M read c.v (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
    (c.i_c*c.BTL) (c.i_v*c.BV) c.BTL c.BV

def backwardUpdate {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (read : InputValues α) (diagonal : Bool) (start : Nat)
    (previous : BackwardAccumulator α c.BTL c.BK c.BV) : BackwardAccumulator α c.BTL c.BK c.BV :=
  let query := matrixRead M read c.q (c.i_bh*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
    (c.i_k*c.BK) start c.BK c.BTS
  let outGrad := fun i => M.cast (some .fp32) .real .real
    (rawMatrixRead M read c.do_ (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
      (c.i_v*c.BV) start c.BV c.BTS i)
  let normGrad := fun (i : Fin c.BTS) =>
    let address := c.i_bh*c.T + start + i.val
    M.fp32Load (if start+i.val<c.T then read c.dz address
      else M.literal (some .fp32) .real 0.0)
  let key := if diagonal then backwardKey M c read
    else fun i => M.cast (some .fp32) .real .real (backwardKey M c read i)
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
    (backwardValue M c read) outGrad
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

def backwardValues {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (read : InputValues α) : BackwardAccumulator α c.BTL c.BK c.BV :=
  let initial : BackwardAccumulator α c.BTL c.BK c.BV :=
    ⟨fun _ => M.literal (some .fp32) .real 0, fun _ => M.literal (some .fp32) .real 0⟩
  let before := foldRange (fun j => backwardUpdate M D c read Bool.false (descendingStart c j))
    0 (descendingCount c) 1 initial
  foldRange (backwardUpdate M D c read Bool.true) (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) c.BTS before

def queryInput {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig) (read : InputValues α) :=
  fun i => M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
    (matrixRead M read c.q (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
      (c.i_c*c.BTL) (c.i_k*c.BK) c.BTL c.BK i) (M.literal (some .fp32) .real c.scale))

def queryOutGrad {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig) (read : InputValues α) :=
  fun i => M.cast (some .fp32) .real .real
    (rawMatrixRead M read c.do_ (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
      (c.i_c*c.BTL) (c.i_v*c.BV) c.BTL c.BV i)

def queryNormGrad {α : Type} [Inhabited α] (M : Algebra α) (c : QueryConfig) (read : InputValues α) :=
  fun (i : TileIndex [c.BTL]) => M.fp32Load
    (if c.i_c*c.BTL+i.1.val<c.T then read c.dz (c.i_bh*c.T+c.i_c*c.BTL+i.1.val)
      else M.literal (some .fp32) .real 0.0)

def queryUpdate {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (read : InputValues α) (diagonal : Bool) (start : Nat)
    (previous : TileIndex [c.BTL,c.BK] → α) : TileIndex [c.BTL,c.BK] → α :=
  let key := matrixRead M read c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
    start (c.i_k*c.BK) c.BTS c.BK
  let value := matrixRead M read c.v (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
    (c.i_v*c.BV) start c.BV c.BTS
  let rawDS := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (queryOutGrad M c read) value
  let mask := fun (i : TileIndex [c.BTL,c.BTS]) => start-c.i_c*c.BTL+i.2.1.val ≤ i.1.val
  let ds := fun i =>
    let raw := if c.i_v=0 then M.binary (some .fp32) .real .add (rawDS i)
      (queryNormGrad M c read ⟨i.1,()⟩) else rawDS i
    if diagonal then M.binary (some .fp32) .real .mul
      (if mask i then raw else M.literal (some .fp32) .real 0.0) (M.literal (some .fp32) .real c.scale)
    else raw
  let rawScore := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (queryInput M c read) (matrixTranspose key)
  let score := fun i => if diagonal then
    if mask i then rawScore i else M.literal (some .fp32) .real 0.0 else rawScore i
  let contribution := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (fun i => M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real 2.0) (ds i)) (score i))) key
  fun i => M.binary (some .fp32) .real .add (previous i) (contribution i)

def queryValues {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (read : InputValues α) :=
  let prior := foldRange (queryUpdate M D c read Bool.false) 0 (c.i_c*c.BTL) c.BTS
    (fun _ => M.literal (some .fp32) .real 0)
  foldRange (queryUpdate M D c read Bool.true) (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) c.BTS
    (queryScale M c prior)

theorem forward_projection {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) :
    ParallelAttentionKernel.forwardValues M D c s =
      forwardValues M D c s.pids (fun r off=>(s.mem r off).read .real) := rfl

theorem backward_projection {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) :
    ParallelAttentionKernel.backwardValues M D c s =
      backwardValues M D c (fun r off=>(s.mem r off).read .real) := rfl

theorem query_projection {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : QueryConfig) (s : State α) :
    ParallelAttentionKernel.queryValues M D c s =
      queryValues M D c (fun r off=>(s.mem r off).read .real) := rfl

#specNonCircular forwardValues avoiding [State,FP.Structural.exec,ParallelAttentionKernel.forwardValues]
#specNonCircular backwardValues avoiding [State,FP.Structural.exec,ParallelAttentionKernel.backwardValues]
#specNonCircular queryValues avoiding [State,FP.Structural.exec,ParallelAttentionKernel.queryValues]
#axiomsClean forward_projection
#axiomsClean backward_projection
#axiomsClean query_projection
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOLogical
