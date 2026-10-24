import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelNativeTerms
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Grid

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (ForwardConfig BackwardConfig queryChunk keyChunk valueChunk)
open Native
noncomputable section
set_option maxHeartbeats 3000000

def config (s : Shape) (L : Nat) : ForwardConfig where
  q := "q"
  k := "k"
  v := "v"
  o := "o"
  z := "unused"
  s_qk_h := s.T*s.DK
  s_qk_t := s.DK
  s_qk_d := 1
  s_vo_h := s.T*s.DV
  s_vo_t := s.DV
  s_vo_d := 1
  scale := (Real.sqrt (s.DK : ℝ))⁻¹
  B := s.B
  H := s.H
  T := s.T
  K := s.DK
  V := s.DV
  BTL := L
  BTS := 32
  BK := s.BK
  BV := s.BV

def backwardConfig (s : Shape) (tile : Tile) : BackwardConfig where
  toForwardConfig := config s 64
  do_ := "g"
  dz := "unused"
  dk := "dk"
  dv := "dv"
  i_bh := tile.bh
  i_c := tile.chunk
  i_k := tile.key
  i_v := tile.value
  i_h := tile.bh%s.H

def regionBase (b : Bases) (r : RegionName) : Nat :=
  if r="q" then b.q else if r="k" then b.k else if r="v" then b.v
  else if r="g" then b.g else 0

/-- The source reads the current physical word memory. The actual source
program indices are decoded by the readable DSL, not supplied as final values. -/
def initial (s : Shape) (b : Bases) (m : Memory α) (tile : Tile) : State (Expr α) where
  mem := fun r off => .mk .real (.input (m (regionBase b r+off)))
  regs := fun _ _ _ => none
  pids := fun index => match index with
    | 0 => ProgramDomain.gridX s tile
    | 1 => tile.chunk
    | 2 => tile.bh
    | _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

theorem source_chunks (s : Shape) (L : Nat) (b : Bases) (m : Memory α) (tile : Tile)
    (valid : ProgramDomain s L tile) :
    queryChunk (config s L) (initial s b m tile)=tile.chunk*L ∧
    keyChunk (config s L) (initial s b m tile)=tile.key*s.BK ∧
    valueChunk (config s L) (initial s b m tile)=tile.value*s.BV := by
  have decode := grid_decode s L tile valid
  change tile.chunk*L=tile.chunk*L ∧
    ProgramDomain.gridX s tile/blocks s.DV s.BV*s.BK=tile.key*s.BK ∧
    ProgramDomain.gridX s tile%blocks s.DV s.BV*s.BV=tile.value*s.BV
  exact ⟨rfl,congrArg (·*s.BK) decode.2.1,congrArg (·*s.BV) decode.2.2⟩

def dotRows (stage : Nat) : Nat := if stage=0 then 128 else 64
def dotCols (s : Shape) (stage site : Nat) : Nat :=
  if stage=0 then (if site=0 then 32 else s.BV)
  else if stage=1 then (if site=0 then 32 else s.BK)
  else if site<2 then 32 else if site=2 then s.BK else s.BV

/-- Bind the independent scalar graph's existing source-site indices to the
same finite matrix instruction trees. Only dimensions and lane numbers enter. -/
def plans (s : Shape) (p : DotTrees) : Plans where
  tree := fun site k => p.tree (dotRows (site[0]?.getD 0)) k
    (dotCols s (site[0]?.getD 0) (site[1]?.getD 0)) (site[6]?.getD 0) (site[8]?.getD 0)
  complete := fun site k => p.complete _ k _ _ _

def realScalar : Scalar ℝ where
  literal := id
  fromNat := Nat.cast
  fromInt := Int.cast
  binary := fun op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y
    | .max => max x y | .pow => x^y
  unary := fun op x => match op with
    | .exp2 => Real.exp (x*Real.log 2)
    | .log2 => Real.log x/Real.log 2
    | _ => 0
  fma := fun x y z => x*y+z

theorem realScalar_ops : realScalar.ops=realOps := rfl

theorem eval_matrixRead (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (R : RegionName)
    (base rows cols rowStride colStride rowOffset colOffset h w : Nat) (i : TileIndex [h,w]) :
    evaluate a p (matrixRead algebra (initial s b m tile) R base rows cols
      rowStride colStride rowOffset colOffset h w i) =
    if rowOffset+i.1.val<rows ∧ colOffset+i.2.1.val<cols then
      m (regionBase b R+(base+(rowOffset+i.1.val)*rowStride+(colOffset+i.2.1.val)*colStride))
    else a.literal 0 := by
  simp only [matrixRead,algebra,initial,Cell.read_same]
  split <;> rfl

theorem eval_logDecay (a : Scalar α) (p : DotTrees) (head : Nat) :
    evaluate a p (FPStages.logDecay (algebra : Algebra (Expr α)) head)=logDecay a.ops head := by
  norm_num [FPStages.logDecay,logDecay,algebra,consume,evaluate,Scalar.ops]

theorem eval_rowDecay (a : Scalar α) (p : DotTrees) (value : Expr α) (n : Nat) :
    evaluate a p (FPStages.rowDecay algebra value n)=decayNat a.ops (evaluate a p value) n := rfl

theorem eval_blockDecay (a : Scalar α) (p : DotTrees) (value : Expr α) (n : Nat) :
    evaluate a p (FPStages.blockDecay algebra value n)=blockDecay a.ops (evaluate a p value) n := rfl

theorem plan_eval_congr (a : Ops α) (tree : DotPlan) (x y u v : Nat → α) (carry : α)
    (hx : ∀ i,i∈tree.leaves → x i=u i) (hy : ∀ i,i∈tree.leaves → y i=v i) :
    tree.eval a x y carry=tree.eval a u v carry := by
  induction tree with
  | zero => rfl
  | acc => rfl
  | product i => simp only [DotPlan.eval,hx i (by simp [DotPlan.leaves]),hy i (by simp [DotPlan.leaves])]
  | add l r hl hr =>
    simp only [DotPlan.eval]
    rw [hl (fun i hi=>hx i (by simp [DotPlan.leaves,hi])) (fun i hi=>hy i (by simp [DotPlan.leaves,hi])),
      hr (fun i hi=>hx i (by simp [DotPlan.leaves,hi])) (fun i hi=>hy i (by simp [DotPlan.leaves,hi]))]
  | fma i rest ih =>
    simp only [DotPlan.eval]
    rw [hx i (by simp [DotPlan.leaves]),hy i (by simp [DotPlan.leaves])]
    rw [ih (fun j hj=>hx j (by simp [DotPlan.leaves,hj])) (fun j hj=>hy j (by simp [DotPlan.leaves,hj]))]

theorem plan_eval_congr_bounded (a : Ops α) (tree : DotPlan) (k : Nat) (hc : tree.Covers k)
    (x y u v : Nat → α) (carry : α)
    (hx : ∀ i,i<k → x i=u i) (hy : ∀ i,i<k → y i=v i) :
    tree.eval a x y carry=tree.eval a u v carry :=
  plan_eval_congr a tree x y u v carry
    (fun i hi=>hx i (List.mem_range.mp (hc.1.mem_iff.mp hi)))
    (fun i hi=>hy i (List.mem_range.mp (hc.1.mem_iff.mp hi)))

theorem dotEval_2d (a : Scalar α) (p : DotTrees) (m k n : Nat)
    (x : TileIndex [m,k] → α) (y : TileIndex [k,n] → α) (u v : Nat → α)
    (carry : α) (row : Fin m) (col : Fin n)
    (hx : ∀ e : Fin k,x (row,e,PUnit.unit)=u e.val)
    (hy : ∀ e : Fin k,y (e,col,PUnit.unit)=v e.val) :
    dotEval a p [] m k n x y carry (row,col,PUnit.unit)=
      (p.tree m k n row.val col.val).eval a.ops u v carry := by
  unfold dotEval
  dsimp only [matrixLane]
  apply plan_eval_congr_bounded a.ops _ k (p.complete _ _ _ _ _)
  · intro e he
    simpa [he,dotInputs] using hx ⟨e,he⟩
  · intro e he
    simpa [he,dotInputs] using hy ⟨e,he⟩

theorem load_tensorRead (a : Ops α) (m : Memory α) (base heads T D bh t e : Nat)
    (hb : bh<heads) :
    load a T D (tensorRead a m base heads T D) bh t e =
      if t<T ∧ e<D then m (base+dense T D bh t e) else a.zero := by
  unfold load tensorRead
  split_ifs <;> simp_all

theorem eval_dense_read (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (R : RegionName) (heads T D bh t e h w : Nat)
    (hb : bh<heads) (i : TileIndex [h,w]) :
    evaluate a p (matrixRead algebra (initial s b m tile) R (bh*(T*D)) T D
      D 1 t e h w i) =
    load a.ops T D (tensorRead a.ops m (regionBase b R) heads T D)
      bh (t+i.1.val) (e+i.2.1.val) := by
  rw [eval_matrixRead,load_tensorRead _ _ _ _ _ _ _ _ _ hb]
  split_ifs
  · congr 1
    unfold dense
    ring
  · rfl

theorem eval_transposed_read (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (R : RegionName) (heads T D bh t e h w : Nat)
    (hb : bh<heads) (i : TileIndex [h,w]) :
    evaluate a p (matrixRead algebra (initial s b m tile) R (bh*(T*D)) D T
      1 D e t h w i) =
    load a.ops T D (tensorRead a.ops m (regionBase b R) heads T D)
      bh (t+i.2.1.val) (e+i.1.val) := by
  rw [eval_matrixRead,load_tensorRead _ _ _ _ _ _ _ _ _ hb]
  simp only [and_comm]
  split_ifs
  · congr 1
    unfold dense
    ring
  · rfl

#axiomsClean source_chunks
#axiomsClean eval_matrixRead
#axiomsClean eval_logDecay
#axiomsClean dotEval_2d
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
