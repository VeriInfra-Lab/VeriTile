import bench.repairs.tritonbench_g.layer_norm_liger.IO

/-! Independent scalar equations and ordered logical writeback. The reference
contains no interpreter state or execution. Alias and stride overlap are handled
by last-writer lookup, including reads of earlier DX writes by later rows. -/
namespace VeriTile.Bench.Repairs.LayerNormLiger.Ordered
open VeriTile.Triton FP.Structural LayerIO
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
variable {α : Type} [Inhabited α]

abbrev Writes (α : Type) := RegionName → Nat → Option α
def put (w : Writes α) (r : RegionName) (o : Nat) (x : α) : Writes α :=
  fun q i => if q=r ∧ i=o then some x else w q i
def read (base : RegionName → Nat → α) (w : Writes α) (r : RegionName) (o : Nat) : α :=
  (w r o).getD (base r o)
def scatter (B n : Nat) (r : RegionName) (offset : TileIndex [B] → Nat)
    (value : TileIndex [B] → α) (w : Writes α) : Writes α :=
  (TileShape.allIndices [B]).foldl
    (fun t i => if i.1.val<n then put t r (offset i) (value i) else t) w

def fLoad (M : Algebra α) (p : Forward.Parameters) (v : ChannelValues α (fInputs p))
    (k : Fin 3) (i : TileIndex [p.block]) : α :=
  if i.1.val<p.columns then fInput p v k i.1 else Forward.padding M
def fMean (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (v : ChannelValues α (fInputs p)) : α :=
  M.binary (some .fp32) .real .div
    (M.cast (some .fp32) .fp32 .real
      (R.apply (some .fp32) ⟨0,by simp⟩ Bool.false (fLoad M p v 0) PUnit.unit))
    (Forward.denominator M p)
def fCentered (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (v : ChannelValues α (fInputs p)) (i : TileIndex [p.block]) : α :=
  M.binary (some .fp32) .real .sub
    (M.cast (some .fp32) .fp32 .real (fLoad M p v 0 i)) (fMean M R p v)
def fVariance (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (v : ChannelValues α (fInputs p)) : α :=
  M.binary (some .fp32) .real .div
    (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
      (fun i : TileIndex [p.block] => if i.1.val<p.columns then
        M.binary (some .fp32) .real .mul (fCentered M R p v i) (fCentered M R p v i)
        else M.literal (some .fp32) .real 0) PUnit.unit) (Forward.denominator M p)
def fRstd (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (v : ChannelValues α (fInputs p)) : α :=
  M.unary (some .fp32) .rsqrt (M.binary (some .fp32) .real .add
    (fVariance M R p v) (M.literal (some .fp32) .real p.epsilon))
def fY (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (v : ChannelValues α (fInputs p)) (i : TileIndex [p.block]) : α :=
  M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul (fCentered M R p v i) (fRstd M R p v))
      (M.cast (some .fp32) .fp32 .real (fLoad M p v 1 i)))
    (M.cast (some .fp32) .fp32 .real (fLoad M p v 2 i))
def stored (M : Algebra α) (v : α) := M.cast none .real .fp32 v

def forwardWrites (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) (v : ChannelValues α (fInputs p)) : Writes α :=
  let w := put (put (fun _ _ => none) p.mean (pid*p.meanStride) (stored M (fMean M R p v)))
    p.rstd (pid*p.rstdStride) (stored M (fRstd M R p v))
  scatter p.block p.columns p.y (fun i => pid*p.yStride+i.1.val) (fun i => stored M (fY M R p v i)) w

def forward (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) (v : ChannelValues α (fInputs p)) : ChannelValues α (fOutputs p)
  | ⟨0,_⟩,i => (forwardWrites M R p pid v p.y (pid*p.yStride+i.1.val)).getD default
  | ⟨1,_⟩,_ => (forwardWrites M R p pid v p.mean (pid*p.meanStride)).getD default
  | ⟨2,_⟩,_ => (forwardWrites M R p pid v p.rstd (pid*p.rstdStride)).getD default

abbrev BSlot (p : Backward.Parameters) := (k : Fin 5) × TileIndex ((bInputs p)[k]).shape

def bScalar (p : Backward.Parameters) (v : ChannelValues α (bInputs p)) :
    (k : Fin 5) → TileIndex ((bInputs p)[k]).shape → α
  | ⟨0,_⟩,i => v (0 : Fin 5) i
  | ⟨1,_⟩,i => v (1 : Fin 5) i
  | ⟨2,_⟩,i => v (2 : Fin 5) i
  | ⟨3,_⟩,i => v (3 : Fin 5) i
  | ⟨4,_⟩,i => v (4 : Fin 5) i

def bAt (p : Backward.Parameters) (pid : Nat) (v : ChannelValues α (bInputs p))
    (r : RegionName) (o : Nat) (slot : BSlot p) : Prop :=
  (bMemory α p pid).inputMask slot.1 v slot.2 ∧
    r=(bInputs p)[slot.1].buf ∧ o=(bMemory α p pid).read slot.1 v slot.2

/-- Simultaneously bound aliases have the same logical value. The choice of a
matching channel therefore has no effect on admissible input bindings. -/
def bInitial (p : Backward.Parameters) (pid : Nat) (v : ChannelValues α (bInputs p))
    (r : RegionName) (o : Nat) : α := by
  classical
  exact if h : ∃ slot,bAt p pid v r o slot then
    bScalar p v (Classical.choose h).1 (Classical.choose h).2 else default

structure Carries (α : Type) (p : Backward.Parameters) where
  writes : Writes α
  dw : TileIndex [p.block] → α
  db : TileIndex [p.block] → α

def bLoad (M : Algebra α) (p : Backward.Parameters) (base : RegionName → Nat → α)
    (a : Carries α p) (r : RegionName) (offset : Nat) (i : TileIndex [p.block]) : α :=
  if i.1.val<p.columns then read base a.writes r (offset+i.1.val) else Backward.padding M
def bMean (p : Backward.Parameters) (base : RegionName → Nat → α) (a : Carries α p) (row : Nat) : α :=
  read base a.writes p.mean row
def bRstd (p : Backward.Parameters) (base : RegionName → Nat → α) (a : Carries α p) (row : Nat) : α :=
  read base a.writes p.rstd row
def bXhat (M : Algebra α) (p : Backward.Parameters) (base : RegionName → Nat → α)
    (a : Carries α p) (row : Nat) (i : TileIndex [p.block]) : α :=
  if i.1.val<p.columns then Backward.cast M (M.binary (some .fp32) .fp32 .mul
    (M.binary (some .fp32) .fp32 .sub (bLoad M p base a p.x (row*p.xStride) i)
      (bMean p base a row)) (bRstd p base a row)) else M.literal none .real 0

def bWdy (M : Algebra α) (p : Backward.Parameters) (base : RegionName → Nat → α)
    (a : Carries α p) (row : Nat) (i : TileIndex [p.block]) : α :=
  M.binary none .fp32 .mul (bLoad M p base a p.weight 0 i) (bLoad M p base a p.dy (row*p.dyStride) i)
def bC1 (M : Algebra α) (p : Backward.Parameters) (base : RegionName → Nat → α)
    (a : Carries α p) (row : Nat) : α :=
  Backward.binary M .div (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
    (fun i => Backward.binary M .mul (bXhat M p base a row i)
      (Backward.cast M (bWdy M p base a row i))) PUnit.unit) (Backward.denom M p)
def bC2 (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters) (base : RegionName → Nat → α)
    (a : Carries α p) (row : Nat) : α :=
  Backward.binary M .div (Backward.cast M (R.apply (some .fp32) ⟨0,by simp⟩ Bool.false
    (bWdy M p base a row) PUnit.unit)) (Backward.denom M p)
def bDX (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters) (base : RegionName → Nat → α)
    (a : Carries α p) (row : Nat) (i : TileIndex [p.block]) : α :=
  Backward.binary M .mul (Backward.binary M .sub (Backward.cast M (bWdy M p base a row i))
    (Backward.binary M .add (Backward.binary M .mul (bXhat M p base a row i) (bC1 M p base a row))
      (bC2 M R p base a row))) (Backward.cast M (bRstd p base a row))

def bStep (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters) (base : RegionName → Nat → α)
    (row : Nat) (a : Carries α p) : Carries α p where
  writes := scatter p.block p.columns p.dx (fun i => row*p.dxStride+i.1.val)
    (fun i => stored M (bDX M R p base a row i)) a.writes
  dw := fun i => Backward.binary M .add (a.dw i)
    (Backward.binary M .mul (Backward.cast M (bLoad M p base a p.dy (row*p.dyStride) i))
      (bXhat M p base a row i))
  db := fun i => Backward.binary M .add (a.db i)
    (Backward.cast M (bLoad M p base a p.dy (row*p.dyStride) i))

def bZero (M : Algebra α) (p : Backward.Parameters) : Carries α p :=
  ⟨fun _ _ => none,fun _ => Backward.zero M,fun _ => Backward.zero M⟩
def bAccumulated (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) : Carries α p :=
  foldRange (bStep M R p (bInitial p pid v)) (first p pid) (last p pid) 1 (bZero M p)
def bFinish (M : Algebra α) (p : Backward.Parameters) (pid : Nat) (a : Carries α p) : Writes α :=
  let w := scatter p.block p.columns p.dw (fun i => pid*p.dwStride+i.1.val)
    (fun i => stored M (a.dw i)) a.writes
  scatter p.block p.columns p.db (fun i => pid*p.dbStride+i.1.val) (fun i => stored M (a.db i)) w

def backwardWrites (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) : Writes α :=
  bFinish M p pid (bAccumulated M R p pid v)
def backward (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) : ChannelValues α (bOutputs p)
  | ⟨0,_⟩,i => (backwardWrites M R p pid v p.dx (i.1.val*p.dxStride+i.2.1.val)).getD default
  | ⟨1,_⟩,i => (backwardWrites M R p pid v p.dw (pid*p.dwStride+i.1.val)).getD default
  | ⟨2,_⟩,i => (backwardWrites M R p pid v p.db (pid*p.dbStride+i.1.val)).getD default

end
end VeriTile.Bench.Repairs.LayerNormLiger.Ordered
