import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPConfig
namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def sequenceLength {α : Type} [Inhabited α] (c : Config) (s : State α) : Nat :=
  (s.mem (Region.cast c.seq) (s.pids 1 / c.H)).read .nat

def rowStart {α : Type} (c : Config) (s : State α) : Nat := s.pids 0*c.BM

def metadataIndex {α : Type} (c : Config) (s : State α) : Nat := s.pids 1*c.NR+s.pids 0

def blockCount {α : Type} [Inhabited α] (c : Config) (s : State α) : Nat :=
  (s.mem (Region.cast c.blocks) (metadataIndex c s)).read .nat

def columnCount {α : Type} [Inhabited α] (c : Config) (s : State α) : Nat :=
  (s.mem (Region.cast c.counts) (metadataIndex c s)).read .nat

def queryBase {α : Type} (c : Config) (s : State α) : Nat :=
  (s.pids 1 / c.H)*c.sqz+(s.pids 1%c.H)*c.sqh

def keyValueBase {α : Type} (c : Config) (s : State α) : Nat :=
  (s.pids 1 / c.H)*c.skz+(s.pids 1%c.H)*c.skh

def queryValues {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (s : State α) :
    TileIndex [c.BM,c.BD] → α := fun i =>
  M.cast none .real .fp16 (M.binary (some .fp32) .real .mul
    (M.cast (some .fp32) .fp16 .real
      ((s.mem c.q (queryBase c s+(rowStart c s+i.1.val)*c.sqm+i.2.1.val*c.sqk)).read .fp16))
    (M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real c.scale)
      (M.literal (some .fp32) .real 1.44269504)))

/-- Mask-only block-offset loads preserve the state's undefined value. The
column-index load uses its actual scalar cond mask, not the lane n_mask. -/
def columnValues {α : Type} [Inhabited α] (c : Config) (s : State α)
    (columnPhase : Bool) (start : Nat) : Fin c.BN → Nat := fun j =>
  if columnPhase then
    if start < columnCount c s then
      (s.mem (Region.cast c.columns) (metadataIndex c s*c.NV+start+j.val)).read .nat else 0
  else
    (if start < blockCount c s then
      (s.mem (Region.cast c.offsets) (metadataIndex c s*c.NS+start)).read .nat
      else s.undef .nat (Region.cast c.offsets) (metadataIndex c s*c.NS+start)) + j.val

def keyActive {α : Type} [Inhabited α] (c : Config) (s : State α)
    (columnPhase : Bool) (start : Nat) (j : Fin c.BN) : Prop :=
  if columnPhase then start+j.val < columnCount c s ∧ start < columnCount c s
  else columnValues c s Bool.false start j < sequenceLength c s ∧ start < blockCount c s

instance {α : Type} [Inhabited α] (c : Config) (s : State α) (ph : Bool) (n : Nat) (j : Fin c.BN) :
    Decidable (keyActive c s ph n j) := by unfold keyActive; infer_instance

def keyValues {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (s : State α)
    (columnPhase : Bool) (start : Nat) : TileIndex [c.BD,c.BN] → α := fun i =>
  if keyActive c s columnPhase start i.2.1 then
    (s.mem c.k (keyValueBase c s+i.1.val*c.skk+columnValues c s columnPhase start i.2.1*c.skn)).read .fp16
  else M.cast none .real .fp16 (M.literal none .real 0.0)

def valueValues {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (s : State α)
    (columnPhase : Bool) (start : Nat) : TileIndex [c.BN,c.BD] → α := fun i =>
  if keyActive c s columnPhase start i.1 then
    (s.mem c.v (keyValueBase c s+i.2.1.val*c.svk+columnValues c s columnPhase start i.1*c.svn)).read .fp16
  else M.cast none .real .fp16 (M.literal none .real 0.0)

structure Accumulator (α : Type) (BM BD : Nat) where
  maximum : TileIndex [BM] → α
  normalizer : TileIndex [BM] → α
  output : TileIndex [BM,BD] → α

def initialAccumulator {α : Type} (M : Algebra α) (c : Config) : Accumulator α c.BM c.BD :=
  ⟨fun _ => M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0) M.negInf,
    fun _ => M.literal (some .fp32) .real 0, fun _ => M.literal (some .fp32) .real 0⟩

/-- Independent online-softmax operation graph. Inactive fixed loop steps
still run: no simplification of infinity-infinity, l_i*0 or masked dot is
assumed. The source order of all casts, products and additions is retained. -/
def update {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (c : Config) (s : State α)
    (columnPhase : Bool) (start : Nat) (old : Accumulator α c.BM c.BD) : Accumulator α c.BM c.BD :=
  let dotQK := D.apply (some .fp32) {} (batch := []) (queryValues M c s)
    (keyValues M c s columnPhase start)
  let scores : TileIndex [c.BM,c.BN] → α := fun i =>
    let active := rowStart c s+i.1.val < sequenceLength c s ∧
      (if columnPhase then keyActive c s Bool.true start i.2.1
       else columnValues c s Bool.false start i.2.1 ≤ rowStart c s+i.1.val)
    M.binary (some .fp32) .real .add
      (if active then M.literal (some .fp32) .real 0 else M.negInf)
      (M.cast (some .fp32) .fp32 .real (dotQK i))
  let rowmax := M.reduceMax (some .fp32) (shape := [c.BM,c.BN]) ⟨1,by simp⟩ Bool.false scores
  let newmax : TileIndex [c.BM] → α := fun i => M.binary (some .fp32) .real .max (old.maximum i) (rowmax i)
  let alpha : TileIndex [c.BM] → α := fun i =>
    M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .sub (old.maximum i) (newmax i))
  let weights : TileIndex [c.BM,c.BN] → α := fun i =>
    M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .sub (scores i) (newmax (i.1,())))
  let contribution := D.apply (some .fp32) {} (batch := [])
    (fun i => M.cast (some .fp32) .real .fp16 (weights i)) (valueValues M c s columnPhase start)
  let rowsum := M.reduceSum (some .fp32) (shape := [c.BM,c.BN]) ⟨1,by simp⟩ Bool.false weights
  { maximum := newmax
    normalizer := fun i => M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (old.normalizer i) (alpha i)) (rowsum i)
    output := fun i => M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (old.output i)
        (M.binary (some .fp32) .real .add
          (M.binary (some .fp32) .real .mul (old.normalizer (i.1,())) (M.literal (some .fp32) .real 0))
          (alpha (i.1,()))))
      (M.cast (some .fp32) .fp32 .real (contribution i)) }

def finalValues {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (c : Config) (s : State α) :=
  foldRange (update M D c s Bool.true) 0 16 c.BN
    (foldRange (update M D c s Bool.false) 0 8 1 (initialAccumulator M c))

end
end VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
