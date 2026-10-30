import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.SourceSyntax

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
noncomputable section

/-- A complete primitive binding, retaining precision, matrix layout and dot
options. Totality describes primitive availability; it assumes no numerical
identity or final output. Native hardware correspondence remains separate. -/
structure DotPrimitive (α : Type) where
  apply : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    Values α .real (batch ++ [m,k]) → Values α .real (batch ++ [k,n]) →
    Values α .real (batch ++ [m,n])

def withDot {α : Type} (M : Algebra α) (D : DotPrimitive α) : Algebra α :=
  { M with dotWithOptions := fun p o {_} {_} {_} {_} a b => some (D.apply p o a b) }

/-- Immutable source launch parameters. Integer addresses are separate from
all floating arithmetic in the value specification. -/
structure ForwardConfig where
  q : RegionName
  k : RegionName
  v : RegionName
  o : RegionName
  z : RegionName
  s_qk_h : Nat
  s_qk_t : Nat
  s_qk_d : Nat
  s_vo_h : Nat
  s_vo_t : Nat
  s_vo_d : Nat
  scale : ℝ
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BTL : Nat
  BTS : Nat
  BK : Nat
  BV : Nat

def queryChunk (c : ForwardConfig) {α : Type} (s : State α) : Nat := s.pids 1 * c.BTL

def keyChunk (c : ForwardConfig) {α : Type} (s : State α) : Nat :=
  s.pids 0 / ((c.V + c.BV - 1) / c.BV) * c.BK

def valueChunk (c : ForwardConfig) {α : Type} (s : State α) : Nat :=
  s.pids 0 % ((c.V + c.BV - 1) / c.BV) * c.BV

/-- Direct memory-address formula for a checked fp32 matrix load. Out-of-bounds
lanes use the selected source explicit zero padding in the access channel. -/
def matrixRead {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (R : RegionName) (base rows cols rowStride colStride rowOffset colOffset m n : Nat) :
    TileIndex [m,n] → α := fun i =>
  let row := rowOffset + i.1.val
  let col := colOffset + i.2.1.val
  let address := base + row * rowStride + col * colStride
  M.fp32Load (if row < rows ∧ col < cols then (s.mem R address).read .real
    else M.literal (some .fp32) .real 0)

def forwardQuery {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s : State α) : TileIndex [c.BTL,c.BK] → α := fun i =>
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
    (matrixRead M s c.q (s.pids 2 * c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
      (queryChunk c s) (keyChunk c s) c.BTL c.BK i)
    (M.literal (some .fp32) .real c.scale))

structure ForwardAccumulator (α : Type) (BTL BV : Nat) where
  output : TileIndex [BTL,BV] → α
  normalizer : TileIndex [BTL] → α

/-- One streaming update, independently expressed as arrays of primitive
values. The score is squared before the optional diagonal mask. Dot operands
retain the source cast and allow_tf32=false setting. -/
def forwardUpdate {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) (diagonal : Bool) (start : Nat)
    (previous : ForwardAccumulator α c.BTL c.BV) : ForwardAccumulator α c.BTL c.BV :=
  let key := matrixRead M s c.k (s.pids 2 * c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
    (keyChunk c s) start c.BK c.BTS
  let value := matrixRead M s c.v (s.pids 2 * c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
    start (valueChunk c s) c.BTS c.BV
  let raw := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (forwardQuery M c s) key
  let score : TileIndex [c.BTL,c.BTS] → α := fun i =>
    let squared := M.binary (some .fp32) .real .mul (raw i) (raw i)
    if diagonal then
      if start - queryChunk c s + i.2.1.val ≤ i.1.val then squared
      else M.literal (some .fp32) .real 0.0
    else squared
  let total := M.reduceSum (some .fp32) (shape := [c.BTL,c.BTS]) ⟨1, by simp⟩ Bool.false score
  let contribution := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (fun i => M.cast (some .fp32) .real .real (score i)) value
  { output := fun i => M.binary (some .fp32) .real .add (previous.output i) (contribution i)
    normalizer := fun i => M.binary (some .fp32) .real .add (previous.normalizer i) (total i) }

/-- Pure value recurrence over the exact source index sequence, independent
of registers, statements, or execution success. -/
def foldRange {β : Type} (update : Nat → β → β) (cur stop stride : Nat) (initial : β) : β :=
  if hzero : stride = 0 then initial
  else if hlt : cur < stop then foldRange update (cur + stride) stop stride (update cur initial)
  else initial
termination_by stop - cur
decreasing_by omega

def forwardValues {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) : ForwardAccumulator α c.BTL c.BV :=
  let zero := M.literal (some .fp32) .real 0
  let initial : ForwardAccumulator α c.BTL c.BV := ⟨fun _ => zero, fun _ => zero⟩
  let previous := foldRange (forwardUpdate M D c s Bool.false) 0 (queryChunk c s) c.BTS initial
  foldRange (forwardUpdate M D c s Bool.true) (queryChunk c s)
    ((s.pids 1 + 1) * c.BTL) c.BTS previous

/-- Two actual output-store folds in source order. This also specifies the
entire memory frame and retains the behavior when output regions alias. -/
def forwardMemory {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) : State α :=
  let values := forwardValues M D c s
  let keyBlock := s.pids 0 / ((c.V + c.BV - 1) / c.BV)
  let outputBase := (s.pids 2 + c.B * c.H * keyBlock) * c.s_vo_h
  let afterO := (TileShape.allIndices [c.BTL,c.BV]).foldl (fun t i =>
    let row := queryChunk c s + i.1.val
    let col := valueChunk c s + i.2.1.val
    if row < c.T ∧ col < c.V then t.write c.o
      (outputBase + row * c.s_vo_t + col * c.s_vo_d)
      (.mk .real (M.cast (some .fp32) .real .real (values.output i))) else t) s
  if s.pids 0 % ((c.V+c.BV-1)/c.BV) = 0 then
    (TileShape.allIndices [c.BTL]).foldl (fun t i =>
    let row := queryChunk c s + i.1.val
    if row < c.T then t.write c.z ((s.pids 2 + c.B * c.H * keyBlock) * c.T + row)
      (.mk .real (M.cast (some .fp32) .real .real (values.normalizer i))) else t) afterO
  else afterO

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
