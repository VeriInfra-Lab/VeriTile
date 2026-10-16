import bench.tritonbench_g.lightning_attention.Proofs.FPSource
namespace VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

/-- A complete default/optioned dot implementation. Total availability is
explicit, and no scalar expansion, exact-real identity or hardware binding is
assumed. Precision, options, batch layout and every operand are retained. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    Values α .real (batch ++ [m,k]) → Values α .real (batch ++ [k,n]) →
    Values α .real (batch ++ [m,n])

def backend {α : Type} (M : Algebra α) (D : Dot α) : Algebra α :=
  { M with dot := fun p {_} {_} {_} {_} a b => some (D.apply p {} a b)
           dotWithOptions := fun p opts {_} {_} {_} {_} a b => some (D.apply p opts a b) }

def read32 {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r : RegionName) (o : Nat) : α := M.cast (some .fp32) .fp32 .real ((s.mem r o).read .fp32)

def maskedRead32 {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r : RegionName) (o : Nat) (active : Bool) : α :=
  M.cast (some .fp32) .fp32 .real
    (if active then (s.mem r o).read .fp32 else M.cast (some .fp32) .real .fp32 (M.literal (some .fp32) .real 0.0))

def kvOffset (p : Parameters) (i : TileIndex [p.keyDim,p.columns]) :=
  i.1.val*p.columns+i.2.1.val

def interOffset (p : Parameters) (i : TileIndex [p.rows,p.columns]) :=
  i.1.val*p.columns+i.2.1.val

def bodyOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.rows,p.columns]) :=
  s.pids 0*p.length*p.valueDim+(p.blockIndex*p.rows+i.1.val)*p.valueDim+
    s.pids 1*p.columns+i.2.1.val

def bodyActive (p : Parameters) (i : TileIndex [p.rows,p.columns]) : Bool :=
  p.blockIndex*p.rows+i.1.val<p.length

def gradOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.rows,p.columns]) :=
  s.pids 0*p.length*p.valueDim+(s.pids 1*p.rows+i.1.val)*p.valueDim+i.2.1.val

def gradActive {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.rows,p.columns]) : Bool :=
  (s.pids 1*p.rows+i.1.val<p.length) && (i.2.1.val<p.columns)

def carryValues {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    Values α .real [p.keyDim,p.columns] := fun i => read32 M s p.carry (kvOffset p i)

def kvValues {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (p : Parameters) (s : State α) :
    Values α .fp32 [p.keyDim,p.columns] :=
  let k : Values α .real [p.keyDim,p.rows] := fun i => read32 M s p.key (i.1.val*p.rows+i.2.1.val)
  let v : Values α .real [p.rows,p.columns] := fun i => read32 M s p.value (i.1.val*p.columns+i.2.1.val)
  let update := D.apply (some .fp32) {} (batch := []) k v
  fun i => M.cast none .real .fp32
    (M.binary (some .fp32) .real .add (carryValues M p s i) (update i))

def interValues {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (p : Parameters) (s : State α) :
    Values α .fp32 [p.rows,p.columns] :=
  let q : Values α .real [p.rows,p.keyDim] := fun i => read32 M s p.query (i.1.val*p.keyDim+i.2.1.val)
  let result := D.apply (some .fp32) {} (batch := []) q (carryValues M p s)
  fun i => M.cast none .real .fp32 (result i)

def bodyQuery {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    Values α .real [p.rows,p.keyDim] := fun i => maskedRead32 M s p.query
  (s.pids 0*p.length*p.keyDim+(p.blockIndex*p.rows+i.1.val)*p.keyDim+i.2.1.val)
  (p.blockIndex*p.rows+i.1.val<p.length)

def bodyKey {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    Values α .real [p.keyDim,p.rows] := fun i => maskedRead32 M s p.key
  (s.pids 0*p.length*p.keyDim+(p.blockIndex*p.rows+i.2.1.val)*p.keyDim+i.1.val)
  (p.blockIndex*p.rows+i.2.1.val<p.length)

def bodyValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    Values α .real [p.rows,p.columns] := fun i => maskedRead32 M s p.value (bodyOffset p s i) (bodyActive p i)

def bodyScore {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (p : Parameters) (s : State α) :
    Values α .real [p.rows,p.rows] :=
  let raw := D.apply (some .fp32) {} (batch := []) (bodyQuery M p s) (bodyKey M p s)
  fun i => if i.2.1.val ≤ i.1.val then raw i else M.literal (some .fp32) .real 0.0

/-- Independently specified intra/inter dot graph from the initial typed
memory. It retains partial-tail padding and the original block-relative causal
predicate; it does not replace either dot by a real finite sum. -/
def bodyValues {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) (p : Parameters) (s : State α) :
    Values α .fp32 [p.rows,p.columns] :=
  let intra := D.apply (some .fp32) {} (batch := []) (bodyScore M D p s) (bodyValue M p s)
  let inter := D.apply (some .fp32) {} (batch := []) (bodyQuery M p s) (carryValues M p s)
  fun i => M.cast none .real .fp32 (M.binary (some .fp32) .real .add (intra i) (inter i))

def gradValues {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    Values α .fp32 [p.rows,p.columns] := fun i => M.cast none .fp32 .fp32
  (if gradActive p s i then (s.mem p.query (gradOffset p s i)).read .fp32
   else M.cast none .real .fp32 (M.literal none .real 0.0))

def written {α : Type} {shape : TileShape} (output : RegionName) (offset : TileIndex shape → Nat)
    (active : TileIndex shape → Bool) (v : TileIndex shape → α) (s : State α) : State α :=
  (TileShape.allIndices shape).foldl
    (fun t i => if active i then t.write output (offset i) (.mk .fp32 (v i)) else t) s

def Post {α : Type} {shape : TileShape} (output : RegionName) (offset : TileIndex shape → Nat)
    (active : TileIndex shape → Bool) (v : TileIndex shape → α) (s final : State α) : Prop :=
  (∀ i, active i = Bool.true → final.mem output (offset i) = .mk .fp32 (v i)) ∧
  (∀ r o, (r ≠ output ∨ ∀ i, active i = Bool.true → o ≠ offset i) → final.mem r o = s.mem r o)
end
end VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
