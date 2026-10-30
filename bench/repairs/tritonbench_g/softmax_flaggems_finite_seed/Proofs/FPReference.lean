import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Register-free source-order FP graphs. Every arithmetic operation retains
its selected FP32 tag. These definitions contain no source AST or interpreter.
The final native adapter must expand reductions into its selected scalar tree;
no whole-reduction/native-to-Real equality is assumed here. -/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPReference
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

def literal (M : Algebra α) (x : ℝ) := M.literal (some .fp32) .real x
def cast (M : Algebra α) := M.cast (some .fp32) .real .real
def add (M : Algebra α) := M.binary (some .fp32) .real .add
def sub (M : Algebra α) := M.binary (some .fp32) .real .sub
def mul (M : Algebra α) := M.binary (some .fp32) .real .mul
def div (M : Algebra α) := M.binary (some .fp32) .real .div
def max (M : Algebra α) := M.binary (some .fp32) .real .max
def exp (M : Algebra α) := M.unary (some .fp32) .exp

def read (s : State α) (r : RegionName) (o : Nat) : α := (s.mem r o).read .real

def backwardInnerOffset {B C : Nat} (row N : Nat) (i : TileIndex [C,B]) :=
  (row * C + i.1.val) * N + i.2.1.val

def backwardInnerActive {B C : Nat} (row rows N : Nat) (i : TileIndex [C,B]) :=
  decide (row*C+i.1.val < rows) && decide (i.2.1.val < N)

def backwardInnerLoad (M : Algebra α) (s : State α) (P : RegionName)
    (row rows N B C : Nat) (i : TileIndex [C,B]) : α :=
  M.fp32Load (if backwardInnerActive row rows N i then read s P (backwardInnerOffset row N i)
  else literal M 0)

def backwardInnerScale (M : Algebra α) (s : State α) (P G : RegionName)
    (row rows N B C : Nat) : Values α .real [C] :=
  M.reduceSum (some .fp32) (shape := [C,B]) ⟨1,by simp⟩ Bool.false
    (fun i => mul M (backwardInnerLoad M s P row rows N B C i)
      (backwardInnerLoad M s G row rows N B C i))

def backwardInnerValue (M : Algebra α) (s : State α) (P G : RegionName)
    (row rows N B C : Nat) (i : TileIndex [C,B]) : α :=
  cast M (mul M (backwardInnerLoad M s P row rows N B C i)
    (sub M (backwardInnerLoad M s G row rows N B C i)
      (backwardInnerScale M s P G row rows N B C ⟨i.1, PUnit.unit⟩)))

def noninnerOffset {B C : Nat} (row col N K : Nat) (i : TileIndex [B,C]) :=
  row*N*K + i.1.val*K + (col*C+i.2.1.val)

def noninnerActive {B C : Nat} (col N K : Nat) (i : TileIndex [B,C]) :=
  decide (i.1.val<N) && decide (col*C+i.2.1.val<K)

def backwardNoninnerLoad (M : Algebra α) (s : State α) (P : RegionName)
    (row col N K B C : Nat) (i : TileIndex [B,C]) : α :=
  M.fp32Load (if noninnerActive col N K i then read s P (noninnerOffset row col N K i)
    else literal M 0)

def backwardNoninnerScale (M : Algebra α) (s : State α) (P G : RegionName)
    (row col N K B C : Nat) : Values α .real [C] :=
  M.reduceSum (some .fp32) (shape := [B,C]) ⟨0,by simp⟩ Bool.false
    (fun i => mul M (backwardNoninnerLoad M s P row col N K B C i)
      (backwardNoninnerLoad M s G row col N K B C i))

def backwardNoninnerValue (M : Algebra α) (s : State α) (P G : RegionName)
    (row col N K B C : Nat) (i : TileIndex [B,C]) : α :=
  cast M (mul M (backwardNoninnerLoad M s P row col N K B C i)
    (sub M (backwardNoninnerLoad M s G row col N K B C i)
      (backwardNoninnerScale M s P G row col N K B C i.2)))

def innerOffset {B : Nat} (row N : Nat) (i : TileIndex [B]) := row*N+i.1.val

def innerActive {B : Nat} (N : Nat) (i : TileIndex [B]) := decide (i.1.val<N)

def forwardInnerLoad (M : Algebra α) (s : State α) (X : RegionName)
    (row N B : Nat) (i : TileIndex [B]) : α :=
  cast M (if innerActive N i then read s X (innerOffset row N i) else M.negInf)

def forwardInnerExp (M : Algebra α) (s : State α) (X : RegionName)
    (row N B : Nat) (i : TileIndex [B]) : α :=
  exp M (sub M (forwardInnerLoad M s X row N B i)
    (M.reduceMax (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false
      (forwardInnerLoad M s X row N B) PUnit.unit))

def forwardInnerValue (M : Algebra α) (s : State α) (X : RegionName)
    (row N B : Nat) (i : TileIndex [B]) : α :=
  cast M (div M (forwardInnerExp M s X row N B i)
    (M.reduceSum (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false
      (forwardInnerExp M s X row N B) PUnit.unit))

def forwardNoninnerLoad (M : Algebra α) (s : State α) (X : RegionName)
    (row col N K B C : Nat) (i : TileIndex [B,C]) : α :=
  M.fp32Load (if noninnerActive col N K i then read s X (noninnerOffset row col N K i)
    else M.negInf)

def forwardNoninnerExp (M : Algebra α) (s : State α) (X : RegionName)
    (row col N K B C : Nat) (i : TileIndex [B,C]) : α :=
  exp M (sub M (forwardNoninnerLoad M s X row col N K B C i)
    (M.reduceMax (some .fp32) (shape:=[B,C]) ⟨0,by simp⟩ Bool.false
      (forwardNoninnerLoad M s X row col N K B C) ⟨i.2.1,PUnit.unit⟩))

def forwardNoninnerValue (M : Algebra α) (s : State α) (X : RegionName)
    (row col N K B C : Nat) (i : TileIndex [B,C]) : α :=
  cast M (div M (forwardNoninnerExp M s X row col N K B C i)
    (M.reduceSum (some .fp32) (shape:=[B,C]) ⟨0,by simp⟩ Bool.false
      (forwardNoninnerExp M s X row col N K B C) ⟨i.2.1,PUnit.unit⟩))

def scatter {shape : TileShape} (s : State α) (Y : RegionName)
    (off : TileIndex shape → Nat) (active : TileIndex shape → Bool)
    (value : TileIndex shape → α) : State α :=
  (TileShape.allIndices shape).foldl
    (fun st i => if active i then st.write Y (off i) (.mk .real (value i)) else st) s

theorem writeFold_mem_congr {ι : Type} (xs : List ι) (r : ι → RegionName)
    (off : ι → Nat) (active : ι → Prop) [DecidablePred active] (value : ι → Cell α)
    (s t : State α) (h : s.mem = t.mem) :
    (xs.foldl (fun st i => if active i then st.write (r i) (off i) (value i) else st) s).mem =
    (xs.foldl (fun st i => if active i then st.write (r i) (off i) (value i) else st) t).mem := by
  induction xs generalizing s t with
  | nil => exact h
  | cons i xs ih =>
    simp only [List.foldl_cons]
    apply ih
    split <;> simp_all [State.write]

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPReference
