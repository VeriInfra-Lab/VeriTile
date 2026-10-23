import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Scalar
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.L2NormBwd.Physical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- Actual two-dimensional metadata after the corresponding host operation.
Zero strides and overlapping read views are permitted. -/
structure Matrix where
  view : View
  rows : Nat
  cols : Nat
  rowStride : Nat
  colStride : Nat
  shape : view.shape=[rows,cols]
  strides : view.strides=[rowStride,colStride]

def Matrix.index (m : Matrix) (r c : Nat) := m.view.storageOffset+r*m.rowStride+c*m.colStride
def Matrix.key (m : Matrix) (r c : Nat) := m.view.storage.key (m.index r c)
def Matrix.backed (m : Matrix) : Prop := ∀ r c,r<m.rows → c<m.cols → m.view.storage.contains (m.key r c)
def Matrix.injective (m : Matrix) : Prop := ∀ r c r' c',r<m.rows → c<m.cols → r'<m.rows → c'<m.cols →
  m.key r c=m.key r' c' → r=r' ∧ c=c'

theorem Matrix.numel (m : Matrix) : m.view.numel=m.rows*m.cols := by simp [View.numel,m.shape]

theorem Matrix.logical_key (m : Matrix) (r c : Nat) (hc : c<m.cols) :
    m.view.key (r*m.cols+c)=m.key r c := by
  simp only [View.key,m.shape,m.strides,offset,List.prod_cons,List.prod_nil,Nat.mul_one,
    Nat.div_one,Nat.mod_one,Nat.zero_mul,Nat.add_zero]
  have pos : 0<m.cols := by omega
  rw [Nat.mul_comm r m.cols,Nat.mul_add_div pos,Nat.div_eq_of_lt hc,Nat.add_zero,Nat.add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt hc]
  simp only [Matrix.key,Matrix.index,Nat.add_assoc,Nat.mod_eq_of_lt hc]

theorem Matrix.from_backed (m : Matrix) (h : m.view.backed) : m.backed := by
  intro r c hr hc
  rw [←m.logical_key r c hc]
  apply h
  rw [m.numel]
  have bound := Nat.mul_le_mul_right m.cols (show r+1≤m.rows by omega)
  simp only [Nat.add_mul,Nat.one_mul] at bound
  omega

theorem Matrix.disjoint (m : Matrix) (a b c d : Nat) (different : m.key a b≠m.key c d) :
    (m.key a b).disjoint (m.key c d) := by
  apply Allocation.indices_disjoint
  intro same
  exact different (congrArg m.view.storage.key same)

structure Layout where
  x : Matrix
  dy : Matrix
  dx : Matrix
  block : Nat

def Layout.strides (p : Layout) : Strides :=
  ⟨p.x.rowStride,p.x.colStride,p.dy.rowStride,p.dy.colStride,p.dx.rowStride,p.dx.colStride⟩

structure Valid (p : Layout) : Prop where
  positive : 0<p.x.cols
  block : p.x.cols≤p.block
  dyCols : p.dy.cols=p.x.cols
  dyRows : p.x.rows≤p.dy.rows
  dxCols : p.dx.cols=p.x.cols
  dxRows : p.dx.rows=p.x.rows
  xBacked : p.x.backed
  dyBacked : p.dy.backed
  dxBacked : p.dx.backed
  dxUnique : p.dx.injective
  xFresh : p.x.view.storage.separate p.dx.view.storage
  dyFresh : p.dy.view.storage.separate p.dx.view.storage
  xResolved : p.x.view.negative=Bool.false
  dyResolved : p.dy.view.negative=Bool.false
  dxResolved : p.dx.view.negative=Bool.false

/-- A row operation with the ordinary active-input extensionality law.
The source and raw-native instances are constructed and proved separately. -/
structure Reduction (α : Type) where
  apply : Nat → Nat → α → (Nat → α) → (Nat → α) → Nat → α
  congr : ∀ N B eps xs dys us vs,
    (∀ i,i<N → xs i=us i) → (∀ i,i<N → dys i=vs i) →
    ∀ i,i<N → apply N B eps xs dys i=apply N B eps us vs i

def sourceReduction {α : Type} (alg : Algebra α) : Reduction α :=
  ⟨Tile.scalar alg,Tile.scalar_congr alg⟩

def inputs {M α β : Type} (ops : Ops M α β) (s : M) (m : Matrix) (r : Nat) : Nat → α :=
  fun c => ops.read s (m.key r c)
def value {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (p : Layout) (eps : α)
    (s : M) (r c : Nat) : α :=
  alg.apply p.x.cols p.block eps (inputs ops s p.x r) (inputs ops s p.dy r) c

def row {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (p : Layout) (eps : α)
    (r : Nat) (s : M) : M :=
  stores ops (p.dx.key r) (value ops alg p eps s r) (List.range p.x.cols) s

theorem row_frame {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (p : Layout) (eps : α)
    (r : Nat) (s : M) (key : Key)
    (outside : ∀ c,c<p.x.cols → key.disjoint (p.dx.key r c)) :
    ops.read (row ops alg p eps r s) key=ops.read s key :=
  read_frame ops _ _ _ _ _ (fun c hc => outside c (List.mem_range.mp hc))

theorem row_bytes {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (p : Layout) (eps : α)
    (r : Nat) (s : M) (owner address : Nat)
    (outside : ∀ c,c<p.x.cols → ¬(p.dx.key r c).covers owner address) :
    ops.observe (row ops alg p eps r s) owner address=ops.observe s owner address :=
  bytes_frame ops _ _ _ _ _ _ (fun c hc => outside c (List.mem_range.mp hc))

theorem row_values {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (p : Layout)
    (valid : Valid p) (eps : α) (r c : Nat) (s : M) (hr : r<p.x.rows) (hc : c<p.x.cols) :
    ops.read (row ops alg p eps r s) (p.dx.key r c)=value ops alg p eps s r c := by
  apply readback ops _ _ _ _ c (List.mem_range.mpr hc)
  · intro j hj same
    have eq := valid.dxUnique r j r c (by rw [valid.dxRows];exact hr)
      (by rw [valid.dxCols];exact List.mem_range.mp hj) (by rw [valid.dxRows];exact hr)
      (by rw [valid.dxCols];exact hc) same
    rw [eq.2]
  · intro j hj different
    exact p.dx.disjoint _ _ _ _ (Ne.symm different)

theorem row_inputs {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (p : Layout)
    (valid : Valid p) (eps : α) (head r c : Nat) (s : M) (hh : head<p.x.rows)
    (hr : r<p.x.rows) (hc : c<p.x.cols) :
    value ops alg p eps (row ops alg p eps head s) r c=value ops alg p eps s r c := by
  apply alg.congr p.x.cols p.block eps _ _ _ _ ?_ ?_ c hc
  · intro i hi
    apply row_frame
    intro j hj
    exact Allocation.spans_separate _ _ valid.xFresh _ _ (valid.xBacked r i hr hi)
      (valid.dxBacked head j (by rw [valid.dxRows];exact hh) (by rw [valid.dxCols];exact hj))
  · intro i hi
    apply row_frame
    intro j hj
    exact Allocation.spans_separate _ _ valid.dyFresh _ _
      (valid.dyBacked r i (lt_of_lt_of_le hr valid.dyRows) (by rw [valid.dyCols];exact hi))
      (valid.dxBacked head j (by rw [valid.dxRows];exact hh) (by rw [valid.dxCols];exact hj))

def run {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (p : Layout) (eps : α) : List Nat → M → M
  | [],s => s
  | r::rest,s => run ops alg p eps rest (row ops alg p eps r s)

theorem run_frame {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (p : Layout) (eps : α)
    (rows : List Nat) (s : M) (key : Key)
    (outside : ∀ r∈rows,∀ c,c<p.x.cols → key.disjoint (p.dx.key r c)) :
    ops.read (run ops alg p eps rows s) key=ops.read s key := by
  induction rows generalizing s with
  | nil => rfl
  | cons r rest ih =>
    rw [run,ih _ (fun j hj => outside j (by simp [hj]))]
    exact row_frame ops alg p eps r s key (outside r (by simp))

theorem run_bytes {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (p : Layout) (eps : α)
    (rows : List Nat) (s : M) (owner address : Nat)
    (outside : ∀ r∈rows,∀ c,c<p.x.cols → ¬(p.dx.key r c).covers owner address) :
    ops.observe (run ops alg p eps rows s) owner address=ops.observe s owner address := by
  induction rows generalizing s with
  | nil => rfl
  | cons r rest ih =>
    rw [run,ih _ (fun j hj => outside j (by simp [hj]))]
    exact row_bytes ops alg p eps r s owner address (outside r (by simp))

theorem run_values {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (p : Layout)
    (valid : Valid p) (eps : α) (rows : List Nat) (unique : rows.Nodup)
    (bounded : ∀ r∈rows,r<p.x.rows) (s : M) (r c : Nat) (member : r∈rows) (hc : c<p.x.cols) :
    ops.read (run ops alg p eps rows s) (p.dx.key r c)=value ops alg p eps s r c := by
  induction rows generalizing s with
  | nil => simp at member
  | cons head rest ih =>
    obtain ⟨notMember,uniqueRest⟩ := List.nodup_cons.mp unique
    have hr := bounded r member
    have hh := bounded head (by simp)
    rcases List.mem_cons.mp member with same | tail
    · subst r
      rw [run,run_frame]
      · exact row_values ops alg p valid eps head c s hh hc
      · intro r hr' j hj
        apply p.dx.disjoint
        intro same
        have eq := valid.dxUnique head c r j (by rw [valid.dxRows];exact hh)
          (by rw [valid.dxCols];exact hc) (by rw [valid.dxRows];exact bounded r (by simp [hr']))
          (by rw [valid.dxCols];exact hj) same
        exact notMember (eq.1 ▸ hr')
    · rw [run,ih uniqueRest (fun j hj => bounded j (by simp [hj])) _ tail]
      exact row_inputs ops alg p valid eps head r c s hh hr hc

#axiomsClean run_values
#axiomsClean run_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Physical
