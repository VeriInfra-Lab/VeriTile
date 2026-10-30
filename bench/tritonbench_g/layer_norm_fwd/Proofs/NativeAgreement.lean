import bench.tritonbench_g.layer_norm_fwd.NativeIO
import bench.tritonbench_g.layer_norm_fwd.Proofs.RealIOInputs

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

def snapshot (r : Regions) (stride : Strides) (s : State α) : Inputs α where
  x:=fun row col=>(s.mem r.x (row*stride.x+col)).read .real
  x1:=fun row col=>(s.mem r.x1 (row*stride.x1+col)).read .real
  residual:=fun row col=>(s.mem r.residual (row*stride.residual+col)).read .real
  weight:=fun col=>(s.mem r.weight col).read .real
  bias:=fun col=>(s.mem r.bias col).read .real
  weight1:=fun col=>(s.mem r.weight1 col).read .real
  bias1:=fun col=>(s.mem r.bias1 col).read .real
  rowscale:=fun row=>(s.mem r.rowscale row).read .real
  seeds:=fun row=>(s.mem r.seeds row).read .int

structure Agree (f : Flags) (M N : Nat) (a b : Inputs α) : Prop where
  x : ∀ i, i<M → ∀ j, j<N → a.x i j=b.x i j
  x1 : f.x1=Bool.true → ∀ i, i<M → ∀ j, j<N → a.x1 i j=b.x1 i j
  residual : f.residual=Bool.true → ∀ i, i<M → ∀ j, j<N → a.residual i j=b.residual i j
  weight : ∀ j, j<N → a.weight j=b.weight j
  bias : f.bias=Bool.true → ∀ j, j<N → a.bias j=b.bias j
  weight1 : f.w1=Bool.true → ∀ j, j<N → a.weight1 j=b.weight1 j
  bias1 : f.w1=Bool.true → f.b1=Bool.true → ∀ j, j<N → a.bias1 j=b.bias1 j
  rowscale : f.rowscale=Bool.true → ∀ i, i<M → a.rowscale i=b.rowscale i
  rowscale1 : f.x1=Bool.true → f.rowscale=Bool.true → ∀ i, i<M → a.rowscale (M+i)=b.rowscale (M+i)
  seeds : f.dropout=Bool.true → ∀ i, i<M → a.seeds i=b.seeds i
  seeds1 : f.x1=Bool.true → f.dropout=Bool.true → ∀ i, i<M → a.seeds (M+i)=b.seeds (M+i)


theorem combined_snapshot (A : Algebra α) (cmp : TypedSource.Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (p : ℝ) (row block : Nat) (s : State α) :
    TypedSource.combinedValues A cmp f s r stride M N p row block=
      combined A cmp f (snapshot r stride s) M N p row block := rfl

namespace Agree
variable {f : Flags} {M N : Nat} {a b : Inputs α} (h : Agree f M N a b)
include h

theorem loaded_x (A : Algebra α) (row : Nat) (hr : row<M) (col : Nat) :
    loaded A N (a.x row) col=loaded A N (b.x row) col := by
  unfold loaded
  split_ifs with on
  · rw [h.x row hr col on]
  · rfl

theorem first (A : Algebra α) (cmp : TypedSource.Comparisons A) (p : ℝ)
    (row : Nat) (hr : row<M) (j : TileIndex [block]) :
    stream A cmp f a row p (fun i=>loaded A N (a.x row) i.1.val) j=
      stream A cmp f b row p (fun i=>loaded A N (b.x row) i.1.val) j := by
  cases hs : f.rowscale <;> cases hd : f.dropout <;>
    simp [stream,hs,hd,h.loaded_x,h.rowscale,h.seeds,hr]

theorem loaded_x1 (A : Algebra α) (hx : f.x1=Bool.true) (row : Nat) (hr : row<M) (col : Nat) :
    loaded A N (a.x1 row) col=loaded A N (b.x1 row) col := by
  unfold loaded
  split_ifs with on
  · rw [h.x1 hx row hr col on]
  · rfl

theorem second (A : Algebra α) (cmp : TypedSource.Comparisons A) (p : ℝ)
    (hx : f.x1=Bool.true) (row : Nat) (hr : row<M) (j : TileIndex [block]) :
    stream A cmp f a (M+row) p (fun i=>loaded A N (a.x1 row) i.1.val) j=
      stream A cmp f b (M+row) p (fun i=>loaded A N (b.x1 row) i.1.val) j := by
  cases hs : f.rowscale <;> cases hd : f.dropout <;>
    simp [stream,hs,hd,h.loaded_x1,h.rowscale1,h.seeds1,hr,hx]

theorem loaded_residual (A : Algebra α) (hx : f.residual=Bool.true) (row : Nat) (hr : row<M) (col : Nat) :
    loaded A N (a.residual row) col=loaded A N (b.residual row) col := by
  unfold loaded
  split_ifs with on
  · rw [h.residual hx row hr col on]
  · rfl

theorem combined (A : Algebra α) (cmp : TypedSource.Comparisons A) (p : ℝ)
    (row : Nat) (hr : row<M) (block : Nat) :
    NativeInterface.combined A cmp f a M N p row block=
      NativeInterface.combined A cmp f b M N p row block := by
  funext j
  dsimp only [NativeInterface.combined]
  rw [h.first A cmp p row hr j]
  cases hx : f.x1 <;> cases hs : f.residual <;>
    simp [hx,hs,h.second,h.loaded_residual,hr]

end Agree
#axiomsClean combined_snapshot
#axiomsClean Agree.combined
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
