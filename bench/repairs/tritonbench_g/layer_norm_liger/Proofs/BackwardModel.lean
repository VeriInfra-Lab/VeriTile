import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardKernels
import bench.tritonbench_g.token_attn_reduceV.Proofs.FPContract
namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

structure Acc (α : Type) (p : Parameters) where
  state : State α
  (dw db : TileIndex [p.block] → α)

def zero {α} (M : Algebra α) := M.literal (some .fp32) .real 0
def padding {α} (M : Algebra α) := M.cast none .real .fp32 (M.literal none .real 0)
def cast {α} (M : Algebra α) (v : α) := M.cast (some .fp32) .fp32 .real v
def stored {α} (M : Algebra α) (v : α) : Cell α := .mk .fp32 (M.cast none .real .fp32 v)
def binary {α} (M : Algebra α) (op : Binary) (x y : α) := M.binary (some .fp32) .real op x y
def denom {α} (M : Algebra α) (p : Parameters) :=
  M.cast (some .fp32) .real .real (M.literal (some .fp32) .real (p.columns : ℝ))
def load {α} [Inhabited α] (M : Algebra α) (p : Parameters) (a : Acc α p)
    (r : RegionName) (offset : Nat) (i : TileIndex [p.block]) :=
  if i.1.val < p.columns then (a.state.mem r (offset+i.1.val)).read .fp32 else padding M
def mean {α} [Inhabited α] (p : Parameters) (a : Acc α p) (row : Nat) :=
  (a.state.mem p.mean row).read .fp32
def rstd {α} [Inhabited α] (p : Parameters) (a : Acc α p) (row : Nat) :=
  (a.state.mem p.rstd row).read .fp32
def xhat {α} [Inhabited α] (M : Algebra α) (p : Parameters) (a : Acc α p)
    (row : Nat) (i : TileIndex [p.block]) :=
  if i.1.val < p.columns then cast M (M.binary (some .fp32) .fp32 .mul
    (M.binary (some .fp32) .fp32 .sub (load M p a p.x (row*p.xStride) i) (mean p a row))
    (rstd p a row)) else M.literal none .real 0
def wdy {α} [Inhabited α] (M : Algebra α) (p : Parameters) (a : Acc α p)
    (row : Nat) (i : TileIndex [p.block]) :=
  M.binary none .fp32 .mul (load M p a p.weight 0 i) (load M p a p.dy (row*p.dyStride) i)
def c1 {α} [Inhabited α] (M : Algebra α) (p : Parameters) (a : Acc α p) (row : Nat) :=
  binary M .div (M.reduceSum (some .fp32) ⟨0, by simp⟩ Bool.false
    (fun i => binary M .mul (xhat M p a row i) (cast M (wdy M p a row i))) PUnit.unit) (denom M p)
def c2 {α} [Inhabited α] (M : Algebra α) (R : SumPrimitive α) (p : Parameters) (a : Acc α p) (row : Nat) :=
  binary M .div (cast M (R.apply (some .fp32) ⟨0, by simp⟩ Bool.false (wdy M p a row) PUnit.unit)) (denom M p)
def dxValue {α} [Inhabited α] (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (a : Acc α p) (row : Nat) (i : TileIndex [p.block]) :=
  binary M .mul (binary M .sub (cast M (wdy M p a row i))
    (binary M .add (binary M .mul (xhat M p a row i) (c1 M p a row)) (c2 M R p a row)))
    (cast M (rstd p a row))
def update {α} [Inhabited α] (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (row : Nat) (a : Acc α p) : Acc α p where
  state := (TileShape.allIndices [p.block]).foldl (fun s i =>
    if i.1.val < p.columns then s.write p.dx (row*p.dxStride+i.1.val) (stored M (dxValue M R p a row i)) else s) a.state
  dw := fun i => binary M .add (a.dw i)
    (binary M .mul (cast M (load M p a p.dy (row*p.dyStride) i)) (xhat M p a row i))
  db := fun i => binary M .add (a.db i) (cast M (load M p a p.dy (row*p.dyStride) i))
def initial {α} (M : Algebra α) (p : Parameters) (s : State α) : Acc α p :=
  ⟨s, fun _ => zero M, fun _ => zero M⟩
def start {α} (p : Parameters) (s : State α) := s.pids 0*p.rowsPerProgram
def stop {α} (p : Parameters) (s : State α) := min ((s.pids 0+1)*p.rowsPerProgram) p.rows
def accumulated {α} [Inhabited α] (M : Algebra α) (R : SumPrimitive α) (p : Parameters) (s : State α) :=
  foldRange (update M R p) (start p s) (stop p s) 1 (initial M p s)
def finish {α} (M : Algebra α) (p : Parameters) (pid : Nat) (a : Acc α p) : State α :=
  let t := (TileShape.allIndices [p.block]).foldl (fun s i => if i.1.val < p.columns then
    s.write p.dw (pid*p.dwStride+i.1.val) (stored M (a.dw i)) else s) a.state
  (TileShape.allIndices [p.block]).foldl (fun s i => if i.1.val < p.columns then
    s.write p.db (pid*p.dbStride+i.1.val) (stored M (a.db i)) else s) t
def reference {α} [Inhabited α] (M : Algebra α) (R : SumPrimitive α) (p : Parameters) (s : State α) :=
  finish M p (s.pids 0) (accumulated M R p s)
end
end VeriTile.Bench.Repairs.LayerNormLiger.Backward
