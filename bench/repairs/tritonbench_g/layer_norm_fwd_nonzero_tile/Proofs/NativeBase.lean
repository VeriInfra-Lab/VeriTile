import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.TypedReturn
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeParameters
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.CompilerRecipes
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPrelude
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageMemory
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeGeometry

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section
namespace NativeMachine
structure Schedule (block : Nat) where
  reductions : Nat → Whole.Reductions block
  plan : AffinePlan
  plan1 : AffinePlan
end NativeMachine
namespace NativeSchedule
def snapshot {α : Type} [Inhabited α] (r : Regions) (stride : Strides) (s : State α) : NativeRow.Inputs α where
  x := fun i j => (s.mem r.x (i*stride.x+j)).read .real
  x1 := fun i j => (s.mem r.x1 (i*stride.x1+j)).read .real
  residual := fun i j => (s.mem r.residual (i*stride.residual+j)).read .real
  weight := fun j => (s.mem r.weight j).read .real
  bias := fun j => (s.mem r.bias j).read .real
  weight1 := fun j => (s.mem r.weight1 j).read .real
  bias1 := fun j => (s.mem r.bias1 j).read .real
  rowscale := fun i => (s.mem r.rowscale i).read .real
  seeds := fun i => (s.mem r.seeds i).read .int

def keep {α : Type} (a : Ops α) (seed : Int) (p : α) (col : Nat) : Bool :=
  a.greater ((Expr.random seed col).eval a) ((Expr.input p).eval a)

/-- Independent ordered FP target, using only the pre-launch input snapshot.
Owner-zero scalar stores and per-column vector reduction owners are distinct. -/
def oracle {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : α) (s : State α) : TypedSource.Result α M N (HostReturn.seedCount f M) :=
  let x := snapshot r stride s
  { y := fun i j => .mk .real ((NativeParameters.y red plan f x M N eps p i.val j.val).eval a)
    y1 := if f.w1 then some (fun i j => .mk .real ((NativeParameters.y1 red plan1 f x M N eps p i.val j.val).eval a)) else none
    mean := if f.rms then none else some (fun i => .mk .real (a.cast32 ((NativeParameters.mean red f x M N p i.val 0).eval a)))
    rstd := fun i => .mk .real (a.cast32 ((NativeParameters.rstd red f x M N eps p i.val 0).eval a))
    residual := if f.residualOut then (fun i j => .mk .real
      (a.cast32 ((NativeParameters.combined f x M N p i.val j.val).eval a)))
      else (fun i j => s.mem r.x (i.val*stride.x+j.val))
    seeds := if f.dropout then some (fun i => s.mem r.seeds i.val) else none
    mask := if f.dropout && f.dropoutMask then some (fun i j => .mk .bool (keep a (x.seeds i.val) p j.val)) else none
    mask1 := if f.dropout && f.dropoutMask && f.x1 then
      some (fun i j => .mk .bool (keep a (x.seeds (M+i.val)) p j.val)) else none }

end NativeSchedule
namespace KernelEntry
/-- Source pointer tags; read aliases are represented by byte descriptors. -/
def regions : Regions :=
  ⟨"X","Y","W","B","RESIDUAL","X1","W1","B1","Y1","RESIDUAL_OUT",
    "ROWSCALE","SEEDS","MASK","MEAN","RSTD"⟩
end KernelEntry
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
