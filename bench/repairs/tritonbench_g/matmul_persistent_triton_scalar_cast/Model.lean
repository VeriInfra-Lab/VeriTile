import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Kernels
import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.Execution
import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

structure Parameters where
  problem : Problem
  am : Nat
  ak : Nat
  bk : Nat
  bn : Nat
  cm : Nat
  cn : Nat

def source (p : Parameters) : ComputeKernel :=
  TypedSource.kernel p.problem.profile "A" "B" "C" p.problem.M p.problem.N p.problem.K
    p.am p.ak p.bk p.bn p.cm p.cn 128 256 p.problem.profile.blockK 8 p.problem.SMS

def prelude (p : Parameters) := (source p).surfaceBody.take 15

def outer (p : Parameters) : List ComputeStmt := match (source p).surfaceBody[15]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []
def tilePrelude (p : Parameters) := (outer p).take 13
def inner (p : Parameters) : List ComputeStmt := match (outer p)[13]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []
def tileSuffix (p : Parameters) := (outer p).drop 14

def dtype : Storage → TileDType
  | .fp16 => .fp16
  | .f8e4 => .f8e4

def matrix {α : Type} (P : Source.Arithmetic α α) {m k n : Nat}
    (a : TileIndex [m,k] → α) (b : TileIndex [k,n] → α)
    (acc : TileIndex [m,n] → α) : Option (TileIndex [m,n] → α) :=
  if hm : m=128 then if hn : n=256 then
    some (fun i => P.dot k
      (fun r t => a (⟨r.val,by omega⟩,t,PUnit.unit))
      (fun t c => b (t,⟨c.val,by omega⟩,PUnit.unit))
      (fun r c => acc (⟨r.val,by omega⟩,⟨c.val,by omega⟩,PUnit.unit))
      ⟨i.1.val,by omega⟩ ⟨i.2.1.val,by omega⟩)
    else none else none

def backend {α : Type} (P : Source.Arithmetic α α) : Algebra α where
  literal := fun _ d _ => if d=.real ∨ d=.fp32 then P.accumulatorZero else P.inputZero
  negInf := P.inputZero
  binary := fun _ _ _ _ _ => P.inputZero
  unary := fun _ _ _ => P.inputZero
  cast := fun _ _ dst x => match dst with
    | .fp16 => P.cast .fp16 x
    | .f8e4 => P.cast .f8e4 x
    | _ => x
  fromNat := fun _ _ => P.inputZero
  fromInt := fun _ _ => P.inputZero
  fp32Bits := fun _ => P.inputZero
  fp32Load := id
  reduceMax := fun _ _ _ _ _ _ => P.inputZero
  reduceSum := fun _ _ _ _ _ _ => P.inputZero
  dotAcc := fun _ left right out _ {batch} {m} {k} {n} a b acc =>
    match batch,left,right,out with
    | [],.fp16,.fp16,.real => matrix P a b acc
    | [],.f8e4,.f8e4,.real => matrix P a b acc
    | _,_,_,_ => none

variable {α : Type} [Inhabited α]

def rawPrelude (p : Parameters) : List ComputeStmt :=
  (TypedSource.kernel .fp16 "A" "B" "C" p.problem.M p.problem.N p.problem.K
    p.am p.ak p.bk p.bn p.cm p.cn 128 256 p.problem.profile.blockK 8 p.problem.SMS).surfaceBody.take 15

theorem prelude_raw (p : Parameters) : prelude p=rawPrelude p := by
  unfold prelude source rawPrelude
  cases p.problem.profile <;> rfl

def rawTilePrelude (p : Parameters) : List ComputeStmt :=
  let kernel := TypedSource.kernel .fp16 "A" "B" "C" p.problem.M p.problem.N p.problem.K
    p.am p.ak p.bk p.bn p.cm p.cn 128 256 p.problem.profile.blockK 8 p.problem.SMS
  (match kernel.surfaceBody[15]? with
   | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
   | _ => []).take 13

theorem tilePrelude_raw (p : Parameters) : tilePrelude p=rawTilePrelude p := by
  unfold tilePrelude outer source rawTilePrelude
  cases p.problem.profile <;> rfl

def ceil (n d : Nat) : Nat := n/d + if n%d=0 then 0 else 1

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
