import bench.tritonbench_g.rms_matmul_rbe.IO
import VeriTile.Triton.Float.ArrayProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Pair
open VeriTile Triton FP.Structural
noncomputable section

def code (fp8 : Bool) (p : Fin 3 → Arguments) : List ComputeStmt :=
  (TypedSource.qkvIndexCommuted fp8 {(p 0) with BM:=16,BN:=64,BK:=64}
    {(p 1) with BM:=16,BN:=64,BK:=64} {(p 2) with BM:=16,BN:=64,BK:=64}).surfaceBody

def job {α : Type} (a : Arithmetic α) (fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (pid : Nat → Nat) : Invocation α :=
  { IO.job a Bool.false fp8 p epsilon theta pid with
    kernel := .mk ((List.finRange 3).flatMap fun phase=>[(p phase).X,(p phase).W,(p phase).RMS])
      ((List.finRange 3).map fun phase=>(p phase).OUT) (code fp8 p) }

def variant {α : Type} (a : Arithmetic α) (fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat)) :
    ArrayProgramIO.Variant (IO.qkvIO a Bool.false fp8 p epsilon theta order) where
  schedule := fun _=>order.map (job a fp8 p epsilon theta)
  observe := (IO.qkvIO a Bool.false fp8 p epsilon theta order).observe

def original {α : Type} (a : Arithmetic α) (fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat)) :=
  ArrayProgramIO.Implementation.original (IO.qkvIO a Bool.false fp8 p epsilon theta order)
    (IO.InputDomain fp8 p)

def indexCommuted {α : Type} (a : Arithmetic α) (fp8 : Bool) (p : Fin 3 → Arguments)
    (epsilon theta : α) (order : List (Nat → Nat)) :=
  ArrayProgramIO.Implementation.ofVariant (variant a fp8 p epsilon theta order)
    (IO.InputDomain fp8 p)

end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Pair
