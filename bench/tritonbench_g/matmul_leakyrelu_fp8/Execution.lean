import bench.tritonbench_g.matmul_leakyrelu_fp8.Kernels
import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Types
import bench.tritonbench_g.matmul_leakyrelu_fp8.Backend

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def source (p : Problem) (b : Buffers) := TypedSource.kernel "A" "B" "C"
  p.M p.N p.K b.A.rowStride b.A.columnStride b.B.rowStride b.B.columnStride
  b.C.rowStride b.C.columnStride p.enabled

def rows (p : Problem) := p.M/128 + if p.M%128 ≠ 0 then 1 else 0
def columns (p : Problem) := p.N/256 + if p.N%256 ≠ 0 then 1 else 0
def trips (p : Problem) := p.K/64 + if p.K%64 ≠ 0 then 1 else 0
def group (p : Problem) := min 8 (rows p)
def perGroup (p : Problem) := group p * columns p
def qDiv (x y : Int) : Int := if x < 0 then -((-x)/y) else x/y
def qMod (x y : Int) : Int := x-y*qDiv x y

def firstM {α : Type} (p : Problem) (s : State α) := qDiv (s.pids 0 : Int) (perGroup p : Int) * (group p : Int)
def groupSize {α : Type} (p : Problem) (s : State α) := min ((rows p : Int)-firstM p s) (group p : Int)
def pidM {α : Type} (p : Problem) (s : State α) := firstM p s + qMod (qMod (s.pids 0 : Int) (perGroup p : Int)) (groupSize p s)
def pidN {α : Type} (p : Problem) (s : State α) := qDiv (qMod (s.pids 0 : Int) (perGroup p : Int)) (groupSize p s)
def row {α : Type} (p : Problem) (s : State α) (i : Fin 128) := pidM p s*128+(i.val : Int)
def col {α : Type} (p : Problem) (s : State α) (j : Fin 256) := pidN p s*256+(j.val : Int)
abbrev AIndex := TileIndex [128,64]
abbrev BIndex := TileIndex [64,256]
abbrev CIndex := TileIndex [128,256]

def prelude (p : Problem) (b : Buffers) := (source p b).surfaceBody.take 20
def body (p : Problem) (b : Buffers) : List ComputeStmt :=
  match (source p b).surfaceBody[20]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []
def epilogue (p : Problem) (b : Buffers) := (source p b).surfaceBody.drop 21

theorem source_decomposition (p : Problem) (b : Buffers) :
    (source p b).surfaceBody = prelude p b ++
      [.forRangeDyn "k" (.constNat 0) (.ref .nat [] "k_tiles") (.constNat 1) (body p b)] ++ epilogue p b := rfl

theorem min_form {α : Type} [LinearOrder α] (a b : α) : min a b = if a < b then a else b := by
  by_cases h : a < b
  · rw [if_pos h,min_eq_left (le_of_lt h)]
  · rw [if_neg h,min_eq_right (le_of_not_gt h)]

structure Context {α : Type} (p : Problem) (initial current : State α) : Prop where
  memory : current.mem = initial.mem
  pm : current.regs .int [] "pid_m" = some (fun _ => pidM p initial)
  pn : current.regs .int [] "pid_n" = some (fun _ => pidN p initial)
  lm : current.regs .int [128] "lanes_m" = some (fun i => (i.1.val : Int))
  ln : current.regs .int [256] "lanes_n" = some (fun i => (i.1.val : Int))
  lk : current.regs .int [64] "lanes_k" = some (fun i => (i.1.val : Int))
  am : current.regs .int [128] "offs_am" = some (fun i => qMod (row p initial i.1) (p.M : Int))
  bn : current.regs .int [256] "offs_bn" = some (fun i => qMod (col p initial i.1) (p.N : Int))
  kt : current.regs .nat [] "k_tiles" = some (fun _ => trips p)

def HasAcc {α : Type} (a : CIndex → α) (s : State α) : Prop :=
  s.regs .real [128,256] "accumulator" = some a

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (s : State α) :
    {t : State α // run (backend M D) (prelude p b) s = some t} := by
  by_cases hm : p.M%128 = 0 <;> by_cases hn : p.N%256 = 0 <;> by_cases hk : p.K%64 = 0
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [prelude,source,TypedSource.kernel,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
      FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,hm,hn,hk]
    rfl

end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
