import bench.tritonbench_g.bgmv_expand_slice.Proofs.Support.SurfaceSource
import VeriTile.Triton.Float.StructuralSafety
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Program
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
/-- Proof-only blocks extracted from the complete canonical quotation. The
constant CAST_TYPE branch is retained in `active`; the two interpretation
lemmas below justify using its selected body in the loop decomposition. -/
def code (h : Call) : List ComputeStmt := (Surface.source h).surfaceBody
def prelude (h : Call) := (code h).take 3
def active (h : Call) : List ComputeStmt :=
  match (code h)[3]? with | some (ComputeStmt.ifThen _ yes) => yes | _ => []
def beforeCast (h : Call) := (active h).take 4
def castYes (h : Call) : List ComputeStmt :=
  match (active h)[4]? with | some (ComputeStmt.ifThenElse _ yes _) => yes | _ => []
def castNo (h : Call) : List ComputeStmt :=
  match (active h)[4]? with | some (ComputeStmt.ifThenElse _ _ no) => no | _ => []
def castSetup (h : Call) := ((Source.intRange? (castYes h) "n").map (·.before)).getD []
def setup (h : Call) := beforeCast h ++ castSetup h
def body (h : Call) := ((Source.intRange? (castYes h) "n").map (·.body)).getD []
def guard : Op .bool [] := .boolNot (.eq .int .nil (.ref .int [] "lora_index") (.constInt (-1)))
def loop (h : Call) : ComputeStmt :=
  .forRangeInt "n" (.castNatToInt (.constNat 0)) (.ref .int [] "split_n_length") (.constNat 256) (body h)

theorem complete_code (h : Call) : code h=prelude h++[.ifThen guard (active h)] := by rfl

theorem complete_active (h : Call) : active h=beforeCast h++
    [.ifThenElse (.constBool Bool.true) (castYes h) (castNo h)] := by rfl

theorem complete_cast (h : Call) : castYes h=castSetup h++[loop h] := by rfl

private theorem run_true {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (yes no : List ComputeStmt) (s : FP.Structural.State α) :
    FP.Structural.run M [.ifThenElse (.constBool Bool.true) yes no] s = FP.Structural.run M yes s := by
  simp [FP.Structural.run,FP.Structural.step,FP.Structural.evalOp_unfold,Option.bind_fun_some]

private theorem safe_true {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (yes no : List ComputeStmt) (bounds : RegionName → Nat) (s : FP.Structural.State α) :
    FP.Structural.Safety.traceSafeList M bounds [.ifThenElse (.constBool Bool.true) yes no] s ↔
      FP.Structural.Safety.traceSafeList M bounds yes s := by
  rw [FP.Structural.Safety.traceSafeList_cons]
  simp only [FP.Structural.Safety.traceSafeList_nil,implies_true,forall_const,and_true]
  simp [FP.Structural.Safety.stepSafe_unfold,FP.Structural.Safety.safeOp_unfold,
    FP.Structural.evalOp_unfold]

theorem active_run_decomposition {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (h : Call) (s : FP.Structural.State α) :
    FP.Structural.run M (active h) s = FP.Structural.run M (setup h++[loop h]) s := by
  rw [setup,List.append_assoc,←complete_cast,complete_active,FP.Structural.run_append,FP.Structural.run_append]
  simp only [run_true]

theorem active_safety_decomposition {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (h : Call) (bounds : RegionName → Nat) (s : FP.Structural.State α) :
    FP.Structural.Safety.traceSafeList M bounds (active h) s ↔
      FP.Structural.Safety.traceSafeList M bounds (setup h++[loop h]) s := by
  rw [setup,List.append_assoc,←complete_cast,complete_active,FP.Structural.Safety.traceSafeList_append,
    FP.Structural.Safety.traceSafeList_append]
  simp only [safe_true]

theorem signed_length (h : Call) (positive : 0<h.N) :
    ((h.N : Int)+h.N-1)/h.N=1 := by
  have hs : (h.N : Int)+h.N-1=((h.N+h.N-1 : Nat) : Int) := by omega
  rw [hs,←Int.natCast_ediv,Kernels.split_length h positive]
  rfl
#axiomsClean complete_code
#axiomsClean complete_active
#axiomsClean active_run_decomposition
#axiomsClean active_safety_decomposition
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Program
