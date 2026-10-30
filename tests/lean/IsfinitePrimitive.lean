import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_BitClassification
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Primitive regression draft, requiring shared patch 01. This executes the
new classifier on actual memory loads and arbitrary algebra callbacks. It is
not a claim that the complete pinned source or native libdevice is verified. -/
namespace VeriTile.Tests.IsfinitePrimitive
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.BitClassification

theorem classification_success {α : Type} [Inhabited α] (M : Algebra α)
    (p : FinitePrecision) (e : ComputeExpr .real shape) (s : State α)
    (values : Values α .real shape) (classify : α → Bool)
    (hv : evalExpr M e s = some values) (hc : M.isFinite p = some classify) :
    evalExpr M (.isFinite p e) s = some (fun i => classify (values i)) := by
  simp only [evalExpr, hv, hc, bind, Option.bind, Option.pure_def]

theorem missing_backend_fails {α : Type} [Inhabited α] (M : Algebra α)
    (p : FinitePrecision) (e : ComputeExpr .real shape) (s : State α)
    (h : M.isFinite p = none) : evalExpr M (.isFinite p e) s = none := by
  cases hv : evalExpr M e s <;>
    simp only [evalExpr, hv, h, bind, Option.bind, Option.pure_def]

theorem failed_operand_fails {α : Type} [Inhabited α] (M : Algebra α)
    (p : FinitePrecision) (e : ComputeExpr .real shape) (s : State α)
    (h : evalExpr M e s = none) : evalExpr M (.isFinite p e) s = none := by
  simp only [evalExpr, h, bind, Option.bind]

/-- The explicit classifier tag is not overwritten by the default FP profile. -/
theorem profile_retains_classifier {α : Type} (M : Algebra α) (default : ComputeDType)
    (p : FinitePrecision) : (M.withDefaultPrecision default).isFinite p = M.isFinite p := rfl

def profiledLoad (p : FinitePrecision) (r : RegionName) (n : Nat) :
    ComputeExpr .real [n] := match p with
  | .fp32 => .compute (.alg .fp32 (.load .real (.region r (.arange n)) .none))
  | .fp64 => .compute (.alg .fp64 (.load .real (.region r (.arange n)) .none))

theorem profiledLoad_success {α : Type} [Inhabited α] (M : Algebra α)
    (p : FinitePrecision) (r : RegionName) (n : Nat) (s : State α) :
    evalExpr M (profiledLoad p r n) s =
      some (fun i => (s.mem r i.1.val).read .real) := by
  cases p <;>
    simp only [profiledLoad, evalExpr, evalComputeOp, evalOp_unfold,
      bind, Option.bind, Option.map, Option.pure_def, Region.cast_id,
      ↓reduceIte] <;> rfl

theorem classify_memory {α : Type} [Inhabited α] (M : Algebra α)
    (p : FinitePrecision) (r : RegionName) (n : Nat) (s : State α)
    (classify : α → Bool) (hc : M.isFinite p = some classify) :
    evalExpr M (.isFinite p (profiledLoad p r n)) s =
      some (fun i => classify ((s.mem r i.1.val).read .real)) :=
  classification_success M p _ s _ classify (profiledLoad_success M p r n s) hc

/-- Projection of a program containing classification rejects the operation. -/
theorem projection_rejects (p : FinitePrecision) (e : ComputeExpr .real shape) :
    (ComputeKernel.mk [] [] [.assign .bool shape "finite" (.isFinite p e)]).toAlgorithm? =
      .error (.requiresComputeSemantics "libdevice finite classification") := rfl

theorem erasure_retains_rejected_program (p : FinitePrecision)
    (e : ComputeExpr .real shape) :
    (ComputeKernel.mk [] [] [.assign .bool shape "finite" (.isFinite p e)]).eraseDType =
      ComputeKernel.mk [] [] [.assign .bool shape "finite" (.isFinite p e)] := by
  apply ComputeKernel.eraseDType_of_projection_error
  exact projection_rejects p e

local instance : Inhabited Float32Bits := ⟨bits32 0⟩
local instance : Inhabited Float64Bits := ⟨bits64 0⟩

def sample32 (n : Nat) : Float32Bits := bits32 (match n with
  | 0 => 0x80000000 -- negative zero
  | 1 => 0x00000001 -- smallest positive subnormal
  | 2 => 0xff800000 -- negative infinity
  | _ => 0x7f800001) -- signaling NaN

def sample64 (n : Nat) : Float64Bits := bits64 (match n with
  | 0 => 0x8000000000000000
  | 1 => 0x0000000000000001
  | 2 => 0xfff0000000000000
  | _ => 0x7ff0000000000001)

def initial32 : State Float32Bits where
  mem _ n := .mk .real (sample32 n)
  regs _ _ _ := none
  pids _ := 0
  numPids _ := 1
  undef d _ _ := defaultValue d

def initial64 : State Float64Bits where
  mem _ n := .mk .real (sample64 n)
  regs _ _ _ := none
  pids _ := 0
  numPids _ := 1
  undef d _ _ := defaultValue d

def fourResults (v : TileIndex [4] → Bool) : List Bool :=
  [v (0, PUnit.unit), v (1, PUnit.unit), v (2, PUnit.unit), v (3, PUnit.unit)]

/-- All other algebra operations are arbitrary and are never invoked here.
The payload carrier and callback hypothesis rule out precision-tag mismatch. -/
theorem mixed_fp32_memory (M : Algebra Float32Bits)
    (h : M.isFinite .fp32 = some finite32) :
    (evalExpr M (.isFinite .fp32 (profiledLoad .fp32 "input" 4)) initial32).map
      fourResults = some [true, true, false, false] := by
  rw [classify_memory M .fp32 "input" 4 initial32 finite32 h]
  rfl

theorem mixed_fp64_memory (M : Algebra Float64Bits)
    (h : M.isFinite .fp64 = some finite64) :
    (evalExpr M (.isFinite .fp64 (profiledLoad .fp64 "input" 4)) initial64).map
      fourResults = some [true, true, false, false] := by
  rw [classify_memory M .fp64 "input" 4 initial64 finite64 h]
  rfl

theorem mixed_fp64_under_fp32_profile (M : Algebra Float64Bits)
    (h : M.isFinite .fp64 = some finite64) :
    (evalExpr (M.withDefaultPrecision .fp32)
      (.isFinite .fp64 (profiledLoad .fp64 "input" 4)) initial64).map
      fourResults = some [true, true, false, false] :=
  mixed_fp64_memory (M.withDefaultPrecision .fp32) h

theorem empty_tile_missing_backend (M : Algebra Float32Bits)
    (h : M.isFinite .fp32 = none) :
    evalExpr M (.isFinite .fp32 (profiledLoad .fp32 "input" 0)) initial32 = none :=
  missing_backend_fails M .fp32 _ initial32 h

#axiomsClean classification_success
#axiomsClean missing_backend_fails
#axiomsClean failed_operand_fails
#axiomsClean profile_retains_classifier
#axiomsClean profiledLoad_success
#axiomsClean classify_memory
#axiomsClean projection_rejects
#axiomsClean erasure_retains_rejected_program
#axiomsClean mixed_fp32_memory
#axiomsClean mixed_fp64_memory
#axiomsClean mixed_fp64_under_fp32_profile
#axiomsClean empty_tile_missing_backend
end VeriTile.Tests.IsfinitePrimitive
