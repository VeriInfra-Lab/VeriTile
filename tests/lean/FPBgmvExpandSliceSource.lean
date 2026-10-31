import bench.tritonbench_g.bgmv_expand_slice.Kernels
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Precision and control regressions for the actual repaired source and its
output-first candidate. The wrapper selects FP32 X/Y, FP16 W, CAST_TYPE=true,
BLOCK_N=256 and SPLIT_N=N. No historical mixed-profile program is constructed. -/
namespace FPBgmvExpandSliceSourceTests
open VeriTile.Triton
open VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000

private def source (h : Call) (candidate : Bool) :=
  if candidate then Kernels.outputFirst h else Kernels.source h
private def activeBody (h : Call) (candidate : Bool) :=
  ((Source.conditional? (source h candidate).surfaceBody
    (fun body => (Source.intRange? body "n").isSome)).map (·.yes)).getD []
private def loop (h : Call) (candidate : Bool) :=
  Source.intRange? (activeBody h candidate) "n"
private def loopBody (h : Call) (candidate : Bool) := ((loop h candidate).map (·.body)).getD []
private def firstConditional (code : List ComputeStmt) :=
  Source.conditional? code (fun _ => Bool.true)
private def branch (code : List ComputeStmt) (yes : Bool) :=
  ((firstConditional code).map (fun c => if yes then c.yes else c.no)).getD []
private def assignment (code : List ComputeStmt) (name : RegName) :=
  (code.drop (Source.prefixBefore code name).length).head?
private def loadDType? : ComputeStmt → Option TileDType
  | .assign _ _ _ (.alg (.load dtype _ _)) => some dtype
  | .assign _ _ _ (.compute (.load .fp32 _ _)) => some .fp32
  | _ => none
private def assignedDType? : ComputeStmt → Option TileDType
  | .assign dtype _ _ _ => some dtype
  | _ => none
private def assignedLoad (code : List ComputeStmt) (name : RegName) :=
  (assignment code name).bind loadDType?

/-- Both EVEN_K arms load FP32 input, before the required conversion to half. -/
theorem input_load_keeps_storage (h : Call) (candidate yes : Bool) :
    assignedLoad (branch (activeBody h candidate) yes) "tiled_a" = some .fp32 := by
  cases candidate <;> cases yes <;> rfl

theorem weight_load_keeps_storage (h : Call) (candidate : Bool) :
    assignedLoad (loopBody h candidate) "tiled_b" = some .fp16 := by
  cases candidate <;> rfl

theorem old_output_load_keeps_storage (h : Call) (candidate : Bool) :
    assignedLoad (branch (loopBody h candidate) Bool.true) "tiled_out" = some .fp32 := by
  cases candidate <;> rfl

/-- The cast is an actual top-level reassignment, outside the input-load branch. -/
theorem input_cast_rebinds_to_half (h : Call) (candidate : Bool) :
    (assignment (activeBody h candidate) "tiled_a").bind assignedDType? = some .fp16 := by
  cases candidate <;> rfl

theorem loop_keeps_signed_bounds (h : Call) (candidate : Bool) :
    (loop h candidate).map (fun r => (r.start,r.stop,r.stride)) =
      some (.constInt 0,.ref .int [] "split_n_length",.constNat 256) := by
  cases candidate <;> rfl

/-- Early return is lowered by guarding the complete remaining body with the
negated signed sentinel test. No arithmetic or store precedes that guard. -/
theorem signed_early_return (h : Call) (candidate : Bool) :
    (Source.conditional? (source h candidate).surfaceBody
      (fun body => (Source.intRange? body "n").isSome)).map (fun c => c.condition) =
        some (.boolNot (.eq .int .nil (.ref .int [] "lora_index") (.constInt (-1)))) := by
  cases candidate <;> rfl

theorem signed_metadata (h : Call) (candidate : Bool) :
    (assignment (source h candidate).surfaceBody "pid_sn").bind assignedDType? = some .int ∧
    (assignment (source h candidate).surfaceBody "cur_batch").bind assignedDType? = some .int ∧
    (assignment (source h candidate).surfaceBody "lora_index").bind assignedDType? = some .int := by
  cases candidate <;> exact ⟨rfl,rfl,rfl⟩

private def halfSum (h : Call) : Op .fp16 [256] :=
  .reduceSumTyped .fp16 ⟨1,by simp⟩ Bool.false
    (.mul .fp16 (.leadL (.consSame .nil)) (.ref .fp16 [h.BK] "tiled_a")
      (.ref .fp16 [256,h.BK] "tiled_b"))

/-- FP16 multiplication/reduction precedes promotion and the optional FP32
addition; the current candidate exchanges just those addition operands. -/
theorem output_promotes_after_half_sum (h : Call) (candidate : Bool) :
    assignment (branch (loopBody h candidate) Bool.true) "accumulator" =
      some (.assign .real [256] "accumulator" (.compute (.alg .fp32 (.castFloat .real .real
        (if candidate then
          .add .real (.consSame .nil) (.ref .real [256] "tiled_out") (.castFloat .fp16 .real (halfSum h))
        else .add .real (.consSame .nil) (.castFloat .fp16 .real (halfSum h)) (.ref .real [256] "tiled_out")))))) := by
  cases candidate <;> rfl

/-- The active output lane test survives the implicit destination conversion. -/
theorem masked_output_store (h : Call) (candidate : Bool) :
    (loopBody h candidate).getLast? = some (.store .real [256]
      (.ptr (.ptrAdd .scalarL (.ref .ptr [] "c_ptr") (.castIntToNat (.ref .int [256] "current_n"))))
      (.compute (.alg .fp32 (.ref .real [256] "accumulator")))
      (.mask (.ref .bool [256] "c_mask"))) := by
  cases candidate <;> rfl

theorem complete_io_signature (h : Call) (candidate : Bool) :
    (source h candidate).inputs = ["X","W","Y","Indices"] ∧
    (source h candidate).outputs = ["Y"] := by
  cases candidate <;> exact ⟨rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end FPBgmvExpandSliceSourceTests
