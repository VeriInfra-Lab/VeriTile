import bench.tritonbench_g.chunk_cumsum_kernel.Kernels
import VeriTile.Meta.StatementAudit

/-! Precision and signed-control regressions on the actual repair and its
current sum-left candidate, before Real erasure. -/
namespace FPChunkCumsumSourceTests
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkCumsumKernel
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

def source (swapped : Bool) (dtype out_dtype : FloatDType) (T BT : Nat) : ComputeKernel :=
  if swapped then sumLeftKernel dtype out_dtype "s" "o" T BT else repairedKernel dtype out_dtype "s" "o" T BT

def loop (swapped : Bool) (dtype out_dtype : FloatDType) (T BT : Nat) : Source.IntRange :=
  (Source.intRange? (source swapped dtype out_dtype T BT).surfaceBody "i_t").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 0,[],[]⟩

private def active (T BT : Nat) : Op .bool [BT] :=
  .lt .int .scalarR (.ref .int [BT] "times") (.castNatToInt (.constNat T))

theorem load_explicitly_promotes (swapped : Bool) (dtype out_dtype : FloatDType) (T BT : Nat) :
    (loop swapped dtype out_dtype T BT).body[3]? = some (.assign .real [BT] "b_s"
      (.compute (.alg .fp32 (.castFloat dtype .real
        (.load dtype.toTileDType (.ptr (.ref .ptr [BT] "p_s")) (.mask (active T BT)) .zero))))) := by
  cases swapped <;> rfl

theorem scan_keeps_input_axis_and_direction (swapped : Bool) (dtype out_dtype : FloatDType) (T BT : Nat) :
    (loop swapped dtype out_dtype T BT).body[4]? = some (.assign .real [BT] "b_o"
      (.compute (.alg .fp32 (.add .real Broadcast.nil.consR
        (.scan .sum ⟨0,by simp⟩ .forward (.ref .real [BT] "b_s"))
        (.expandDim ⟨0,by simp⟩ (.ref .real [] "b_z")))))) := by
  cases swapped <;> rfl

theorem total_is_fp32 (swapped : Bool) (dtype out_dtype : FloatDType) (T BT : Nat) :
    (loop swapped dtype out_dtype T BT).body[5]? = some (.assign .real [] "b_zz"
      (.compute (.alg .fp32 (.reduceSum ⟨0,by simp⟩ Bool.false (.ref .real [BT] "b_s"))))) := by
  cases swapped <;> rfl

theorem output_cast_and_mask_retained (swapped : Bool) (dtype out_dtype : FloatDType) (T BT : Nat) :
    (loop swapped dtype out_dtype T BT).body[7]? = some (.store out_dtype.toTileDType [BT]
      (.ptr (.ref .ptr [BT] "p_o"))
      (.alg (.castFloat .real out_dtype (.withPrecision .real .fp32 (.ref .real [BT] "b_o"))))
      (.mask (active T BT))) := by
  cases swapped <;> rfl

theorem signed_range_retained (swapped : Bool) (dtype out_dtype : FloatDType) (T BT : Nat) :
    ((loop swapped dtype out_dtype T BT).start,(loop swapped dtype out_dtype T BT).stop,
      (loop swapped dtype out_dtype T BT).stride)=
      (.constInt 0,.constInt ((T+BT-1)/BT:Nat),.constNat 1) := by
  cases swapped <;> rfl

theorem signed_times_retained (swapped : Bool) (dtype out_dtype : FloatDType) (T BT : Nat) :
    (loop swapped dtype out_dtype T BT).body[0]? = some (.assign .int [BT] "times" (.alg
      (.add .int .scalarL (.mul .int .nil (.ref .int [] "i_t") (.constInt (BT : Int)))
        (.castNatToInt (.arange BT))))) := by
  cases swapped <;> rfl

theorem candidate_keeps_fp32_carry_operands (swapped : Bool) (dtype out_dtype : FloatDType) (T BT : Nat) :
    (loop swapped dtype out_dtype T BT).body[6]? = some (.assign .real [] "b_z" (.compute (.alg .fp32
      (if swapped then .add .real .nil (.ref .real [] "b_zz") (.ref .real [] "b_z")
        else .add .real .nil (.ref .real [] "b_z") (.ref .real [] "b_zz"))))) := by
  cases swapped <;> rfl

theorem full_signature_retained (swapped : Bool) (dtype out_dtype : FloatDType) (T BT : Nat) :
    (source swapped dtype out_dtype T BT).inputs=["s"] ∧
    (source swapped dtype out_dtype T BT).outputs=["o"] := by
  cases swapped <;> exact ⟨rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end FPChunkCumsumSourceTests
