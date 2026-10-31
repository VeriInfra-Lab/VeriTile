import bench.tritonbench_g.max_reduction.Sources.Mathematical_Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

noncomputable def mathStage1 (v : Bool) : ValueSource :=
  fun inp mid M B => (stage1 v inp mid M B).eraseDType

noncomputable def mathStage2 (v : Bool) : ValueSource :=
  fun mid out M B => (stage2 v mid out M B).eraseDType

theorem stage1_projects (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 v inp out M B).toAlgorithm? =
      .ok (stage1 v inp out M B).toAlgKernel := by
  cases v <;> rfl

theorem stage1_supported (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 v inp out M B).eraseDType.toAlgorithm? =
      .ok (stage1 v inp out M B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, stage1_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

theorem stage1_projection (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 v inp out M B).eraseDType.toAlgKernel =
      (mathStage1 v inp out M B).toAlgKernel := rfl

theorem stage2_projects (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 v inp out M B).toAlgorithm? =
      .ok (stage2 v inp out M B).toAlgKernel := by
  cases v <;> rfl

theorem stage2_supported (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 v inp out M B).eraseDType.toAlgorithm? =
      .ok (stage2 v inp out M B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, stage2_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

theorem stage2_projection (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 v inp out M B).eraseDType.toAlgKernel =
      (mathStage2 v inp out M B).toAlgKernel := rfl

abbrev DimensionSource := RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel

noncomputable def mathDimension (v : Bool) : DimensionSource :=
  fun inp ov oi M N K BM BN => (dimensionSource v inp ov oi M N K BM BN).eraseDType

theorem dimension_projects (v : Bool) (inp ov oi : RegionName)
    (M N K BM BN : Nat) :
    (dimensionSource v inp ov oi M N K BM BN).toAlgorithm? =
      .ok (dimensionSource v inp ov oi M N K BM BN).toAlgKernel := by
  cases v <;> rfl

theorem dimension_supported (v : Bool) (inp ov oi : RegionName)
    (M N K BM BN : Nat) :
    (dimensionSource v inp ov oi M N K BM BN).eraseDType.toAlgorithm? =
      .ok (dimensionSource v inp ov oi M N K BM BN).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, dimension_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

theorem dimension_projection (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource v inp ov oi M N K BM BN).eraseDType.toAlgKernel =
      (mathDimension v inp ov oi M N K BM BN).toAlgKernel := rfl

/-- The dimension wrapper stores signed int64 indices; natural index arithmetic
is explicitly converted before the masked store, never substituted as storage. -/
theorem dimension_index_store (v : Bool)
    (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource v inp ov oi M N K BM BN).surfaceBody[14]? = some
      (.store .int [BM] (.ptr (.ref .ptr [BM] "out_index_ptrs"))
        (.alg (.castNatToInt (.ref .nat [BM] "result_index")))
        (.mask (.ref .bool [BM] "mask1"))) := by
  cases v <;> rfl

/-- fp32 indexed values retain their explicit fp32 load and paired profile. -/
theorem dimension_fp32_pair (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource v inp ov oi M N K BM BN).surfaceBody[9]? = some
      (.assign .real [BM,BN] "inp_vals" (.compute (.load .fp32
        (.ptr (.ref .ptr [BM,BN] "inp_ptrs"))
        (.maskOther (.ref .bool [BM,BN] "mask") (.broadcast .negInf [BM,BN]))))) ∧
    (dimensionSource v inp ov oi M N K BM BN).surfaceBody[10]? = some
      (.ifThen (.constBool Bool.true)
        [.assign .real [BM] "result_value" (.compute (.alg .fp32 (.indexedMaxValue .real ⟨1, by simp⟩
          (.ref .real [BM,BN] "inp_vals")))),
         .assign .nat [BM] "result_index" (.alg (.argMaxTyped .real ⟨1, by simp⟩
          (.ref .real [BM,BN] "inp_vals") (some .fp32)))]) := by
  cases v <;> exact ⟨rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaxReduction
