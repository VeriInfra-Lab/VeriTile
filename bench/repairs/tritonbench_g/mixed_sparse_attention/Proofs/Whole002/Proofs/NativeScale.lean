import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.NativeReturned
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Native
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- The exact parsed qk_scale assignment: one FP32 multiply with the unique
literal coefficient, before the query load and query narrowing. -/
theorem scale_source_node (c : Config) :
    (activeBody c)[16]? = some (ComputeStmt.assign .real [] "qk_scale"
      (.compute (.alg .fp32 (.mul .real .nil (.full [] (.const c.scale)) (.const 1.44269504))))) := by
  cases hs : c.storage <;>
    simp [activeBody,parsedActiveBody,parsedKernel,TypedSource.FP16.source,TypedSource.BF16.source,
      ComputeKernel.surfaceBody,hs]

/-- Native tensor inputs are actual FP32 words, not arbitrary staged syntax. -/
structure InputWords (c : Config) (before : Physical.Store Expr) : Prop where
  allocated : ∀ r∈[c.q,c.k,c.v],∀ i,∃ word : BitVec 32,
    before.cells r i=.mk .real (fun _=>Stage.ready (.bits word))
  outside : ∃ word : BitVec 32,before.outside=.mk .real (fun _=>Stage.ready (.bits word))

theorem input_ready (c : Config) (before : Physical.Store Expr) (h : InputWords c before)
    (r : RegionName) (hr : r∈[c.q,c.k,c.v]) (off : Nat) :
    ∃ word : BitVec 32,(before.extend.mem r off).read .real=fun _=>Stage.ready (.bits word) := by
  by_cases hi : off<before.capacity r
  · obtain ⟨word,hw⟩:=h.allocated r hr ⟨off,hi⟩
    exact ⟨word,by simp [Physical.Store.extend,hi,hw]⟩
  · obtain ⟨word,hw⟩:=h.outside
    exact ⟨word,by simp [Physical.Store.extend,hi,hw]⟩

theorem ready_not_scale (word : BitVec 32) :
    scaleRhs (.ready (.bits word))=Bool.false := rfl

theorem narrow_not_scale (dtype : TypedSource.Storage) (a : Stage) :
    scaleRhs (convert .real dtype.float a)=Bool.false := by
  cases dtype <;> cases a <;> rfl

/-- Exact ABI word staging at the qk_scale operation. No recovery from a Real
value, signed-zero identification, or final-result premise is involved. -/
theorem scale_word (runtime : Runtime) (scale : ℝ) (u : Fin 4) :
    lower ((algebra runtime).binary (some .fp32) .real .mul
      ((algebra runtime).literal (some .fp32) .real scale)
      ((algebra runtime).literal (some .fp32) .real 1.44269504) u)=
      .binary .mul32 (.bits runtime.scaleBits) (.literal32 1.44269504) := by
  norm_num [algebra,binaryFor,scaleRhs,literal,lower]

/-- The outer query multiplication consumes the constructed scale stage,
so it cannot be mistaken for the runtime-scale binding site. -/
theorem query_scale_word (runtime : Runtime) (c : Config) (s : State Expr)
    (i : TileIndex [c.BM,c.BD]) (u : Fin 4) :
    lower (queryValues (algebra runtime) c s i u)=
      .unary (match c.storage with | .fp16=>.narrowHalf | .bf16=>.narrowBF16)
        (.binary .mul32
          (.binary .mul32 (.bits runtime.scaleBits) (.literal32 1.44269504))
          (lower ((if rowStart c s+i.1.val<sequenceLength c s then
            (s.mem c.q (queryBase c s+(rowStart c s+i.1.val)*c.sqm+i.2.1.val*c.sqk)).read .real
           else fun _=>Stage.zero) u))) := by
  cases hs : c.storage <;>
    norm_num [queryValues,algebra,binaryFor,scaleRhs,literal,binary,convert,lower,hs,
      TypedSource.Storage.float]

/-- These are every remaining right-multiply form in the source recurrence:
+0 in acc_scale, exp2(alpha), and the constructed acc_scale sum. -/
theorem zero_multiply (runtime : Runtime) (a : Expr) (u : Fin 4) :
    (algebra runtime).binary (some .fp32) .real .mul a
      ((algebra runtime).literal (some .fp32) .real 0) u=.mul (a u) .zero := by
  simp [algebra,literal,binaryFor,scaleRhs,binary]

theorem exp_multiply (runtime : Runtime) (a b : Expr) (u : Fin 4) :
    (algebra runtime).binary (some .fp32) .real .mul a
      ((algebra runtime).unary (some .fp32) .exp2 b) u=
      .mul (a u) (.ready (.unary .exp2ApproxFTZ (lower (b u)))) := by
  simp [algebra,binaryFor,scaleRhs,binary]

theorem sum_multiply (runtime : Runtime) (a b d : Expr) (u : Fin 4) :
    (algebra runtime).binary (some .fp32) .real .mul a
      ((algebra runtime).binary (some .fp32) .real .add b d) u=
      .mul (a u) (.add (b u) (d u)) := by
  simp [algebra,binaryFor,scaleRhs,binary]

#axiomsClean scale_source_node
#axiomsClean input_ready
#axiomsClean narrow_not_scale
#axiomsClean scale_word
#axiomsClean query_scale_word
#axiomsClean zero_multiply
#axiomsClean exp_multiply
#axiomsClean sum_multiply
#specNonCircular InputWords avoiding [FP.Structural.exec,Source.finalValues]
#specNonCircular ScaleBinding avoiding [FP.Structural.exec,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Native
