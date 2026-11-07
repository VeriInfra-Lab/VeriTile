import bench.tritonbench_g.attn_fwd_triton.Proofs.RegisterViews
import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedPrefix
import bench.tritonbench_g.attn_fwd_triton.Proofs.AttnFwdTriton_Verified_Loop_Loop

/-! The mathematical loop body is a register view of each complete pasted
helper invocation. Capture registers and local scopes are obtained from the
source frontend; no alternative kernel is constructed here. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HelperRegisters
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport
open RegisterViews
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

abbrev workName := AttnFwdTriton.TypedExecution.workName

def parts (BM BN BD : Nat) (p : Parameters) (second : Bool) : Source.IntRange :=
  (Source.intRange? (source BM BN BD p Bool.false).surfaceBody (workName second "start_n")).getD
    ⟨[],.constInt 0,.constInt 0,.constNat 0,[],[]⟩

def body (BM BN BD : Nat) (p : Parameters) (second : Bool) : List Stmt :=
  Stmt.eraseDTypeList (ComputeKernel.mk [] [] (parts BM BN BD p second).body).toAlgKernel.body

def namePairs (second : Bool) : List (RegName × RegName) :=
  [
    ("acc",workName second "acc"),
    ("l_i",workName second "l_i"),
    ("m_i",workName second "m_i"),
    ("q",workName second "q"),
    ("q_scale",workName second "q_scale"),
    ("start_m",workName second "start_m"),
    ("stride_kn64",workName second "stride_kn"),
    ("stride_vk64",workName second "stride_vk"),
    ("offs_m",workName second "offs_m"),
    ("offs_n",workName second "offs_n"),
    ("n_ctx",workName second "N_CTX"),
    ("start_n",workName second "start_n"),
    ("k_mask",workName second "k_mask"),
    ("k",workName second "k"),
    ("k_scale",workName second "k_scale"),
    ("qk",workName second "qk"),
    ("m_ij",workName second "m_ij"),
    ("p",workName second "p"),
    ("l_ij",workName second "l_ij"),
    ("alpha",workName second "alpha"),
    ("v",workName second "v"),
    (keyName second,workName second "K_ptrs"),
    (scaleName second,workName second "K_scale_ptr"),
    (valueName second,workName second "V_ptrs"),
    (maskName second,workName second "mask")]

/-- A permutation also defines the view on unrelated registers, so updates
commute without a hidden freshness premise. -/
def names (second : Bool) : RegName ≃ RegName :=
  (namePairs second).foldr (fun pair rest=>(Equiv.swap pair.1 pair.2).trans rest) (Equiv.refl _)

@[simp] theorem names_acc (second : Bool) :
    names second "acc"=workName second "acc" := by
  cases second <;> decide

@[simp] theorem names_l_i (second : Bool) :
    names second "l_i"=workName second "l_i" := by
  cases second <;> decide

@[simp] theorem names_m_i (second : Bool) :
    names second "m_i"=workName second "m_i" := by
  cases second <;> decide

@[simp] theorem names_q (second : Bool) :
    names second "q"=workName second "q" := by
  cases second <;> decide

@[simp] theorem names_q_scale (second : Bool) :
    names second "q_scale"=workName second "q_scale" := by
  cases second <;> decide

@[simp] theorem names_start_m (second : Bool) :
    names second "start_m"=workName second "start_m" := by
  cases second <;> decide

@[simp] theorem names_stride_kn64 (second : Bool) :
    names second "stride_kn64"=workName second "stride_kn" := by
  cases second <;> decide

@[simp] theorem names_stride_vk64 (second : Bool) :
    names second "stride_vk64"=workName second "stride_vk" := by
  cases second <;> decide

@[simp] theorem names_offs_m (second : Bool) :
    names second "offs_m"=workName second "offs_m" := by
  cases second <;> decide

@[simp] theorem names_offs_n (second : Bool) :
    names second "offs_n"=workName second "offs_n" := by
  cases second <;> decide

@[simp] theorem names_n_ctx (second : Bool) :
    names second "n_ctx"=workName second "N_CTX" := by
  cases second <;> decide

@[simp] theorem names_start_n (second : Bool) :
    names second "start_n"=workName second "start_n" := by
  cases second <;> decide

@[simp] theorem names_k_mask (second : Bool) :
    names second "k_mask"=workName second "k_mask" := by
  cases second <;> decide

@[simp] theorem names_k (second : Bool) :
    names second "k"=workName second "k" := by
  cases second <;> decide

@[simp] theorem names_k_scale (second : Bool) :
    names second "k_scale"=workName second "k_scale" := by
  cases second <;> decide

@[simp] theorem names_qk (second : Bool) :
    names second "qk"=workName second "qk" := by
  cases second <;> decide

@[simp] theorem names_m_ij (second : Bool) :
    names second "m_ij"=workName second "m_ij" := by
  cases second <;> decide

@[simp] theorem names_p (second : Bool) :
    names second "p"=workName second "p" := by
  cases second <;> decide

@[simp] theorem names_l_ij (second : Bool) :
    names second "l_ij"=workName second "l_ij" := by
  cases second <;> decide

@[simp] theorem names_alpha (second : Bool) :
    names second "alpha"=workName second "alpha" := by
  cases second <;> decide

@[simp] theorem names_v (second : Bool) :
    names second "v"=workName second "v" := by
  cases second <;> decide

@[simp] theorem names_key (second : Bool) :
    names second (keyName second)=workName second "K_ptrs" := by
  cases second <;> decide

@[simp] theorem names_scale (second : Bool) :
    names second (scaleName second)=workName second "K_scale_ptr" := by
  cases second <;> decide

@[simp] theorem names_value (second : Bool) :
    names second (valueName second)=workName second "V_ptrs" := by
  cases second <;> decide

@[simp] theorem names_mask (second : Bool) :
    names second (maskName second)=workName second "mask" := by
  cases second <;> decide

@[simp] theorem names_false_K_ptrs_first :
    names Bool.false "K_ptrs_first"=workName Bool.false "K_ptrs" := by decide

@[simp] theorem names_false_K_scale_ptr_first :
    names Bool.false "K_scale_ptr_first"=workName Bool.false "K_scale_ptr" := by decide

@[simp] theorem names_false_V_ptrs_first :
    names Bool.false "V_ptrs_first"=workName Bool.false "V_ptrs" := by decide

@[simp] theorem names_false_mask_first :
    names Bool.false "mask_first"=workName Bool.false "mask" := by decide

@[simp] theorem names_true_K_ptrs_second :
    names Bool.true "K_ptrs_second"=workName Bool.true "K_ptrs" := by decide

@[simp] theorem names_true_K_scale_ptr_second :
    names Bool.true "K_scale_ptr_second"=workName Bool.true "K_scale_ptr" := by decide

@[simp] theorem names_true_V_ptrs_second :
    names Bool.true "V_ptrs_second"=workName Bool.true "V_ptrs" := by decide

@[simp] theorem names_true_mask :
    names Bool.true "mask"=workName Bool.true "mask" := by decide

private theorem assign_named (permutation : RegName ≃ RegName)
    (name actualName : RegName) {d : TileDType} {shape : TileShape}
    (view actual : Op d shape) (named : permutation name=actualName)
    (expression : ∀ s,evalOp actual s=evalOp view (pull permutation s))
    (before after : List Stmt) (rest : Corresponds permutation before after) :
    Corresponds permutation (.assign d shape name view::before)
      (.assign d shape actualName actual::after) := by
  rw [←named]
  exact RegisterViews.assign permutation name view actual expression before after rest

theorem iteration_corresponds (BM BN BD : Nat) (p : Parameters) (second : Bool) :
    Corresponds (names second)
      (sourceIteration BM BN BD p.headDim second (decide (p.stage=2)))
      (body BM BN BD p second) := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [body,parts,source,TypedSource.repaired,originalKernel,
    hd,decide_true,decide_false,workName,AttnFwdTriton.TypedExecution.workName,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody]
  all_goals simp [Source.intRange?,workName,AttnFwdTriton.TypedExecution.workName]
  all_goals try dsimp only [ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals try simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,FloatDType.eraseFloat,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals simp only [sourceIteration,flatIteration,loadPrefix,guard,suffix,
    valueLoad,advance,RealBody.score,RealBody.scoreExpr,RealBody.center,
    RealBody.centerExpr,RealBody.probability,RealBody.rescale,RealBody.accumulate,
    Bool.false_or,Bool.true_or,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    List.nil_append,List.cons_append]
  all_goals repeat' first | apply assign_named | exact RegisterViews.nil _
  all_goals try simp [
    workName,AttnFwdTriton.TypedExecution.workName,keyName,scaleName,valueName,maskName,
    RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName,RealPreparation.maskName]
  all_goals try intro s
  all_goals simp [evalOp.eq_def,pull,
    workName,AttnFwdTriton.TypedExecution.workName,keyName,scaleName,valueName,maskName,
    RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName,RealPreparation.maskName,
    maskExpr,tail,keyMask,valueMask,wide,BlockState.readMem]
  all_goals norm_num

#axiomsClean iteration_corresponds
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HelperRegisters
