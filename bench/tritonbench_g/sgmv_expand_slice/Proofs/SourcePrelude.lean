import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceParts
import bench.tritonbench_g.sgmv_expand_slice.Proofs.Geometry

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourcePrelude
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

theorem common (abi : SourceABI) (p : KernelArgs) :
    SourceParts.prelude abi p = SourceParts.prelude primaryHalf p := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

theorem active_common (abi : SourceABI) (p : KernelArgs) :
    SourceParts.activePrelude abi p = SourceParts.activePrelude primaryHalf p := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

def scalarBindings {α : Type} [Inhabited α] (p : KernelArgs) (s : State α) :
    List (RegName × Int) :=
  [("pid",s.pids 0),("cur_batch",p.batch),("N",p.columns),("K",p.inner),
   ("xm_stride",p.inputRowStride),("xk_stride",p.inputInnerStride),
   ("l0_stride",p.adapterStride),("lora_k_stride",p.weightColumnStride),
   ("lora_n_stride",p.weightInnerStride),("cm_stride",p.outputRowStride),
   ("cn_stride",p.outputColumnStride),("slice_offset",p.slice),
   ("cta_n_num",tiles p.columns),("pid_m",s.pids 0 / tiles p.columns),
   ("pid_n",s.pids 0 % tiles p.columns),("M",(s.mem p.lengths p.batch).read .int)]

def prepared {α : Type} [Inhabited α] (p : KernelArgs) (s : State α) : State α :=
  (scalarBindings p s).foldl (fun t binding => t.setReg binding.1 .int [] (fun _ => binding.2)) s

theorem signed_column_tiles (columns : Nat) :
    ((columns : Int)+32-1)/32 = (tiles columns : Int) := by unfold tiles; omega

theorem run_prelude {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (p : KernelArgs) (s : State α) :
    run M (SourceParts.prelude abi p) s = some (prepared p s) := by
  rw [common]
  simp [SourceParts.prelude,SourceParts.outer,kernel,primaryHalf,sourceFace,sourceHalfSingle323232,
    ComputeKernel.surfaceBody,ScalarFormat.dtype,LowFormat.scalar,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    numeric,bop,numericLt,numericLe,numericEq,integralFloorDiv,integralMod,Region.cast,
    State.setReg,prepared,scalarBindings,signed_column_tiles]
  have hn : ¬ (p.columns : Int)+32 < 1 := by omega
  have hp : ¬ (s.pids 0 : Int) < 0 := by omega
  simp [bop,IntegerExpr.sameShape,hn,hp,←Int.emod_def]
  funext d sh name
  by_cases hd : TileDType.int = d
  · subst d
    by_cases hs : sh = []
    · subst sh
      by_cases hm : name = "M"
      · simp [hm]
      · by_cases hpid : name = "pid_n"
        · simp [hm,hpid]
          funext i
          change (s.pids 0 : Int) + 0 -
            (0 + (tiles p.columns : Int)) * ((s.pids 0 : Int) / (tiles p.columns : Int)) =
            (s.pids 0 : Int) % (tiles p.columns : Int)
          simp only [add_zero,zero_add,Int.emod_def]
        · simp [hm,hpid]
    · simp [hs]
  · simp [hd]

#axiomsClean common
#axiomsClean active_common
#axiomsClean run_prelude
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourcePrelude
