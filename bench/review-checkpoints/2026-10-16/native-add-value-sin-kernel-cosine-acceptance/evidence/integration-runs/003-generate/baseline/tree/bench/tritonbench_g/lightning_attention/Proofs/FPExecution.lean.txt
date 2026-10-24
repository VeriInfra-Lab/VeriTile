import bench.tritonbench_g.lightning_attention.Proofs.FPReference
namespace VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem fold_memory_congr {α ι : Type} (l : List ι) (output : RegionName)
    (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (v : ι → Cell α) (s t : State α) (h : s.mem=t.mem) :
    (l.foldl (fun u i => if active i then u.write output (offset i) (v i) else u) s).mem =
    (l.foldl (fun u i => if active i then u.write output (offset i) (v i) else u) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i tail ih =>
    simp only [List.foldl_cons]
    split <;> apply ih
    · simp only [State.write,h]
    · exact h

theorem scatter_memory_congr {α ι : Type} (l : List ι) (output : RegionName)
    (offset : ι → Nat) (v : ι → Cell α) (s t : State α) (h : s.mem=t.mem) :
    (l.foldl (fun u i => u.write output (offset i) (v i)) s).mem =
    (l.foldl (fun u i => u.write output (offset i) (v i)) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i tail ih =>
    simp only [List.foldl_cons]
    exact ih _ _ (by simp only [State.write,h])

theorem pair_eta (i : TileIndex [a,b]) : (i.1,i.2.1,PUnit.unit)=i := by
  rcases i with ⟨i,j,⟨⟩⟩
  rfl

theorem kv_execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    ∃ final, FP.Structural.exec (backend M D) (kvSource p) s = some final ∧
      final.mem = (written p.output (kvOffset p) (fun _ => Bool.true) (kvValues M D p s) s).mem := by
  simp [kvSource,kvTyped,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,natLe,
    Region.cast,ofFloat,toFloat,store,State.setReg,backend,
    kvValues,interValues,bodyValues,bodyQuery,bodyKey,bodyValue,bodyScore,carryValues,
    gradValues,read32,maskedRead32,kvOffset,interOffset,bodyOffset,bodyActive,gradOffset,gradActive]
  simp only [written,kvOffset,interOffset,bodyOffset,bodyActive,gradOffset,gradActive,
    kvValues,interValues,bodyValues,bodyQuery,bodyKey,bodyValue,bodyScore,carryValues,
    gradValues,read32,maskedRead32,pair_eta,Fin.isLt,decide_true,Bool.and_true,
    Bool.true_eq,ite_true,decide_eq_true_eq]
  first | apply scatter_memory_congr | apply fold_memory_congr
  rfl
#axiomsClean kv_execution_memory

theorem inter_execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    ∃ final, FP.Structural.exec (backend M D) (interSource p) s = some final ∧
      final.mem = (written p.output (interOffset p) (fun _ => Bool.true) (interValues M D p s) s).mem := by
  simp [interSource,interTyped,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,natLe,
    Region.cast,ofFloat,toFloat,store,State.setReg,backend,
    kvValues,interValues,bodyValues,bodyQuery,bodyKey,bodyValue,bodyScore,carryValues,
    gradValues,read32,maskedRead32,kvOffset,interOffset,bodyOffset,bodyActive,gradOffset,gradActive]
  simp only [written,kvOffset,interOffset,bodyOffset,bodyActive,gradOffset,gradActive,
    kvValues,interValues,bodyValues,bodyQuery,bodyKey,bodyValue,bodyScore,carryValues,
    gradValues,read32,maskedRead32,pair_eta,Fin.isLt,decide_true,Bool.and_true,
    Bool.true_eq,ite_true,decide_eq_true_eq]
  first | apply scatter_memory_congr | apply fold_memory_congr
  rfl
#axiomsClean inter_execution_memory

theorem body_execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    ∃ final, FP.Structural.exec (backend M D) (bodySource p) s = some final ∧
      final.mem = (written p.output (bodyOffset p s) (bodyActive p) (bodyValues M D p s) s).mem := by
  unfold bodyValues bodyScore bodyQuery bodyKey bodyValue carryValues maskedRead32 read32
  simp [bodySource,bodyTyped,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,natLe,
    Region.cast,ofFloat,toFloat,store,State.setReg,backend,
    kvValues,interValues,bodyValues,bodyQuery,bodyKey,bodyValue,bodyScore,carryValues,
    gradValues,read32,maskedRead32,kvOffset,interOffset,bodyOffset,bodyActive,gradOffset,gradActive]
  simp only [written,kvOffset,interOffset,bodyOffset,bodyActive,gradOffset,gradActive,
    kvValues,interValues,bodyValues,bodyQuery,bodyKey,bodyValue,bodyScore,carryValues,
    gradValues,read32,maskedRead32,pair_eta,Fin.isLt,decide_true,Bool.and_true,
    Bool.true_eq,ite_true,decide_eq_true_eq]
  first | apply scatter_memory_congr | apply fold_memory_congr
  rfl
#axiomsClean body_execution_memory

theorem grad_execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    ∃ final, FP.Structural.exec (backend M D) (gradSource p) s = some final ∧
      final.mem = (written p.output (gradOffset p s) (gradActive p s) (gradValues M p s) s).mem := by
  simp [gradSource,gradTyped,ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,natLe,
    Region.cast,ofFloat,toFloat,store,State.setReg,backend,
    kvValues,interValues,bodyValues,bodyQuery,bodyKey,bodyValue,bodyScore,carryValues,
    gradValues,read32,maskedRead32,kvOffset,interOffset,bodyOffset,bodyActive,gradOffset,gradActive]
  simp only [written,kvOffset,interOffset,bodyOffset,bodyActive,gradOffset,gradActive,
    kvValues,interValues,bodyValues,bodyQuery,bodyKey,bodyValue,bodyScore,carryValues,
    gradValues,read32,maskedRead32,pair_eta,Fin.isLt,decide_true,Bool.and_true,
    Bool.true_eq,ite_true,decide_eq_true_eq]
  first | apply scatter_memory_congr | apply fold_memory_congr
  rfl
#axiomsClean grad_execution_memory

end
end VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
