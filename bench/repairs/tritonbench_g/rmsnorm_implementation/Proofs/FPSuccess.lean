import bench.repairs.tritonbench_g.rmsnorm_implementation.Kernels
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RmsnormImplementation.FPSuccess
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

def prelude (sxb sxm B : Nat) : List ComputeStmt :=
  (kernel "x" "weights" "out" sxb sxm 1 1 1 1 1 1 B 0).surfaceBody.take 5

def bodyOf : ComputeStmt → List ComputeStmt
  | .forRange _ _ _ _ body => body
  | _ => []

def varBody (sxk n B : Nat) : List ComputeStmt :=
  bodyOf ((kernel "x" "weights" "out" 0 0 sxk 1 1 1 1 n B 0).surfaceBody[5]'(by simp [kernel,ComputeKernel.surfaceBody]))

def post (n B : Nat) (eps : ℝ) : List ComputeStmt :=
  ((kernel "x" "weights" "out" 0 0 1 1 1 1 1 n B eps).surfaceBody.drop 6).take 2

def outBody (sxk sw sob som sok n B : Nat) : List ComputeStmt :=
  bodyOf ((kernel "x" "weights" "out" 0 0 sxk sw sob som sok n B 0).surfaceBody[8]'(by simp [kernel,ComputeKernel.surfaceBody]))

def Fixed (B : Nat) (s : State α) : Prop :=
  (∃ v,s.regs .nat [] "pid_batch"=some v) ∧
  (∃ v,s.regs .nat [] "pid_m"=some v) ∧
  (∃ v,s.regs .nat [] "offset_m"=some v) ∧
  (∃ v,s.regs .nat [B] "block_n_size"=some v)

def VarInvariant (B : Nat) (s : State α) : Prop := Fixed B s ∧ ∃ v,s.regs .real [B] "var"=some v
def OutInvariant (B : Nat) (s : State α) : Prop := Fixed B s ∧ ∃ v,s.regs .real [] "std"=some v

theorem prelude_success (M : Algebra α) (sxb sxm B : Nat) (s : State α) :
    ∃ t,run M (prelude sxb sxm B) s=some t ∧ VarInvariant B t := by
  refine ⟨?_,?_,?_⟩
  rotate_left
  · simp [prelude,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,evalDeltas,ComputeDType.eraseDType,ofFloat,toFloat,Option.bind,State.setReg,numeric,bop]
    rfl
  · simp [VarInvariant,Fixed,State.setReg]

theorem var_success (M : Algebra α) (sxk n B off : Nat) (s : State α)
    (inv : VarInvariant B s) :
    ∃ t,run M (varBody sxk n B) (s.setReg "block_n_strart_ptr" .nat [] (fun _=>off))=some t ∧
      VarInvariant B t := by
  rcases inv with ⟨⟨⟨pb,hpb⟩,⟨pm,hpm⟩,⟨offset,ho⟩,⟨lanes,hl⟩⟩,⟨acc,ha⟩⟩
  refine ⟨?_,?_,?_⟩
  rotate_left
  · simp [varBody,bodyOf,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,evalDeltas,ComputeDType.eraseDType,ofFloat,toFloat,Option.bind,State.setReg,numeric,bop,natLt,ho,hl,ha]
    rfl
  · simp [VarInvariant,Fixed,State.setReg,hpb,hpm,ho,hl]

theorem post_success (M : Algebra α) (n B : Nat) (eps : ℝ) (s : State α)
    (inv : VarInvariant B s) :
    ∃ t,run M (post n B eps) s=some t ∧ OutInvariant B t := by
  rcases inv with ⟨⟨⟨pb,hpb⟩,⟨pm,hpm⟩,⟨offset,ho⟩,⟨lanes,hl⟩⟩,⟨acc,ha⟩⟩
  refine ⟨?_,?_,?_⟩
  rotate_left
  · simp [post,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,evalDeltas,ComputeDType.eraseDType,ofFloat,toFloat,Option.bind,State.setReg,numeric,bop,ha]
    rfl
  · simp [OutInvariant,Fixed,State.setReg,hpb,hpm,ho,hl]

theorem masked_write_regs {ι : Type} (lanes : List ι) (r : RegionName)
    (address : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (lanes.foldl (fun t i=>if active i then t.write r (address i) (value i) else t) s).regs=s.regs := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons,ih]
    split <;> rfl

theorem out_success (M : Algebra α) (sxk sw sob som sok n B off : Nat) (s : State α)
    (inv : OutInvariant B s) :
    ∃ t,run M (outBody sxk sw sob som sok n B)
      (s.setReg "block_n_strart_ptr" .nat [] (fun _=>off))=some t ∧ OutInvariant B t := by
  rcases inv with ⟨⟨⟨pb,hpb⟩,⟨pm,hpm⟩,⟨offset,ho⟩,⟨lanes,hl⟩⟩,⟨std,hs⟩⟩
  refine ⟨?_,?_,?_⟩
  rotate_left
  · simp [outBody,bodyOf,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,evalDeltas,ComputeDType.eraseDType,ofFloat,toFloat,Option.bind,State.setReg,numeric,bop,natLt,
      ho,hl,hpb,hpm,hs,store,Region.cast]
    rfl
  · simp [OutInvariant,Fixed,masked_write_regs,State.setReg,hpb,hpm,ho,hl,hs]

/-- The invariant proves successful termination; it is not a premise about a
completed source or an assumed output. -/
theorem range_success (M : Algebra α) (idx : RegName) (stop stride : Nat)
    (body : List ComputeStmt) (P : State α → Prop)
    (advance : ∀ off s,P s → ∃ t,run M body (s.setReg idx .nat [] (fun _=>off))=some t ∧ P t)
    (cur : Nat) (s : State α) (hs : P s) :
    ∃ t,range M idx cur stop stride body s=some t ∧ P t := by
  induction cur using (measure (fun i=>stop-i)).wf.induction generalizing s with
  | h cur ih =>
    rw [range]
    by_cases hz : stride=0
    · exact ⟨s,by simp [hz],hs⟩
    by_cases hc : cur<stop
    · obtain ⟨u,hu,hp⟩ := advance cur s hs
      obtain ⟨t,ht,pt⟩ := ih (cur+stride) (by simp_wf;omega) u hp
      exact ⟨t,by simp [hz,hc,hu,ht],pt⟩
    · exact ⟨s,by simp [hz,hc],hs⟩

theorem success (M : Algebra α) (sxb sxm sxk sw sob som sok n B : Nat) (eps : ℝ) (s : State α) :
    ∃ t,exec M (kernel "x" "weights" "out" sxb sxm sxk sw sob som sok n B eps) s=some t := by
  have shape : (kernel "x" "weights" "out" sxb sxm sxk sw sob som sok n B eps).surfaceBody =
      prelude sxb sxm B ++ [.forRange "block_n_strart_ptr" 0 n B (varBody sxk n B)] ++
      post n B eps ++ [.forRange "block_n_strart_ptr" 0 n B (outBody sxk sw sob som sok n B)] := rfl
  obtain ⟨s1,h1,i1⟩ := prelude_success M sxb sxm B s
  obtain ⟨s2,h2,i2⟩ := range_success M "block_n_strart_ptr" n B (varBody sxk n B)
    (VarInvariant B) (var_success M sxk n B) 0 s1 i1
  obtain ⟨s3,h3,i3⟩ := post_success M n B eps s2 i2
  obtain ⟨s4,h4,_⟩ := range_success M "block_n_strart_ptr" n B (outBody sxk sw sob som sok n B)
    (OutInvariant B) (out_success M sxk sw sob som sok n B) 0 s3 i3
  refine ⟨s4,?_⟩
  change run M (kernel "x" "weights" "out" sxb sxm sxk sw sob som sok n B eps).surfaceBody s=some s4
  rw [shape]
  simp [run_append,h1,Option.bind,run,step,h2,h3,h4]

#axiomsClean success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.FPSuccess
