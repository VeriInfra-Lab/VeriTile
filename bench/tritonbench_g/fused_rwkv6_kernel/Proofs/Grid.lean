import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.ModelBridge

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite FusedRwkv6Kernel
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

abbrev Cells (α : Type) := RegionName → Nat → Cell α
def words (m : Cells α) : FusedRwkv6Kernel.Memory α :=
  fun r b => (m (regionName r) b).read .real

def prepared (c : Shape) (p : FusedRwkv6Kernel.Program) (scale : α) (m : Cells α) : State α :=
  let s : State α := {
    mem:=m,regs:=fun _ _ _=>none,
    pids:=fun ax=>if ax=0 then p.iv else if ax=1 then p.ik else if ax=2 then p.z else 0,
    numPids:=fun ax=>if ax=0 then c.NV else if ax=1 then c.NK else if ax=2 then c.Z else 0,
    undef:=fun dtype _ _=>defaultValue dtype }
  s.setReg "scale" .real [] (fun _=>scale)

@[simp] theorem prepared_program (c : Shape) (p : FusedRwkv6Kernel.Program) (scale : α) (m : Cells α) :
    program (prepared c p scale m)=p := by cases p; simp [program,prepared]
@[simp] theorem prepared_memory (c : Shape) (p : FusedRwkv6Kernel.Program) (scale : α) (m : Cells α) :
    memory (prepared c p scale m)=words m := rfl

def runOne (M : Algebra α) (c : Shape) (p : FusedRwkv6Kernel.Program) (scale : α)
    (m : Cells α) : Option (Cells α) :=
  (FP.Structural.exec (engine M) (kernel c) (prepared c p scale m)).map State.mem

def runGrid (M : Algebra α) (c : Shape) (scale : α) :
    List FusedRwkv6Kernel.Program → Cells α → Option (Cells α)
  | [],m=>some m
  | p::rest,m=>runOne M c p scale m >>= runGrid M c scale rest

/-- Internal reduction-to-scalar interface, discharged by the concrete
Real/native backend lemmas. It is not a premise of public task endpoints. -/
def ReductionMatches (M : Algebra α) (c : Shape) (plan : Plans c) : Prop :=
  ∀ (s : State α) scale n (i : TileIndex [c.BV]),rowValues M c s scale n i=
    (ops M).storeCast (scalarPartial (ops M) c plan (inputs c (memory s)) scale
      (program s).z (program s).ik n (valueIndex c (program s) i.1.val))

theorem one_model (M : Algebra α) (c : Shape) (plan : Plans c) (reduction : ReductionMatches M c plan)
    (p : FusedRwkv6Kernel.Program) (scale : α) (m : Cells α) :
    ∃ out,runOne M c p scale m=some out ∧
      words out=FusedRwkv6Kernel.runProgram (ops M) c plan p scale (words m) := by
  let s:=prepared c p scale m
  obtain ⟨t,ht,hm⟩:=source_memory M c s scale (by simp [s,prepared])
  refine ⟨t.mem,by simp [runOne,ht,s],?_⟩
  funext r b
  change (t.mem (regionName r) b).read .real=_
  rw [hm,program_model M c s scale plan (reduction s scale)]
  simp [s]

theorem grid_model (M : Algebra α) (c : Shape) (plan : Plans c) (reduction : ReductionMatches M c plan)
    (ps : List FusedRwkv6Kernel.Program) (scale : α) (m : Cells α) :
    ∃ out,runGrid M c scale ps m=some out ∧
      words out=FusedRwkv6Kernel.runGrid (ops M) c plan scale ps (words m) := by
  induction ps generalizing m with
  | nil => exact ⟨m,rfl,rfl⟩
  | cons p ps ih =>
    obtain ⟨mid,hm,hwords⟩:=one_model M c plan reduction p scale m
    obtain ⟨out,ho,hw⟩:=ih mid
    refine ⟨out,by simp [runGrid,hm,ho],?_⟩
    rw [hw,hwords]
    rfl

theorem native_grid_model (p : Native.ReductionPlan) (c : Shape)
    (ps : List FusedRwkv6Kernel.Program) (scale : Native.Term) (m : Cells Native.Term) :
    ∃ out,runGrid (Native.algebra p) c scale ps m=some out ∧
      words out=FusedRwkv6Kernel.runGrid (ops (Native.algebra p)) c (Native.plans c p) scale ps (words m) :=
  grid_model (Native.algebra p) c (Native.plans c p) (native_rows p c) ps scale m

theorem real_grid_model (c : Shape) (ps : List FusedRwkv6Kernel.Program) (scale : ℝ) (m : Cells ℝ) :
    ∃ out,runGrid Real.algebra c scale ps m=some out ∧
      words out=FusedRwkv6Kernel.runGrid realOps c (sequentialPlans c) scale ps (words m) := by
  simpa only [Real.source_ops] using
    grid_model Real.algebra c (sequentialPlans c) (real_rows c) ps scale m

#axiomsClean native_grid_model
#axiomsClean real_grid_model
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
