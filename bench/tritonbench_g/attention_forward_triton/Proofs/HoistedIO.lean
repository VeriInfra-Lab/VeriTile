import bench.tritonbench_g.attention_forward_triton.Proofs.HoistedGrid
import bench.tritonbench_g.attention_forward_triton.Proofs.HoistedSafety
import bench.tritonbench_g.attention_forward_triton.Proofs.PublicResult

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedIO
open VeriTile Triton FP.Structural Source Whole
open AttentionIO
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem invocation_initial {A : Type} (M : Algebra A) (dot : Dot A)
    (p : Parameters) (pid : Whole.Program) (s : State A) :
    (invocation M dot p pid).initial s=start (grid p) pid s := rfl

theorem schedule_run {A : Type} [Inhabited A] (M : Algebra A) (dot : Dot A)
    (p : Parameters) (work : List Whole.Program) (s : State A) :
    runSchedule (work.map (invocation M dot p)) s=
      HoistedWhole.runPrograms (fun _ => M) (fun _ => dot) p (grid p) work s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,invocation_initial,HoistedWhole.runPrograms]
    exact congrArg (Option.bind (FP.Structural.exec (VeriTile.Triton.FP.ProfiledRewrite.engine (withDot M dot)) (HoistedSource.kernel p)
      (start (grid p) pid s))) (funext ih)

theorem schedule_safe {A : Type} [Inhabited A] (M : Algebra A) (dot : Dot A)
    (p : Parameters) (work : List Whole.Program)
    (legal : ∀ pid∈work,pid∈programs (grid p)) (bounds : RegionName → Nat)
    (input : ∀ i j, inputAddress p i j<bounds (inputChannels p)[i].buf)
    (output : ∀ j : DataLane p, dataAddress p.o j<bounds p.Out) (s : State A) :
    ScheduleSafe bounds (work.map (invocation M dot p)) s := by
  induction work generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨?_,fun t _ => ih (fun pid member => legal pid (by simp [member])) t⟩
    change TraceSafe (VeriTile.Triton.FP.ProfiledRewrite.engine (withDot M dot)) bounds (HoistedSource.kernel p) _
    rw [invocation_initial]
    exact HoistedAccess.source_safe M dot p _ bounds
      (source_bounds p s pid (legal pid (by simp)) bounds input output)

theorem native_implements (I : Native.Instructions) (p : Parameters) (sep : Separated p)
    (injective : RawKernel.OutputInjective p) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    (nativeIO I p work).Implements (nativeTarget I p) := by
  intro vals _ bounds backed s bound
  have input : ∀ i j,inputAddress p i j<bounds (inputChannels p)[i].buf :=
    fun i j => backed.2.1 i j trivial
  have output : ∀ j : DataLane p,dataAddress p.o j<bounds p.Out := by
    intro j
    simpa [nativeIO,AttentionIO.nativeIO,memory,outputChannels,Fin.cases] using
      backed.2.2.1 ⟨0,by simp [nativeIO,AttentionIO.nativeIO,memory,outputChannels]⟩ j trivial
  obtain ⟨t,ran,values,frame⟩ := HoistedWhole.grid_result (fun _ => Native.algebra)
    (fun _ => Native.dotModel) p sep injective s work legal
  refine ⟨schedule_safe Native.algebra Native.dotModel p work
    (fun pid member => legal.mem_iff.mpr member) bounds input output s,t,?_,?_,?_⟩
  · exact (schedule_run Native.algebra Native.dotModel p work s).trans ran
  · intro o j _
    fin_cases o
    change mapCell (fun value => (Native.lower (value (Native.physicalOwner j.2.2.2.1.val))).eval I.ops)
      (t.mem p.Out (logicalAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val))=Cell.mk .fp16 (nativeTarget I p vals ⟨0,by simp [outputChannels]⟩ j)
    rw [values _ j.1.isLt _ j.2.1.isLt _ j.2.2.1.isLt _ j.2.2.2.1.isLt]
    exact congrArg (Cell.mk .fp16) (native_formula I p work vals s bound j)
  · intro r off outside
    apply frame r off
    apply exterior p r off
    intro j
    simpa [nativeIO,AttentionIO.nativeIO,memory,outputChannels,Fin.cases] using
      outside.1 ⟨0,by simp [nativeIO,AttentionIO.nativeIO,memory,outputChannels]⟩ j trivial


#axiomsClean schedule_safe
#axiomsClean schedule_run
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedIO
