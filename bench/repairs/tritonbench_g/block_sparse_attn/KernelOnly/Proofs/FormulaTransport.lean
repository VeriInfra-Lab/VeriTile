import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.InputAgreement

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

theorem source_conditions {α β : Type} [Inhabited β]
    (a : Arithmetic β) (native : Bool) (encode : α→β) (observe : β→α)
    (p : Parameters) (batches visits : Nat) (xs : Inputs α p batches visits)
    (s : State (Replicas β)) (binding : (io a native encode observe p batches visits).InputBinding xs s)
    (r : Row p) (h : Head p batches) (c : Fin p.blocks)
    (ids : s.pids=PhysicalKernel.Program.pids ⟨r.val,h.val,c.val⟩)
    (admitted : Admissible p batches visits xs) :
    0≤KernelOnly.first p s ∧ ∀ v,v<visitCount p s → 0≤csrColumn p s (KernelOnly.first p s+v) := by
  have blocks : 0<p.blocks := Nat.zero_lt_of_lt c.isLt
  have hf : KernelOnly.first p s=first xs r h := by
    simpa [KernelOnly.first,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      PhysicalKernel.Program.pids,ids,layout,head,first] using
      endpoint_binding a native encode observe p batches visits xs s binding r h 0 blocks
  have hl : KernelOnly.last p s=last xs r h := by
    simpa [KernelOnly.last,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      PhysicalKernel.Program.pids,ids,layout,head,last] using
      endpoint_binding a native encode observe p batches visits xs s binding r h 1 blocks
  have hc : visitCount p s=count xs r h := by simp only [visitCount,hf,hl,count]
  have domain := admitted r h blocks
  refine ⟨hf.symm ▸ domain.2.2.1,?_⟩
  intro v hv
  have live : v<count xs r h := hc ▸ hv
  have limit : v<visits := lt_of_lt_of_le live domain.2.2.2.1
  have read := column_binding a native encode observe p batches visits xs s binding r h ⟨v,limit⟩ blocks live
  have result := (domain.2.2.2.2 ⟨v,limit⟩ live).2
  rw [← read] at result
  simpa only [csrColumn,hf,KernelOnly.layout,KernelOnly.head,ids,
    PhysicalKernel.Program.pids,layout,head,Int.add_assoc] using result

theorem real_output (p : Parameters) (batches visits : Nat) (xs : Inputs ℝ p batches visits)
    (s : State (Replicas R)) (binding : (realIO p batches visits).InputBinding xs s)
    (r : Row p) (h : Head p batches) (c : Fin p.blocks)
    (ids : s.pids=PhysicalKernel.Program.pids ⟨r.val,h.val,c.val⟩)
    (admitted : Admissible p batches visits xs) (i : TileIndex [16,16])
    (live : KernelOnly.row s i.1<p.length) :
    KernelOnly.output mathArithmetic Bool.false p s i (owner i.2.1.val)=
      some (realFormula p batches visits xs (0 : Fin 1) (r,h,c,i.1,i.2.1,PUnit.unit)) := by
  let t := reified (0 : ℝ) some p batches visits xs s.pids
  have agree := input_agreement mathArithmetic Bool.false (0 : ℝ) some (WithBot.unbotD 0)
    p batches visits xs s binding r h c ids admitted
  have conditions := source_conditions mathArithmetic Bool.false some (WithBot.unbotD 0)
    p batches visits xs s binding r h c ids admitted
  rw [agree.output Bool.false]
  have views : Views t (BlockSparseAttn.Real.lift (memory 0 p batches visits xs))
      (rows p batches visits xs) (columns p batches visits xs) :=
    reified_views 0 some p batches visits xs s.pids
  have positive : 0≤KernelOnly.first p t := agree.first_eq ▸ conditions.1
  have columnsPositive : ∀ v,v<visitCount p t → 0≤csrColumn p t (KernelOnly.first p t+v) := by
    intro v hv
    have active : v<visitCount p s := agree.count.symm ▸ hv
    exact agree.column_eq v active ▸ conditions.2 v active
  have value := output_real p batches t (memory 0 p batches visits xs) _ _ views positive columnsPositive i live
  simpa only [realFormula,targetLayout,coordinate,work,t,reified,KernelOnly.batch,KernelOnly.head,
    KernelOnly.row,KernelOnly.channel,ids,PhysicalKernel.Program.pids,batch,head] using value

theorem native_output (ops : PhysicalKernel.NativePrimitives) (p : Parameters) (batches visits : Nat)
    (xs : Inputs Word p batches visits) (s : State (Replicas Word))
    (binding : (nativeIO ops p batches visits).InputBinding xs s)
    (r : Row p) (h : Head p batches) (c : Fin p.blocks)
    (ids : s.pids=PhysicalKernel.Program.pids ⟨r.val,h.val,c.val⟩)
    (admitted : Admissible p batches visits xs) (i : TileIndex [16,16]) :
    KernelOnly.output ops.arithmetic Bool.true p s i (owner i.2.1.val)=
      nativeFormula ops p batches visits xs (0 : Fin 1) (r,h,c,i.1,i.2.1,PUnit.unit) := by
  let t := reified (0 : Word) id p batches visits xs s.pids
  have agree := input_agreement ops.arithmetic Bool.true (0 : Word) id id
    p batches visits xs s binding r h c ids admitted
  have conditions := source_conditions ops.arithmetic Bool.true id id
    p batches visits xs s binding r h c ids admitted
  rw [agree.output Bool.true]
  have views : Views t (memory 0 p batches visits xs)
      (rows p batches visits xs) (columns p batches visits xs) :=
    reified_views 0 id p batches visits xs s.pids
  have positive : 0≤KernelOnly.first p t := agree.first_eq ▸ conditions.1
  have columnsPositive : ∀ v,v<visitCount p t → 0≤csrColumn p t (KernelOnly.first p t+v) := by
    intro v hv
    have active : v<visitCount p s := agree.count.symm ▸ hv
    exact agree.column_eq v active ▸ conditions.2 v active
  have value := output_view (BlockSparseAttn.Native.ops ops.instructions) ops.literal (fun _=>ops.scale)
    ops.zero ops.negativeInfinity p batches t (memory 0 p batches visits xs) _ _ views positive columnsPositive i
  simpa only [nativeFormula,targetLayout,coordinate,work,t,reified,KernelOnly.batch,KernelOnly.head,
    KernelOnly.row,KernelOnly.channel,ids,PhysicalKernel.Program.pids,batch,head] using value

#axiomsClean real_output
#axiomsClean native_output
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
