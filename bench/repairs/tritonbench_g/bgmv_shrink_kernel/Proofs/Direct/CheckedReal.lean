import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Contract
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.AtomicOrder
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.Linearization

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.CheckedReal
open VeriTile.Triton
open scoped BigOperators
noncomputable section

/-- Independent GEMV in original dense caller coordinates. -/
def math (h : Call) (initial : BlockState) (indices : Nat → Int) (b : Fin h.batches) (n : Fin h.N) : ℝ :=
  let old:=initial.readMem "Y" (b.val*h.columns+n.val)
  if indices b.val=-1 then old else old+h.scaling*∑ k : Fin h.K,
    initial.readMem "X" (b.val*h.K+k.val)*initial.readMem "W" ((indices b.val).toNat*(h.N*h.K)+n.val*h.K+k.val)

def initial (h : Call) (u : BlockState) (b : Fin h.batches) (n : Fin h.N) : ℝ :=
  u.readMem "Y" (Geometry.outputOffset h b.val n.val)
def step (h : Call) (u : BlockState) (indices : Nat → Int)
    (b : Fin h.batches) (n : Fin h.N) (split : Fin 64) (old : ℝ) :=
  old+RealExecution.contribution h u b.val (indices b.val).toNat n.val split.val
abbrev Run (h : Call) (orders : Workset.Orders h) (u : BlockState) (indices : Nat → Int) :=
  Linearization.Run h orders indices (initial h u) (step h u indices)

def SourceWorkset (h : Call) (u : BlockState) (indices : Nat → Int) : Prop :=
  ∀ (b : Fin h.batches) (_n : Fin h.N) (split : Fin 64),indices b.val≠-1 →
    ∀ current : BlockState,current.pids 1=b.val → current.pids 0=split.val →
    current.readMemValue .int "Indices" b.val=indices b.val →
    (∀ off,current.readMem "X" off=u.readMem "X" off) →
    (∀ off,current.readMem "W" off=u.readMem "W" off) →
    ∃ final,exec (Kernels.source h).eraseDType.toAlgKernel current=some final ∧
      (∀ n : Fin h.N,final.readMem "Y" (Geometry.outputOffset h b.val n.val)=
        step h u indices b n split (current.readMem "Y" (Geometry.outputOffset h b.val n.val))) ∧
      (∀ r off,¬RealExecution.writes h ⟨b,split⟩ (indices b.val) r off → final.mem r off=current.mem r off)

theorem source_workset (x w y i : Contract.TensorInfo) (s : Contract.Scale) (indices : Nat → Int)
    (ok : Contract.Admitted x w y i s indices) (u : BlockState) :
    SourceWorkset (Contract.call x w y i s indices ok) u indices := by
  intro b n split active current batch splitId metadata input weights
  have valid:=(Contract.selected_valid x w y i s indices ok b.val n.val b.isLt n.isLt).resolve_left active
  obtain ⟨final,execution,_,values,frame⟩:=RealExecution.source_executes (Contract.call x w y i s indices ok)
    ⟨b,split⟩ current batch splitId (indices b.val) (Or.inr valid.1) metadata
  refine ⟨final,execution,?_,frame⟩
  intro m
  rw [values active m]
  simp only [step,RealExecution.contribution,input,weights]

theorem readback (h : Call) (orders : Workset.Orders h) (u : BlockState) (indices : Nat → Int)
    (run : Run h orders u indices) (b : Fin h.batches) (n : Fin h.N) :
    run.output b n=math h u indices b n := by
  rw [Linearization.Run.readback]
  unfold math initial
  by_cases skip : indices b.val=-1
  · simp [skip,Geometry.outputOffset]
  · simp only [skip,if_false]
    exact AtomicOrder.contribution_readback h orders u b n (indices b.val).toNat

theorem complete (x w y i : Contract.TensorInfo) (s : Contract.Scale) (indices : Nat → Int)
    (ok : Contract.Admitted x w y i s indices) (u : BlockState)
    (orders : Workset.Orders (Contract.call x w y i s indices ok))
    (run : Run (Contract.call x w y i s indices ok) orders u indices) :
    SourceWorkset (Contract.call x w y i s indices ok) u indices ∧
    ∀ b n,run.output b n=math (Contract.call x w y i s indices ok) u indices b n :=
  ⟨source_workset x w y i s indices ok u,readback _ orders u indices run⟩

#axiomsClean source_workset
#axiomsClean readback
#axiomsClean complete
#specNonCircular math avoiding [Run,RealExecution.contribution,Kernels.source,VeriTile.Triton.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.CheckedReal
