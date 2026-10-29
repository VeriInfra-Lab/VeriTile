import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.NativeWhole
import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.RealWhole
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.TorchBinding
open VeriTile.Triton FP.Structural Host Checked ByteBridge
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000

theorem helper_contract (a : Arguments) (h : InputContract a .s1) : InputContract a .dequant16 := by
  simpa [InputContract,metadataChecks,indexChecks,candidates,expressionChecks,allocationChecks,
    ABI.count,ABI.expressions,ABI.roots,ABI.accesses] using h

def helper_tensors (a : Arguments) (t : TensorBinding.Tensors a .s1) : TensorBinding.Tensors a .dequant16 where
  get := t.get
  bound := by
    intro d hd
    apply t.bound
    simp [TensorBinding.live,tensors] at hd ⊢
    tauto
  storageIdentity := t.storageIdentity

structure MMResult where
  memory : Memory
  output : TensorBinding.Tensor

def OutsideOutput (a : Arguments) (r : RegionName) (b : Nat) : Prop :=
  r≠a.c.region ∨ b<2*a.c.storageOffset ∨ 2*(a.c.storageOffset+a.M*a.N)≤b

/-- Documented external torch.mm FP16/out API, indexed by the actual source
Tensor arguments. It is an ordinary library-call boundary, not one of the
Triton scalar/fixed-ISA numerical leaves. Native library algorithms and Torch
internals are outside this task. No Triton/dequantized result is assumed. -/
structure NativeMM (a : Arguments) (t : TensorBinding.Tensors a .s1) where
  run : TensorBinding.Tensor → TensorBinding.Tensor → TensorBinding.Tensor → Memory → Option MMResult
  complete : InputContract a .s1 → ∀ m,∃ result,
    run (t.get a.a.region) (t.get a.d.region) (t.get a.c.region) m=some result ∧
    result.output=t.get a.c.region ∧
    (∀ r b,OutsideOutput a r b → result.memory r b=m r b)

def invoke (a : Arguments) (t : TensorBinding.Tensors a .s1) (h : InputContract a .s1)
    (api : NativeMM a t) (plan : Native.Plan) (actual : Native.WordISA) (before : Memory) :
    Except Error (Option TensorBinding.Result) := do
  checkInputs a .s1
  let scratch ← NativeWhole.invoke a .dequant16 (helper_tensors a t)
    (DequantGrid16.geometry (Host.dequant16 a)) 2 Bool.true plan actual before
    (NativeWhole.dequant16Action a (helper_contract a h) before)
  pure (scratch.bind fun d =>
    (api.run (t.get a.a.region) d.output (t.get a.c.region) d.memory).map fun result =>
      ⟨result.memory,result.output,some result.output,!a.supplied,Bool.true⟩)

def dequantized (a : Arguments) (plan : Native.Plan) (target : Native.WordISA) (before : Memory) :=
  NativeWhole.expected a.d (DequantGrid16.geometry (Host.dequant16 a)) 2 plan target before
    (NativeWhole.dequant16Reference a before)

def Guarantee (a : Arguments) (t : TensorBinding.Tensors a .s1) (api : NativeMM a t)
    (plan : Native.Plan) (target : Native.WordISA) (before : Memory) (result : TensorBinding.Result) : Prop :=
  result.output=t.get a.c.region ∧ result.returned=some (t.get a.c.region) ∧
  result.allocated=(!a.supplied) ∧ result.scratchReleased=Bool.true ∧
  TensorBinding.Binding a.c result.output ∧ ABI.SourceExact a .s1 .m64n256 ∧
  (∃ libraryResult,api.run (t.get a.a.region) (t.get a.d.region) (t.get a.c.region)
    (dequantized a plan target before)=some libraryResult ∧
    libraryResult.output=result.output ∧ libraryResult.memory=result.memory) ∧
  (∀ r b,r≠a.d.region → OutsideOutput a r b → result.memory r b=before r b)

theorem native (a : Arguments) (t : TensorBinding.Tensors a .s1) (h : InputContract a .s1)
    (api : NativeMM a t) (plan : Native.Plan) (actual target : Native.WordISA)
    (ports : Numerics actual target) (before : Memory) :
    ∃ result,invoke a t h api plan actual before=.ok (some result) ∧ Guarantee a t api plan target before result := by
  obtain ⟨scratch,hd,hg⟩ := NativeWhole.dequant16 a (helper_tensors a t) (helper_contract a h) plan actual target ports before
  have scratchOutput : scratch.output=t.get a.d.region := hg.1
  have scratchMemory : scratch.memory=dequantized a plan target before := hg.2.2.2.2.2.2.1
  obtain ⟨mm,hm,output,frame⟩ := api.complete h scratch.memory
  refine ⟨⟨mm.memory,mm.output,some mm.output,!a.supplied,Bool.true⟩,?_,output,congrArg some output,rfl,rfl,?_,?_,?_,?_⟩
  · simp [invoke,(checkInputs_iff a .s1).mpr h,hd,scratchOutput,hm]
    rfl
  · rw [output]
    exact TensorBinding.output_bound a .s1 t
  · apply ABI.source_exact a .s1 .m64n256
    exact indices a .s1 .m64n256 (by simp [candidates]) h
  · exact ⟨mm,by simpa only [scratchMemory] using hm,rfl,rfl⟩
  · intro r b hd hc
    exact (frame r b hc).trans (hg.2.2.2.2.2.2.2.2 r b (Or.inl hd))

/-- The real library boundary is its documented mathematical matrix product.
The exact A/scratch/C Tensor arguments are supplied by `t`, and the scratch
state is produced by the proved dequantization grid before this call. -/
theorem real (api : TorchMM.RealAPI) (a : Arguments) (t : TensorBinding.Tensors a .s1)
    (h : InputContract a .s1) (raw : Memory) (heap : RealWhole.Heap) :
    ∃ result,RealWhole.invoke a .s1 t
      (TorchMM.s1RealRun api a (admitted a .s1 h)
        (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a)))) (RealWhole.entry a raw heap))=.ok (some result) ∧
      RealWhole.Guarantee a .s1 .m64n256 t (SecondGrid.geometry (second a)) Bool.true
        (RealWhole.entry a raw heap)
        (fun x => Real.rawMatmul (RealFirst.inputs (first a .m64n256) (RealWhole.entry a raw heap)) a.K a.group x.1.val x.2.val) result := by
  have ha := admitted a .s1 h
  have candidate : FirstConfig.m64n256∈candidates .s1 := by simp [candidates]
  obtain ⟨_,middle,final,_,_,run,values,frame,_⟩ := TorchMM.s1_real api a ha
    (indices a .s1 .m64n256 candidate h) (RealWhole.entry a raw heap) (RealWhole.packed a .s1 ha raw heap)
    (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a)))) (List.Perm.refl _)
  apply RealWhole.complete a .s1 .m64n256 t h candidate (SecondGrid.geometry (second a)) Bool.true
    (RealWhole.entry a raw heap)
    (fun x => Real.rawMatmul (RealFirst.inputs (first a .m64n256) (RealWhole.entry a raw heap)) a.K a.group x.1.val x.2.val)
    _ final run
  · exact ⟨ha.cv.1,ha.cv.2.1⟩
  · exact ⟨rfl,rfl⟩
  · intro x; simp only [TensorBinding.output,reduceCtorEq,false_or,or_true,true_or,ite_false,ite_true]; rw [values x]; rfl
  · intro r o scratch outside
    exact frame r o (scratch rfl) outside

#specNonCircular Guarantee avoiding [invoke,NativeWhole.invoke,Grid.run,ByteBridge.serialize]
#axiomsClean helper_contract
#axiomsClean native
#axiomsClean real
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.TorchBinding
