import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageDomain

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

/-- Ordinary selected CUDA grid and compilation/setup admission. Successful
compilation of the source `tl.arange(0, BLOCK_N)` requires a positive end.
This is a local compiler configuration rule, not an execution/output contract.
The wrapper itself retains N=0 and computes BLOCK_N=0 for that request. -/
structure DispatchAdmission (api : DispatchAPI world) : Prop where
  grid : ∀ args s, ((api.admit args).run s).2=.ok () → args.prepared.validated.M≤2^31-1
  block : ∀ args s, ((api.admit args).run s).2=.ok () → 0<args.prepared.block

theorem width_fp32 (api : TensorAPI world) (v : Validated) (input : Inputs)
    (s : S) (block : Nat) (h : ((width api v input).run s).result=.ok block)
    (htype : input.x.dtype=DType.fp32) : block≤16384 := by
  unfold width at h
  dsimp only [Bind.bind] at h
  obtain ⟨bytes,hbytes,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨next,hnext,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  have hb : bytes=input.x.dtype.bytes := by simpa [checked] using hbytes.symm
  have hblock : block=min (65536/bytes) next := by simpa [Pure.pure,Action.pure] using h.symm
  rw [hb,htype] at hblock
  norm_num [DType.fp32] at hblock
  rw [hblock]
  exact Nat.min_le_left _ _

theorem prepare_block_fp32 (api : TensorAPI world) (q : Request) (s : S) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) (htype : p.inputs.x.dtype=DType.fp32) :
    p.block≤16384 := by
  obtain ⟨_,_,v,input,output,block,_,_,_,hw,he⟩ := prepare_success api q s p h
  rw [he] at htype ⊢
  exact width_fp32 api v input _ block hw htype

/-- Triton's actual next_power_of_2(0) is zero. The source width check accepts
that value; later compilation rejects the zero-length arange. -/
theorem width_zero (api : TensorAPI world) (v : Validated) (input : Inputs)
    (s : S) (block : Nat) (h : ((width api v input).run s).result=.ok block)
    (hN : v.N=0) : block=0 := by
  unfold width at h
  dsimp only [Bind.bind] at h
  obtain ⟨bytes,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨next,hnext,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  have hn := api.nextPower_value v.N _ next (call_success _ _ _ _ hnext)
  have hblock : block=min (65536/bytes) next := by simpa [Pure.pure,Action.pure] using h.symm
  have hz : next=0 := by simpa [hN] using hn
  simpa [hz] using hblock

theorem prepare_block_zero (api : TensorAPI world) (q : Request) (s : S) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) (hN : p.validated.N=0) : p.block=0 := by
  obtain ⟨_,_,v,input,output,block,_,_,_,hw,he⟩ := prepare_success api q s p h
  rw [he] at hN ⊢
  exact width_zero api v input _ block hw hN

/-- The actual zero-block compiler/setup outcome remains an error, including
its error code, state and admission-only trace. No kernel action runs. -/
theorem zero_block_admission_failure (api : DispatchAPI world) (ha : DispatchAdmission api)
    (args : LaunchArguments) (s : S) (hz : args.prepared.block=0)
    (kernel : LaunchArguments → Action world Unit) :
    ∃ code, ((api.admit args).run s).2=.error code ∧
      ((do call .admission (api.admit args); note .execute; kernel args : Action world Unit).run s)=
        ⟨((api.admit args).run s).1,.error (.api .admission code),[.admission]⟩ := by
  cases h : ((api.admit args).run s).2 with
  | error code => exact ⟨code,rfl,admission_failure api args kernel s code h⟩
  | ok value =>
    cases value
    have hp := ha.block args s h
    omega

theorem route_admission (tensor : TensorAPI world) (device : DispatchAPI world)
    (ha : DispatchAdmission device) (kernel : LaunchArguments → Action world Unit)
    (q : Request) (s : S) (result : Returned) (p : Prepared) (args : LaunchArguments)
    (route : SuccessRoute tensor device kernel q s result p args) (hs : SelectedStorage q p) :
    NativeABI.Accepted p.validated.M p.validated.N p.block := by
  have htype : p.inputs.x.dtype=DType.fp32 := hs .x trivial p.inputs.x rfl
  have hb := prepare_block_fp32 tensor q s p route.prepared htype
  have hg := ha.grid args _ route.admitted
  have hpos := ha.block args _ route.admitted
  have hargs := arguments_bound device q p _ args route.arguments
  rw [hargs.prepared] at hg hpos
  exact ⟨hg,(prepare_success tensor q s p route.prepared).1,hpos,hb⟩

#axiomsClean width_fp32
#axiomsClean prepare_block_fp32
#axiomsClean width_zero
#axiomsClean prepare_block_zero
#axiomsClean zero_block_admission_failure
#axiomsClean route_admission
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
