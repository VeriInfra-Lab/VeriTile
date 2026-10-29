import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPriorFrame
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperMaskViews

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value Descriptor : Type} {world : World S Key Value}

def readMatrix (store : TensorStore S Descriptor Value) (s : S) (M N : Nat) (tensor : Tensor) :
    Fin M → Fin N → Value := fun i j => store.read s tensor [i.val,j.val]

def readSecondMatrix (store : TensorStore S Descriptor Value) (s : S) (M N : Nat) (tensor : Tensor) :
    Fin M → Fin N → Value := fun i j => store.read s tensor [M+i.val,j.val]

def readVector (store : TensorStore S Descriptor Value) (s : S) (N : Nat) (tensor : Tensor) :
    Fin N → Value := fun i => store.read s tensor [i.val]

/-- Both returned mask handles read the actual post-dispatch allocation, with
its true split at M. This theorem uses the successful suffix call itself. -/
theorem finish_mask_readback (store : TensorStore S Descriptor Value) (api : DispatchAPI world)
    (hs : DispatchStorage store api) (hm : MaskSplit store api) (p : Prepared) (s : S) (result : Returned)
    (hshape : ∀ mask, p.outputs.mask=some mask → p.inputs.x1.isSome=Bool.true →
      mask.shape=[2*p.validated.M,p.validated.N])
    (h : ((finish api p).run s).result=.ok result) :
    result.mask.map (readMatrix store ((finish api p).run s).state p.validated.M p.validated.N)=
      p.outputs.mask.map (readMatrix store s p.validated.M p.validated.N) ∧
    result.mask1.map (readMatrix store ((finish api p).run s).state p.validated.M p.validated.N)=
      (if p.inputs.x1.isSome then p.outputs.mask.map (readSecondMatrix store s p.validated.M p.validated.N) else none) := by
  cases hmask : p.outputs.mask with
  | none =>
    have he := finish_no_mask_run api p s hmask
    rw [he] at h ⊢
    have hr : result=returnedWithMasks p none none := Except.ok.inj h.symm
    simp [hr,returnedWithMasks,hmask]
  | some mask =>
    cases hx : p.inputs.x1 with
    | none =>
      have he := finish_no_second_run api p s hx
      rw [he] at h ⊢
      have hr : result=returnedWithMasks p p.outputs.mask none := Except.ok.inj h.symm
      simp [hr,returnedWithMasks,hx,hmask]
    | some x1 =>
      cases hsplit : ((api.split mask).run s).2 with
      | error code =>
        rw [finish_split_failure api p s mask x1 hmask hx code hsplit] at h
        cases h
      | ok pair =>
        rcases pair with ⟨first,second⟩
        have he := finish_split_run api p s mask x1 first second hmask hx hsplit
        rw [he] at h ⊢
        have hr : result=returnedWithMasks p (some first) (some second) := Except.ok.inj h.symm
        obtain ⟨_,_,_,_,_,_,_,_,hfirst,hsecond⟩ :=
          hm.views mask p.validated.M p.validated.N s first second
            (hshape mask hmask (by simp [hx])) hsplit
        simp only [hr,returnedWithMasks,hmask,hx,Option.isSome_some,Bool.true_eq,ite_true,Option.map_some]
        exact ⟨congrArg some (funext fun i => funext fun j => hfirst i j),
          congrArg some (funext fun i => funext fun j => hsecond i j)⟩

/-- The mask shape needed by the return suffix is derived from the allocation
request. It remains meaningful for zero rows or columns. -/
theorem prepared_mask_shape {α Host : Type}
    (store : TensorStore (HostState α Host) WrapperNative.Descriptor (Cell α))
    (tensor : TensorAPI (WrapperNative.world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (h : LaunchStateBound store tensor q initial p args current)
    (mask : Tensor) (he : p.outputs.mask=some mask) (hx : p.inputs.x1.isSome=Bool.true) :
    mask.shape=[2*p.validated.M,p.validated.N] := by
  have ho := allocated_of_some h.outputs.2.2.2.2.2.2 mask he
  simpa [outputRequests,hx] using ho.shape

#axiomsClean finish_mask_readback
#axiomsClean prepared_mask_shape
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
