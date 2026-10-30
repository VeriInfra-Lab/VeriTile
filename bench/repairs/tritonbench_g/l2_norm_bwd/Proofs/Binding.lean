import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Execution
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Contract

namespace VeriTile.Bench.Repairs.L2NormBwd.Host
open VeriTile.Triton Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

def checkedRun {M α β : Type} (ops : Ops M α β) (red : Reduction α) (rt : Runtime M)
    (x dy : TensorMetadata) (input gradient : View) (eps : α) (s : World M) : Except InputError (Result M View) := do
  checkInputs (.tensor x) (.tensor dy)
  return run ops red rt ⟨input,gradient⟩ eps s

theorem checked_domain {M : Type} (x dy : TensorMetadata) (input gradient : View)
    (s : World M) (bx : TensorBinding x input) (bd : TensorBinding dy gradient)
    (lx : s.live input.storage) (ld : s.live gradient.storage) : InputDomain ⟨input,gradient⟩ s := by
  refine ⟨bx.backed,bd.backed,?_⟩
  intro old ho
  simp only [Call.caller,List.mem_cons,List.not_mem_nil,or_false] at ho
  rcases ho with rfl | rfl
  · exact lx
  · exact ld

theorem checked_run {M α β : Type} (ops : Ops M α β) (red : Reduction α) (rt : Runtime M)
    (x dy : TensorMetadata) (input gradient : View) (eps : α) (s : World M) (h : InputContract x dy) :
    checkedRun ops red rt x dy input gradient eps s=.ok (run ops red rt ⟨input,gradient⟩ eps s) := by
  simp [checkedRun,checkInputs,(checkTensorInputs_iff x dy).mpr h]


/-- Both input views retain their original backing allocations. Singleton
axes may keep arbitrary metadata strides; no contiguous conversion is run. -/
theorem normalized_aliases {M α β : Type} (ops : Ops M α β) (neg : Negation α) (rt : Runtime M)
    (lib : Library ops neg rt) (call : Call) (s : World M) (q : Request) (after : World M)
    (success : normalize rt call s=.ok q after) :
    q.x.view.storage=call.x.storage ∧ q.dy.view.storage=call.dy.storage := by
  unfold normalize at success
  split at success
  · cases success
  · cases hx : rt.perform (.flatten call.x call.cols) s with
    | error why failed => simp [hx,Result.bind] at success
    | ok x s1 =>
      rw [hx] at success
      simp only [Result.bind] at success
      obtain ⟨_,_,ax,_⟩:=lib.flattenAlias _ _ _ _ _ hx
      cases hd : rt.perform (.flatten call.dy call.cols) s1 with
      | error why failed => simp [hd,Result.bind] at success
      | ok dy s2 =>
        rw [hd] at success
        simp only [Result.bind] at success
        obtain ⟨_,_,ad,_⟩:=lib.flattenAlias _ _ _ _ _ hd
        simp only [prepareDY,Result.bind] at success
        cases he : rt.perform (.emptyLike x) s2 with
        | error why failed => simp [he,Result.bind] at success
        | ok dx s3 =>
          rw [he] at success
          simp only [Result.bind] at success
          split at success
          · cases success
          · split at success
            · cases success
            · split at success
              · cases success
              · obtain ⟨rfl,rfl⟩:=Result.ok.inj success
                exact ⟨ax.storage,ad.storage⟩

/-- Same-call direct input handles and the actual returned output allocation. -/
structure DirectPath {M α β : Type} (ops : Ops M α β) (red : Reduction α) (rt : Runtime M)
    (call : Call) (eps : α) (initial : World M) (output : View) (after : World M) where
  path : WholePath ops red rt call eps initial output after
  inputStorage : path.request.x.view.storage=call.x.storage
  gradientStorage : path.request.dy.view.storage=call.dy.storage
  kernelInput : path.context.body.x=path.request.x
  kernelGradient : path.context.body.dy=path.request.dy
  returnedStorage : output.storage=path.request.dx.view.storage

theorem direct_path {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call) (eps : α) (initial after : World M)
    (output : View) (domain : InputDomain call initial)
    (success : run ops red rt call eps initial=.ok output after) :
    Nonempty (DirectPath ops red rt call eps initial output after) := by
  obtain ⟨path⟩:=successful_path ops neg red rt lib call eps initial after output domain success
  obtain ⟨ax,ad⟩:=normalized_aliases ops neg rt lib call initial path.request path.prepared path.normalized
  obtain ⟨_,_,xn,xalias,_⟩:=lib.resolveAlias _ _ _ _ path.context.body.resolvedX
  obtain ⟨_,_,dn,dalias,_⟩:=lib.resolveAlias _ _ _ _ path.context.body.resolvedDY
  have hx:=(lib.resolveIdentity _ _ _ _ path.context.body.resolvedX (xalias.negative.symm.trans xn)).1
  have hd:=(lib.resolveIdentity _ _ _ _ path.context.body.resolvedDY (dalias.negative.symm.trans dn)).1
  have returned:=(lib.returnAlias _ _ _ _ _ path.returned).2.storage
  exact ⟨⟨path,ax,ad,hx,hd,returned⟩⟩

#axiomsClean normalized_aliases
#axiomsClean direct_path

#axiomsClean checked_domain
#axiomsClean checked_run
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Host
