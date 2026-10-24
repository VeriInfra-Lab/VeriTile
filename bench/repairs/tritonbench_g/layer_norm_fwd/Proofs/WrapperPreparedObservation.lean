import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperReturnViews

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
variable {S Key α Descriptor : Type} {world : World S Key (Cell α)}

structure PreparedPresent (store : TensorStore S Descriptor (Cell α)) (p : Prepared) (s : S) : Prop where
  y : store.present s p.outputs.y
  y1 : OptionPresent store s p.outputs.y1
  mean : OptionPresent store s p.outputs.mean
  rstd : store.present s p.outputs.rstd
  residualOut : OptionPresent store s p.outputs.residualOut
  original : store.present s p.inputs.originalX
  seeds : OptionPresent store s p.outputs.seeds
  mask : OptionPresent store s p.outputs.mask

theorem prepared_present (store : TensorStore S WrapperNative.Descriptor (Cell α))
    (tensor : TensorAPI world) (q : Request) (initial current : S) (p : Prepared) (args : LaunchArguments)
    (hp : RequestPresent store q initial) (h : LaunchStateBound store tensor q initial p args current) :
    PreparedPresent store p current :=
  ⟨h.outputs.1.present,optional_allocated_present h.outputs.2.1,
    optional_allocated_present h.outputs.2.2.2.1,h.outputs.2.2.2.2.1.present,
    optional_allocated_present h.outputs.2.2.1,
    h.inputs.original.symm ▸ (h.preserved q.x hp.x).1,
    optional_allocated_present h.outputs.2.2.2.2.2.1,optional_allocated_present h.outputs.2.2.2.2.2.2⟩

theorem PreparedPresent.transfer {store : TensorStore S Descriptor (Cell α)} {p : Prepared} {s t : S}
    (h : PreparedPresent store p s) (he : store.Extends s t) : PreparedPresent store p t :=
  ⟨(he _ h.y).1,h.y1.transfer he,h.mean.transfer he,(he _ h.rstd).1,h.residualOut.transfer he,
    (he _ h.original).1,h.seeds.transfer he,h.mask.transfer he⟩

theorem PreparedPresent.fallback {store : TensorStore S Descriptor (Cell α)} {p : Prepared} {s : S}
    (h : PreparedPresent store p s) : store.present s (p.outputs.residualOut.getD p.inputs.originalX) := by
  cases he : p.outputs.residualOut with
  | none => simpa [he] using h.original
  | some value => simpa [he,OptionPresent] using h.residualOut

def preparedObserve (store : TensorStore S Descriptor (Cell α)) (s : S) (p : Prepared) (K : Nat) :
    TypedSource.Result α p.validated.M p.validated.N K where
  y := readMatrix store s p.validated.M p.validated.N p.outputs.y
  y1 := p.outputs.y1.map (readMatrix store s p.validated.M p.validated.N)
  mean := p.outputs.mean.map (readVector store s p.validated.M)
  rstd := readVector store s p.validated.M p.outputs.rstd
  residual := readMatrix store s p.validated.M p.validated.N (p.outputs.residualOut.getD p.inputs.originalX)
  seeds := p.outputs.seeds.map (readVector store s K)
  mask := p.outputs.mask.map (readMatrix store s p.validated.M p.validated.N)
  mask1 := if p.inputs.x1.isSome then p.outputs.mask.map (readSecondMatrix store s p.validated.M p.validated.N) else none

def returnedObserve (store : TensorStore S Descriptor (Cell α)) (s : S) (p : Prepared) (K : Nat)
    (result : Returned) : TypedSource.Result α p.validated.M p.validated.N K where
  y := readMatrix store s p.validated.M p.validated.N result.y
  y1 := result.y1.map (readMatrix store s p.validated.M p.validated.N)
  mean := result.mean.map (readVector store s p.validated.M)
  rstd := readVector store s p.validated.M result.rstd
  residual := readMatrix store s p.validated.M p.validated.N result.residual
  seeds := result.seeds.map (readVector store s K)
  mask := result.mask.map (readMatrix store s p.validated.M p.validated.N)
  mask1 := result.mask1.map (readMatrix store s p.validated.M p.validated.N)

theorem readMatrix_extends (store : TensorStore S Descriptor (Cell α)) (s t : S)
    (he : store.Extends s t) (M N : Nat) (value : Tensor) (hp : store.present s value) :
    readMatrix store t M N value=readMatrix store s M N value :=
  funext fun i => funext fun j => (he value hp).2.2 [i.val,j.val]

theorem readVector_extends (store : TensorStore S Descriptor (Cell α)) (s t : S)
    (he : store.Extends s t) (N : Nat) (value : Tensor) (hp : store.present s value) :
    readVector store t N value=readVector store s N value :=
  funext fun i => (he value hp).2.2 [i.val]

theorem optional_read_extends {β : Type} (store : TensorStore S Descriptor (Cell α)) (s t : S)
    (he : store.Extends s t) (value : Option Tensor) (hp : OptionPresent store s value)
    (read : S → Tensor → β)
    (hr : ∀ tensor, store.present s tensor → read t tensor=read s tensor) :
    value.map (read t)=value.map (read s) := by
  cases value with
  | none => rfl
  | some value => exact congrArg some (hr value hp)

/-- The eight values in the actual returned tuple are the prepared tensor
coordinates before the suffix, with mask halves composed through the real
split operation. No kernel-value premise is used. -/
theorem finish_observe (store : TensorStore S Descriptor (Cell α)) (api : DispatchAPI world)
    (hs : DispatchStorage store api) (hm : MaskSplit store api) (p : Prepared) (s : S)
    (hp : PreparedPresent store p s) (K : Nat) (result : Returned)
    (hshape : ∀ mask, p.outputs.mask=some mask → p.inputs.x1.isSome=Bool.true →
      mask.shape=[2*p.validated.M,p.validated.N])
    (h : ((finish api p).run s).result=.ok result) :
    returnedObserve store ((finish api p).run s).state p K result=preparedObserve store s p K := by
  have he := persists_finish store api hs p s
  obtain ⟨hy,hy1,hmean,hrstd,hres,hseeds⟩ := finish_return_identity api p s result h
  obtain ⟨hmask,hmask1⟩ := finish_mask_readback store api hs hm p s result hshape h
  apply TypedSource.Result.ext
  · simpa only [returnedObserve,preparedObserve,hy] using readMatrix_extends store s _ he _ _ _ hp.y
  · simpa only [returnedObserve,preparedObserve,hy1] using
      optional_read_extends store s _ he p.outputs.y1 hp.y1
        (fun s tensor => readMatrix store s p.validated.M p.validated.N tensor)
        (fun tensor hp => readMatrix_extends store s _ he _ _ tensor hp)
  · simpa only [returnedObserve,preparedObserve,hmean] using
      optional_read_extends store s _ he p.outputs.mean hp.mean
        (fun s tensor => readVector store s p.validated.M tensor)
        (fun tensor hp => readVector_extends store s _ he _ tensor hp)
  · simpa only [returnedObserve,preparedObserve,hrstd] using readVector_extends store s _ he _ _ hp.rstd
  · simpa only [returnedObserve,preparedObserve,hres] using readMatrix_extends store s _ he _ _ _ hp.fallback
  · simpa only [returnedObserve,preparedObserve,hseeds] using
      optional_read_extends store s _ he p.outputs.seeds hp.seeds
        (fun s tensor => readVector store s K tensor)
        (fun tensor hp => readVector_extends store s _ he _ tensor hp)
  · exact hmask
  · exact hmask1

#axiomsClean prepared_present
#axiomsClean PreparedPresent.transfer
#axiomsClean PreparedPresent.fallback
#axiomsClean readMatrix_extends
#axiomsClean readVector_extends
#axiomsClean optional_read_extends
#axiomsClean finish_observe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
