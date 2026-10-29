import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.RealPreparation

/-! Same-call returned handles read the Real physical memory through their
actual descriptors. The only additional split boundary is local coordinate
address preservation, including offsets and arbitrary valid row strides. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.RealRevision
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
variable {α Host : Type} {caller : Address → Prop}

structure SplitCoordinates (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (device : DispatchAPI (world (α:=α) (Host:=Host) caller)) : Prop where
  views : ∀ (mask : Tensor) (M N : Nat) (s : HostState α Host) (first second : Tensor),
    mask.shape=[2*M,N] → ((device.split mask).run s).2=.ok (first,second) →
    first.negative=mask.negative ∧ second.negative=mask.negative ∧
    (∀ (i : Fin M) (j : Fin N),
      tensorAddress store ((device.split mask).run s).1 first [i.val,j.val]=tensorAddress store s mask [i.val,j.val]) ∧
    (∀ (i : Fin M) (j : Fin N),
      tensorAddress store ((device.split mask).run s).1 second [i.val,j.val]=tensorAddress store s mask [M+i.val,j.val])

def readStore (store : TensorStore (HostState α Host) Descriptor (Cell α)) (memory : RealMemory) :
    TensorStore (HostState α Host) Descriptor MemCell :=
  ⟨store.present,store.descriptor,fun s value index => logicalRead store s memory value index⟩

theorem readStore_extends
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (memory : RealMemory)
    (s t : HostState α Host) (he : store.Extends s t) : (readStore store memory).Extends s t := by
  intro value hp
  obtain ⟨ht,hd,_⟩ := he value hp
  refine ⟨ht,hd,?_⟩
  intro index
  simp only [readStore,logicalRead,rawRead,tensorAddress,hd]

theorem readStore_host
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hh : HostDescriptors store)
    (memory : RealMemory) (s t : HostState α Host) (he : s.host=t.host) :
    (readStore store memory).Extends s t := by
  intro value hp
  refine ⟨hh.present s t he value hp,(hh.descriptor s t he value).symm,?_⟩
  intro index
  simp only [readStore,logicalRead,rawRead,tensorAddress,hh.descriptor t s he.symm value]

theorem readStore_dispatch
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (memory : RealMemory)
    (device : DispatchAPI (world (α:=α) (Host:=Host) caller)) (hd : DispatchStorage store device) :
    DispatchStorage (readStore store memory) (revalueDispatch device) :=
  ⟨fun n s => readStore_extends store memory s _ (hd.enter n s),
    fun n s => readStore_extends store memory s _ (hd.leave n s),
    fun n s => readStore_extends store memory s _ (hd.select n s),
    fun n s => readStore_extends store memory s _ (hd.admit n s),
    fun n s => readStore_extends store memory s _ (hd.split n s)⟩

theorem readStore_split
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (memory : RealMemory)
    (device : DispatchAPI (world (α:=α) (Host:=Host) caller)) (hm : MaskSplit store device)
    (hc : SplitCoordinates store device) : MaskSplit (readStore store memory) (revalueDispatch device) := by
  constructor
  intro mask M N s first second hshape hsuccess
  obtain ⟨ha,hb,hc₁,hd,he,hf,hp,hq,_,_⟩ := hm.views mask M N s first second hshape hsuccess
  obtain ⟨hn,hk,hfirst,hsecond⟩ := hc.views mask M N s first second hshape hsuccess
  refine ⟨ha,hb,hc₁,hd,he,hf,hp,hq,?_,?_⟩
  · intro i j
    have hsrc := coordinate_matrix mask (2*M) N i.val j.val hshape (by omega) j.isLt
    have hdst := coordinate_matrix first M N i.val j.val ha i.isLt j.isLt
    simp only [readStore,logicalRead,if_pos hsrc,if_pos hdst,rawRead,revalueDispatch,revalueLocal,hn,hfirst i j]
  · intro i j
    have hsrc := coordinate_matrix mask (2*M) N (M+i.val) j.val hshape (by omega) j.isLt
    have hdst := coordinate_matrix second M N i.val j.val hb i.isLt j.isLt
    simp only [readStore,logicalRead,if_pos hsrc,if_pos hdst,rawRead,revalueDispatch,revalueLocal,hk,hsecond i j]

def preparedObservation (store : TensorStore (HostState α Host) Descriptor MemCell)
    (s : HostState α Host) (p : Prepared) (K : Nat) : HostReturn.Result p.validated.M p.validated.N K where
  y := readMatrix store s p.validated.M p.validated.N p.outputs.y
  y1 := p.outputs.y1.map (readMatrix store s p.validated.M p.validated.N)
  mean := p.outputs.mean.map (readVector store s p.validated.M)
  rstd := readVector store s p.validated.M p.outputs.rstd
  residual := readMatrix store s p.validated.M p.validated.N (p.outputs.residualOut.getD p.inputs.originalX)
  seeds := p.outputs.seeds.map (readVector store s K)
  mask := p.outputs.mask.map (readMatrix store s p.validated.M p.validated.N)
  mask1 := if p.inputs.x1.isSome then p.outputs.mask.map (readSecondMatrix store s p.validated.M p.validated.N) else none

def returnedObservation (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (s : HostState α Host) (memory : RealMemory) (p : Prepared) (K : Nat) (result : Returned) :
    HostReturn.Result p.validated.M p.validated.N K where
  y := readMatrix (readStore store memory) s p.validated.M p.validated.N result.y
  y1 := result.y1.map (readMatrix (readStore store memory) s p.validated.M p.validated.N)
  mean := result.mean.map (readVector (readStore store memory) s p.validated.M)
  rstd := readVector (readStore store memory) s p.validated.M result.rstd
  residual := readMatrix (readStore store memory) s p.validated.M p.validated.N result.residual
  seeds := result.seeds.map (readVector (readStore store memory) s K)
  mask := result.mask.map (readMatrix (readStore store memory) s p.validated.M p.validated.N)
  mask1 := result.mask1.map (readMatrix (readStore store memory) s p.validated.M p.validated.N)

theorem matrix_extends (store : TensorStore (HostState α Host) Descriptor MemCell)
    (s t : HostState α Host) (he : store.Extends s t) (M N : Nat) (value : Tensor)
    (hp : store.present s value) : readMatrix store t M N value=readMatrix store s M N value :=
  funext fun i => funext fun j => (he value hp).2.2 _

theorem vector_extends (store : TensorStore (HostState α Host) Descriptor MemCell)
    (s t : HostState α Host) (he : store.Extends s t) (N : Nat) (value : Tensor)
    (hp : store.present s value) : readVector store t N value=readVector store s N value :=
  funext fun i => (he value hp).2.2 _

theorem optional_extends {β : Type} (store : TensorStore (HostState α Host) Descriptor MemCell)
    (s t : HostState α Host) (value : Option Tensor) (hp : OptionPresent store s value)
    (read : HostState α Host → Tensor → β)
    (hr : ∀ tensor, store.present s tensor → read t tensor=read s tensor) :
    value.map (read t)=value.map (read s) := by
  cases value with
  | none => rfl
  | some value => exact congrArg some (hr value hp)

theorem option_read_present
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (memory : RealMemory)
    (s : HostState α Host) (value : Option Tensor) (hp : OptionPresent store s value) :
    OptionPresent (readStore store memory) s value := by
  cases value with
  | none => trivial
  | some value => exact hp

theorem finish_observation
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (memory : RealMemory)
    (device : DispatchAPI (world (α:=α) (Host:=Host) caller)) (hd : DispatchStorage store device)
    (hm : MaskSplit store device) (hc : SplitCoordinates store device)
    (p : Prepared) (s : HostState α Host) (hp : PreparedPresent store p s) (K : Nat) (result : Returned)
    (hshape : ∀ mask, p.outputs.mask=some mask → p.inputs.x1.isSome=Bool.true → mask.shape=[2*p.validated.M,p.validated.N])
    (h : ((finish device p).run s).result=.ok result) :
    returnedObservation store ((finish device p).run s).state memory p K result=
      preparedObservation (readStore store memory) s p K := by
  let rs := readStore store memory
  let api := revalueDispatch device
  have hs := readStore_dispatch store memory device hd
  have hmask := readStore_split store memory device hm hc
  have he := persists_finish rs api hs p s
  obtain ⟨hy,hy1,hmean,hrstd,hres,hseeds⟩ := finish_return_identity device p s result h
  have h' : ((finish api p).run s).result=.ok result := by rw [finish_run]; exact h
  obtain ⟨hmask,hmask1⟩ := finish_mask_readback rs api hs hmask p s result hshape h'
  rw [finish_run] at he hmask hmask1
  apply HostReturn.Result.ext
  · simpa only [returnedObservation,preparedObservation,hy] using matrix_extends rs s _ he _ _ _ hp.y
  · simpa only [returnedObservation,preparedObservation,hy1] using optional_extends rs s _ p.outputs.y1 (option_read_present store memory s _ hp.y1)
      (fun s t => readMatrix rs s p.validated.M p.validated.N t) (fun t hp => matrix_extends rs s _ he _ _ t hp)
  · simpa only [returnedObservation,preparedObservation,hmean] using optional_extends rs s _ p.outputs.mean (option_read_present store memory s _ hp.mean)
      (fun s t => readVector rs s p.validated.M t) (fun t hp => vector_extends rs s _ he _ t hp)
  · simpa only [returnedObservation,preparedObservation,hrstd] using vector_extends rs s _ he _ _ hp.rstd
  · simpa only [returnedObservation,preparedObservation,hres] using matrix_extends rs s _ he _ _ _ hp.fallback
  · simpa only [returnedObservation,preparedObservation,hseeds] using optional_extends rs s _ p.outputs.seeds (option_read_present store memory s _ hp.seeds)
      (fun s t => readVector rs s K t) (fun t hp => vector_extends rs s _ he _ t hp)
  · exact hmask
  · exact hmask1

#axiomsClean finish_observation
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.RealRevision
