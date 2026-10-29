import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperAllocated

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value Descriptor : Type} {world : World S Key Value}

structure OutputRequests where
  y : Allocation
  y1 : Allocation
  residualOut : Allocation
  mean : Allocation
  rstd : Allocation
  seeds : Allocation
  mask : Allocation

/-- The actual sequential allocation requests. Later freshness lists contain
every earlier returned object, exactly as in `allocateOutputs`. -/
def outputRequests (api : TensorAPI world) (q : Request) (v : Validated) (input : Inputs) (output : Outputs) : OutputRequests :=
  let used0 := api.callerObjects++input.objects
  let used1 := used0++[output.y.object]
  let used2 := used1++objectIds [output.y1]
  let used3 := used2++objectIds [output.residualOut]
  let used4 := used3++objectIds [output.mean]
  let used5 := used4++[output.rstd.object]
  let used6 := used5++objectIds [output.seeds]
  let count := if input.x1.isSome then 2*v.M else v.M
  { y := ⟨input.x.shape,q.outDtype.getD input.x.dtype,input.x.device,some input.x,Bool.false,used0⟩
    y1 := ⟨output.y.shape,output.y.dtype,output.y.device,some output.y,Bool.false,used1⟩
    residualOut := ⟨[v.M,v.N],v.residualDtype.getD input.x.dtype,input.x.device,none,Bool.false,used2⟩
    mean := ⟨[v.M],.fp32,input.x.device,none,Bool.false,used3⟩
    rstd := ⟨[v.M],.fp32,input.x.device,none,Bool.false,used4⟩
    seeds := ⟨[count],.int64,input.x.device,none,Bool.true,used5⟩
    mask := ⟨[count,v.N],.bool,input.x.device,none,Bool.false,used6⟩ }

def OutputsAllocated (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (q : Request) (v : Validated) (input : Inputs) (output : Outputs) (s : S) : Prop :=
  let req := outputRequests api q v input output
  Allocated store req.y output.y s ∧
  OptionAllocated store input.weight1.isSome req.y1 output.y1 s ∧
  OptionAllocated store (needsResidual q v input) req.residualOut output.residualOut s ∧
  OptionAllocated store (!q.rms) req.mean output.mean s ∧
  Allocated store req.rstd output.rstd s ∧
  OptionAllocated store q.dropout req.seeds output.seeds s ∧
  OptionAllocated store (q.returnMask && q.dropout) req.mask output.mask s

theorem OutputsAllocated.transfer {store : TensorStore S Descriptor Value} {api : TensorAPI world}
    {q : Request} {v : Validated} {input : Inputs} {output : Outputs} {s t : S}
    (h : OutputsAllocated store api q v input output s) (he : store.Extends s t) :
    OutputsAllocated store api q v input output t :=
  ⟨h.1.transfer he,h.2.1.transfer he,h.2.2.1.transfer he,h.2.2.2.1.transfer he,
    h.2.2.2.2.1.transfer he,h.2.2.2.2.2.1.transfer he,h.2.2.2.2.2.2.transfer he⟩

theorem allocateOutputs_allocated (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (q : Request) (v : Validated) (input : Inputs) (s : S) (output : Outputs)
    (h : ((allocateOutputs api q v input).run s).result=.ok output) :
    OutputsAllocated store api q v input output ((allocateOutputs api q v input).run s).state := by
  have hh := h
  unfold allocateOutputs at hh
  dsimp only [Bind.bind] at hh
  obtain ⟨y,hy,hh,h₀,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨_,_,hh,h₁,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨y1,hy1,hh,h₂,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨res,hres,hh,h₃,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨mean,hmean,hh,h₄,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨rstd,hrstd,hh,h₅,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨seeds,hseeds,hh,h₆,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨mask,hmask,hh,h₇,_⟩ := bind_success _ _ _ _ hh
  let out : Outputs := ⟨y,y1,res,mean,rstd,seeds,mask⟩
  have ho : output=out := by simpa [out,Pure.pure,Action.pure] using hh.symm
  let req := outputRequests api q v input out
  let sy := ((call (.allocate .y) (api.allocate .y req.y)).run s).state
  let st := ((checkStride (world:=world) .y y).run sy).state
  let sy1 := ((allocateOptional api .y1 input.weight1.isSome req.y1 Bool.true).run st).state
  let sr := ((allocateOptional api .residualOut (needsResidual q v input) req.residualOut Bool.true).run sy1).state
  let sm := ((allocateOptional api .mean (!q.rms) req.mean Bool.false).run sr).state
  let ss := ((call (.allocate .rstd) (api.allocate .rstd req.rstd)).run sm).state
  let se := ((allocateOptional api .seeds q.dropout req.seeds Bool.false).run ss).state
  let sf := ((allocateOptional api .mask (q.returnMask && q.dropout) req.mask Bool.false).run se).state
  have hfinal : ((allocateOutputs api q v input).run s).state=sf := by
    unfold allocateOutputs
    dsimp only [Bind.bind]
    rw [h₀,h₁,h₂,h₃,h₄,h₅,h₆,h₇]
    rfl
  have et : store.Extends sy st := persists_checkStride store .y y sy
  have ey1 : store.Extends st sy1 := persists_allocateOptional store api hc .y1 _ req.y1 Bool.true st
  have er : store.Extends sy1 sr := persists_allocateOptional store api hc .residualOut _ req.residualOut Bool.true sy1
  have em : store.Extends sr sm := persists_allocateOptional store api hc .mean _ req.mean Bool.false sr
  have es : store.Extends sm ss := hc.allocate_preserves .rstd req.rstd sm
  have ee : store.Extends ss se := persists_allocateOptional store api hc .seeds _ req.seeds Bool.false ss
  have ef : store.Extends se sf := persists_allocateOptional store api hc .mask _ req.mask Bool.false se
  have vy : Allocated store req.y y sy := allocate_allocated store api hc .y req.y s y hy
  have vy1 : OptionAllocated store input.weight1.isSome req.y1 y1 sy1 :=
    allocateOptional_allocated store api hc .y1 _ req.y1 Bool.true st y1 hy1
  have vr : OptionAllocated store (needsResidual q v input) req.residualOut res sr :=
    allocateOptional_allocated store api hc .residualOut _ req.residualOut Bool.true sy1 res hres
  have vm : OptionAllocated store (!q.rms) req.mean mean sm :=
    allocateOptional_allocated store api hc .mean _ req.mean Bool.false sr mean hmean
  have vs : Allocated store req.rstd rstd ss := allocate_allocated store api hc .rstd req.rstd sm rstd hrstd
  have ve : OptionAllocated store q.dropout req.seeds seeds se :=
    allocateOptional_allocated store api hc .seeds _ req.seeds Bool.false ss seeds hseeds
  have vf : OptionAllocated store (q.returnMask && q.dropout) req.mask mask sf :=
    allocateOptional_allocated store api hc .mask _ req.mask Bool.false se mask hmask
  rw [ho,hfinal]
  exact ⟨vy.transfer (et.trans (ey1.trans (er.trans (em.trans (es.trans (ee.trans ef)))))),
    vy1.transfer (er.trans (em.trans (es.trans (ee.trans ef)))),
    vr.transfer (em.trans (es.trans (ee.trans ef))),vm.transfer (es.trans (ee.trans ef)),
    vs.transfer (ee.trans ef),ve.transfer ef,vf⟩

#axiomsClean OutputsAllocated.transfer
#axiomsClean allocateOutputs_allocated
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
