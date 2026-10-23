import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperFreshAllocation
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperOutputAllocations

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

def outputTensors (output : Outputs) : List Tensor :=
  ((((([output.y]++optionTensors output.y1)++optionTensors output.residualOut)++
    optionTensors output.mean)++[output.rstd])++optionTensors output.seeds)++optionTensors output.mask

/-- Physical storage separation follows the seven actual source allocation
calls. Existing resolved inputs may overlap each other. Every new output is
disjoint from all earlier live tensors and all other newly allocated outputs. -/
theorem allocateOutputs_storage (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hs : StorageAPI store api) (hf : FreshAllocationAPI store api)
    (q : Request) (v : Validated) (input : Inputs) (s : S) (output : Outputs)
    (h : ((allocateOutputs api q v input).run s).result=.ok output) :
    AllocationBatch store s (outputTensors output) ((allocateOutputs api q v input).run s).state := by
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
  have by0 : AllocationBatch store s ([]++[y]) sy :=
    allocate_batch store api hs hf .y req.y s s [] (AllocationBatch.empty store s) y hy
  have byt : AllocationBatch store s [y] st := by0.transfer (persists_checkStride store .y y sy)
  have by1 := allocateOptional_batch store api hs hf .y1 input.weight1.isSome req.y1 Bool.true
    s st [y] byt y1 hy1
  have br := allocateOptional_batch store api hs hf .residualOut (needsResidual q v input) req.residualOut Bool.true
    s sy1 _ by1 res hres
  have bm := allocateOptional_batch store api hs hf .mean (!q.rms) req.mean Bool.false
    s sr _ br mean hmean
  have bs := allocate_batch store api hs hf .rstd req.rstd s sm _ bm rstd hrstd
  have be := allocateOptional_batch store api hs hf .seeds q.dropout req.seeds Bool.false
    s ss _ bs seeds hseeds
  have bf := allocateOptional_batch store api hs hf .mask (q.returnMask && q.dropout) req.mask Bool.false
    s se _ be mask hmask
  rw [ho,hfinal]
  exact bf

#axiomsClean allocateOutputs_storage
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
