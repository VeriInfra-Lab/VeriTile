import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeReturn

namespace VeriTile.Bench.Repairs.AttnFwdTriton.Checked
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.LayerNormFwd
open Verified.PhysicalCells (Role Layout width FreshBytes)
open TypedPhysical (Physical pull)
open TypedExecution NativeGraph
noncomputable section
set_option maxHeartbeats 2000000

theorem input_dtype (p : Config) (objects : Role→TensorDescriptor)
    (hinput : inputChecks (typedGeometry p) (objects .query).metadata (objects .key).metadata
      (objects .value).metadata (objects .queryScale).metadata (objects .keyScale).metadata=Bool.true)
    (r : Role) (hr : r≠Role.output) : (objects r).dtype=.fp32 := by
  simp only [inputChecks,Bool.and_eq_true,List.all_eq_true] at hinput
  have hm : (objects r).metadata∈[(objects .query).metadata,(objects .key).metadata,
      (objects .value).metadata,(objects .queryScale).metadata,(objects .keyScale).metadata] := by
    cases r <;> simp_all
  have hc:=hinput.1 _ hm
  simp only [commonChecks,decide_eq_true_eq] at hc
  have ht:=(objects r).metadataType
  rw [hc.2.1] at ht
  exact of_decide_eq_true ht.symm

/-- Complete native return from the caller's actual FP32 bytes. Successful
host checks supply all arithmetic/alignment bounds. Ordinary allocation and
Tensor interpretation supply identity and views; neither supplies outputs.

The pinned SM90 source-to-instruction graph remains the ordinary compiler/ISA
contract evidenced by the profile audit. This theorem proves the selected
graph's execution, exact BF16 return and full physical byte frame; it makes no
compiler-internals or unconditional native-to-Real numerical claim. -/
theorem whole_native (ops : Operations) (p : Config) (a : Layout)
    (objects : Role→TensorDescriptor) (views : Views a p (pointers objects))
    (memory : ByteMemory) (template : State Word)
    (hinput : inputChecks (typedGeometry p) (objects .query).metadata (objects .key).metadata
      (objects .value).metadata (objects .queryScale).metadata (objects .keyScale).metadata=Bool.true)
    (hl : launchChecks (typedGeometry p) (objects .queryScale).metadata (objects .keyScale).metadata=Bool.true)
    (hp : pointerChecks (typedGeometry p) (pointers objects)=Bool.true)
    (binding : OutputBinding p objects) (stage : p.stage=3) (nonoverlap : NonOverlapping p)
    (fresh : FreshBytes a (fun k=>Window p k.1 k.2)) (allocationFresh : FreshInputs objects) :
    NativeLaunch.CudaLaunchable (launchShape p) (HostShape.blockD p.headDim) ∧
    (∀ r, r≠Role.output → (objects r).dtype=.fp32) ∧
    (∀ z h n d r, r≠Role.output → roleActive p z h n d r →
      TypedPhysical.read a.toLayout (callerCells a memory) (roleKey p z h n d r)=
        Cell.mk .fp32 (descriptorRead objects memory r z h n d)) ∧
    (∀ r n, n<envelope (typedGeometry p) →
      (objects r).dataPointer%width r=0 ∧ (objects r).dataPointer+width r*n+width r<2^64) ∧
    ∃ final, ∃ graph : Physical Expr a.toLayout, ∃ physical : Physical Word a.toLayout,
    ∃ result : ReturnedBytes,
      Nonempty (GridRun backend p (HostShape.blockD p.headDim) Bool.false
        (pull a.toLayout (liftPhysical (callerCells a memory)) (liftState template)) final) ∧
      pull a.toLayout graph final=final ∧ physical=evaluatePhysical ops graph ∧
      result=finishBytes p objects
        (fun c=>BitVec.ofNat 16 ((physical (outputCell a p (pointers objects) views c)).read .bf16).toNat) memory ∧
      result.tensor=objects .output ∧
      result.tensor.dtype=.bf16 ∧ result.tensor.metadata.shape=[p.batches,p.heads,p.context,p.headDim] ∧
      result.tensor.metadata.strides=((typedGeometry p).stride .output).raw ∧
      result.tensor.metadata.device=(objects .query).metadata.device ∧
      (∀ c : Coordinate p,
        result.read p c=BitVec.ofNat 16
          ((NativeGraph.reference ops p (HostShape.blockD p.headDim) a (callerCells a memory) template c).read .bf16).toNat) ∧
      (∀ c : Coordinate p, ∃ w,
        physical (outputCell a p (pointers objects) views c)=Cell.mk .bf16 w ∧
        result.read p c=BitVec.ofNat 16 w.toNat) ∧
      (∀ address : Fin (2^64), (∀ i,address.val≠outputByte p objects i) →
        result.memory address=memory address) ∧
      (∀ r, r≠Role.output → ∀ address : Fin (2^64),
        (objects r).allocationBase≤address.val →
        address.val<(objects r).allocationBase+(objects r).allocationBytes →
        result.memory address=memory address) ∧
      (∀ cell : Fin a.capacity,
        (∀ k,Window p k.1 k.2 → a.address k≠cell.val) → physical cell=callerCells a memory cell) := by
  obtain ⟨_,_,hlaunch,hpointer,final,graph,physical,hgrid,hpull,heval,hread,hframe⟩:=
    whole_native_cells ops p a (pointers objects) views
      (objects .query).metadata (objects .key).metadata (objects .value).metadata
      (objects .queryScale).metadata (objects .keyScale).metadata hinput hl hp
      (callerCells a memory) template stage nonoverlap fresh
  let word : Coordinate p→BF16Word:=fun c=>BitVec.ofNat 16
    ((physical (outputCell a p (pointers objects) views c)).read .bf16).toNat
  let result:=finishBytes p objects word memory
  refine ⟨hlaunch,input_dtype p objects hinput,?_,hpointer,
    final,graph,physical,result,hgrid,hpull,heval,rfl,rfl,binding.dtype,binding.shape,
    binding.strides,binding.device,?_,?_,?_,?_,hframe⟩
  · exact fun z h n d r hr ha=>caller_inputs p a objects views memory hinput z h n d r hr ha
  · intro c
    rw [finish_read p objects nonoverlap binding word memory hp c]
    unfold word
    rw [hread c]
  · intro c
    obtain ⟨w,hw⟩:=reference_bf16 ops p (HostShape.blockD p.headDim) a
      (callerCells a memory) template (HostShape.blockD_covers _) c
    refine ⟨w,(hread c).trans hw,?_⟩
    rw [finish_read p objects nonoverlap binding word memory hp c]
    unfold word
    rw [hread c,hw]
    rfl
  · exact fun address h=>finish_frame p objects word memory address h
  · exact fun r hr address hlo hhi=>finish_inputs_frame p objects binding allocationFresh
      word memory r hr address hlo hhi

/-- Any separately established numerical relation for the precise local native
graph transports to a read through the actual returned Tensor descriptor. -/
theorem returned_relation (p : Config) (objects : Role→TensorDescriptor)
    (hn : NonOverlapping p) (binding : OutputBinding p objects)
    (word : Coordinate p→BF16Word) (memory : ByteMemory)
    (hp : pointerChecks (typedGeometry p) (pointers objects)=Bool.true)
    (relation : Coordinate p→BF16Word→Prop) (hlocal : ∀ c,relation c (word c)) :
    ∀ c,relation c ((finishBytes p objects word memory).read p c) := by
  intro c
  rw [finish_read p objects hn binding word memory hp c]
  exact hlocal c

/-- The empty branch returns the original fresh output object and preserves
every physical byte. The source still performs its O and unused M allocations. -/
theorem empty_native_return (p : Config) (objects : Role→TensorDescriptor)
    (memory : ByteMemory) (h : p.batches*p.heads*p.context*p.headDim=0) :
    IsEmpty (Coordinate p) ∧
    (finishBytes p objects (fun _=>0) memory).tensor=objects .output ∧
    (finishBytes p objects (fun _=>0) memory).memory=memory := by
  have he : IsEmpty (Coordinate p) := ⟨by
    intro c
    have hb:=Nat.pos_of_ne_zero (show p.batches≠0 by have:=c.1.isLt; omega)
    have hh:=Nat.pos_of_ne_zero (show p.heads≠0 by have:=c.2.1.isLt; omega)
    have hn:=Nat.pos_of_ne_zero (show p.context≠0 by have:=c.2.2.1.isLt; omega)
    have hd:=Nat.pos_of_ne_zero (show p.headDim≠0 by have:=c.2.2.2.isLt; omega)
    have hp:=Nat.mul_pos (Nat.mul_pos (Nat.mul_pos hb hh) hn) hd
    omega⟩
  refine ⟨he,rfl,?_⟩
  funext address
  apply finish_frame
  intro i
  exact False.elim (he.false i.1)

#axiomsClean input_dtype
#axiomsClean whole_native
#axiomsClean returned_relation
#axiomsClean empty_native_return
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Checked
