import bench.tritonbench_g.attention_llama.Proofs.FiniteInputs

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.GridGeometry
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.AttentionLlama HostGrid MaskCoverage
open Backing FiniteInputs
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

@[simp] theorem schedule_length (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) :
    (NativeIO.schedule C arg B c).length=Fintype.card (NativeIO.Program B c) :=
  List.length_ofFn

theorem schedule_get (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (k : Fin (NativeIO.schedule C arg B c).length) :
    (NativeIO.schedule C arg B c)[k]=NativeIO.invocation C arg B c
      (NativeIO.program B c ⟨k.val,by simpa only [schedule_length] using k.isLt⟩) := by
  exact List.get_ofFn (fun i=>NativeIO.invocation C arg B c (NativeIO.program B c i)) k

theorem invocation_initial (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (hD : 0<c.D) (p : NativeIO.Program B c) (s : State (Term α)) :
    (NativeIO.invocation C arg B c p).initial s=
      bindScale (Tensor.initialFor c s (anchor hD p)) (Native.argument C arg) := by
  have h0:=congrArg (fun q=>q.1.val) (owner_anchor hD p)
  have h1:=congrArg (fun q=>q.2.val) (owner_anchor hD p)
  change (anchor hD p).query.val/64=p.1.val at h0
  change (anchor hD p).batch.val*c.NH+(anchor hD p).head.val=p.2.val at h1
  simp only [NativeIO.invocation,Invocation.initial,bindRegisters,List.foldl_cons,List.foldl_nil,
    bindScale,Tensor.initialFor,h0,h1,↓reduceIte,Option.getD_none]
  rfl

theorem rows_anchor (c : Config) (s : State (Term α)) (hD : 0<c.D)
    (p : NativeIO.Program B c) (j : Fin 64) :
    rows (Tensor.initialFor c s (anchor hD p)) j=p.1.val*64+j.val := by
  have h0:=congrArg (fun q=>q.1.val) (owner_anchor hD p)
  change (anchor hD p).query.val/64=p.1.val at h0
  simp only [rows,Tensor.initialFor,h0,↓reduceIte]

theorem tensor_injective (c : Config)
    (hinj : Function.Injective (IOLayout.output (B:=B) c)) :
    Function.Injective (Tensor.offset (B:=B) c) := by
  intro i j he
  simpa only [index_tile] using congrArg index
    (hinj (show IOLayout.output c (tile i)=IOLayout.output c (tile j) from he))

/-- Every source active output lane belongs to its declared finite IO owner. -/
theorem active_writes (c : Config) (s : State (Term α)) (hD : 0<c.D)
    (p : NativeIO.Program B c) (j : TileIndex [64,c.D])
    (hj : rows (Tensor.initialFor c s (anchor hD p)) j.1<c.H) :
    NativeIO.writes B c p c.Out (address c (Tensor.initialFor c s (anchor hD p)) j) := by
  have hj' : p.1.val*64+j.1.val<c.H := by simpa only [rows_anchor] using hj
  let i:=Backing.laneIndex c hD p j hj'
  have own:=owner_lane c hD p j hj'
  refine ⟨tile i,?_,?_,rfl,?_⟩
  · exact (congrArg (fun q=>q.1.val) own).symm
  · exact (congrArg (fun q=>q.2.val) own).symm
  · exact lane_address c s hD p j hj'

/-- Every logical output is owned by exactly its original physical CTA. -/
theorem owner_writes (c : Config) (i : TileIndex [B,c.NH,c.H,c.D]) :
    NativeIO.writes B c (owner (index i)) c.Out (IOLayout.output c i) :=
  ⟨i,rfl,rfl,rfl,rfl⟩

theorem writes_disjoint (c : Config) (hinj : Function.Injective (IOLayout.output (B:=B) c))
    (p q : NativeIO.Program B c) (r : RegionName) (o : Nat)
    (hp : NativeIO.writes B c p r o) (hq : NativeIO.writes B c q r o) : p=q := by
  obtain ⟨i,hi0,hi1,_,hio⟩:=hp
  obtain ⟨j,hj0,hj1,_,hjo⟩:=hq
  have he : i=j := hinj (hio.symm.trans hjo)
  subst j
  exact Prod.ext (Fin.ext (hi0.trans hj0.symm)) (Fin.ext (hi1.trans hj1.symm))

#axiomsClean invocation_initial
#axiomsClean active_writes
#axiomsClean writes_disjoint
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.GridGeometry
