import bench.tritonbench_g.fused_rotary_embedding.Proofs.Arena
import bench.tritonbench_g.fused_rotary_embedding.Proofs.Addresses
import bench.tritonbench_g.fused_rotary_embedding.Proofs.HistoricalFPEquiv

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding.Binding
open Checked Arena
open Classical
noncomputable section

structure Result where
  value : Option Unit
  tensors : Region → Tensor
  deviceRestored : Nat
  memory : Arena.Memory

def invoke (c : Call) (callerDevice : Nat) (p : Native.Instructions) (m : Arena.Memory) : Option Result :=
  if check c (mirror c m) then some ⟨none,c.tensor,callerDevice,Arena.run c p m⟩ else none

def read (c : Call) (m : Arena.Memory) (r : Region) (a : Nat) : Native.Word :=
  Physical.decode (fun i => m (address c r (4*a+i.val)))

theorem logical_extent (c : Call) (initial : Physical.Bytes) (checked : check c initial=true)
    (kind : Kind) (t h d : Nat) (ht : t<c.shape.tokens) (hh : h<c.shape.heads kind)
    (hd : d<c.shape.width) :
    logicalOutput (host c initial).config kind t h d<extent c kind.output := by
  have hw := column_work c.shape (shape_valid c initial checked) kind t h d ht hh hd
  have bound := output_extent c initial checked (columnWork c.shape kind t h d) hw.1 (secondHalf c.shape d)
  have out := output_logical (host c initial).config (columnWork c.shape kind t h d) (secondHalf c.shape d)
  change (host c initial).config.output (columnWork c.shape kind t h d) (secondHalf c.shape d)=
    logicalOutput (host c initial).config kind t h (c.shape.dim (columnWork c.shape kind t h d) (secondHalf c.shape d)) at out
  rw [out,hw.2] at bound
  exact bound

/-- Complete returned objects and finite byte memory. The target reads the
initial four-byte cells, including both selected int32 metadata arrays. -/
structure NativeResult (c : Call) (callerDevice : Nat) (actual target : Native.Instructions)
    (initial : Arena.Memory) (result : Result) : Prop where
  returnedNone : result.value=none
  objects : result.tensors=c.tensor
  device : result.deviceRestored=callerDevice
  source : Physical.load (mirror c result.memory)=Native.project
    (actualSource (host c (mirror c initial)).config (Native.ops actual)
      (Native.liftMemory (Physical.load (mirror c initial))))
  values : ∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width →
    read c result.memory kind.output (logicalOutput (host c (mirror c initial)).config kind t h d)=
      wordFormula c.shape target (Physical.load (mirror c initial)) kind t h d
  bytes : ∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width → ∀ i : Fin 4,
    result.memory (address c kind.output (4*logicalOutput (host c (mirror c initial)).config kind t h d+i.val))=
      Physical.encode (wordFormula c.shape target (Physical.load (mirror c initial)) kind t h d) i
  pointers : ∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width → ∀ i : Fin 4,
    (address c kind.output (4*logicalOutput (host c (mirror c initial)).config kind t h d+i.val)).val=
      (c.tensor kind.output).base+4*logicalOutput (host c (mirror c initial)).config kind t h d+i.val
  frame : ∀ a,Outside c (mirror c initial) a → result.memory a=initial a
  readOnly : ∀ r b,r≠.q ∧ r≠.kcache ∧ r≠.vcache → b<4*extent c r →
    result.memory (address c r b)=initial (address c r b)
  empty : c.shape.tokens=0 → result.memory=initial

theorem complete (c : Call) (callerDevice : Nat) (p : Native.Instructions) (m : Arena.Memory)
    (checked : check c (mirror c m)=true) :
    ∃ result,invoke c callerDevice p m=some result ∧ NativeResult c callerDevice p p m result := by
  let initial := mirror c m
  have hc := host_admitted c initial checked
  have old := whole_native (host c initial) hc callerDevice p initial rfl
  have hm := mirror_run c p m checked
  have bytes : ∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width → ∀ i : Fin 4,
      Arena.run c p m (address c kind.output (4*logicalOutput (host c initial).config kind t h d+i.val))=
        Physical.encode (wordFormula c.shape p (Physical.load initial) kind t h d) i := by
    intro kind t h d ht hh hd i
    have b := old.bytes kind t h d ht hh hd i
    have bound := logical_extent c initial checked kind t h d ht hh hd
    rw [Host.selected_memory] at b
    change Physical.dynamic c.shape p (mirror c m) kind.output
      (4*logicalOutput (host c initial).config kind t h d+i.val)=
      Physical.encode (wordFormula c.shape p (Physical.load initial) kind t h d) i at b
    rw [←hm] at b
    simpa only [mirror,show 4*logicalOutput (host c initial).config kind t h d+i.val<4*extent c kind.output by omega,ite_true] using b
  refine ⟨⟨none,c.tensor,callerDevice,Arena.run c p m⟩,by simp [invoke,checked],
    rfl,rfl,rfl,?_,?_,bytes,?_,Arena.run_frame c p m checked,?_,empty_run c p m⟩
  · rw [hm]
    simpa only [Host.selected_memory,host] using old.source
  · intro kind t h d ht hh hd
    unfold read
    rw [←Physical.decode_encode (wordFormula c.shape p (Physical.load initial) kind t h d)]
    apply congrArg Physical.decode
    funext i
    exact bytes kind t h d ht hh hd i
  · intro kind t h d ht hh hd i
    have bound := logical_extent c initial checked kind t h d ht hh hd
    have h := (byte_bound c initial checked (by omega) kind.output
      (4*logicalOutput (host c initial).config kind t h d+i.val) (by omega)).2
    simpa only [Nat.add_assoc] using h
  · intro r b hr hb
    have frame := old.readOnly r b hr
    rw [Host.selected_memory] at frame
    change Physical.dynamic c.shape p (mirror c m) r b=mirror c m r b at frame
    rw [←hm] at frame
    simpa only [mirror,hb,ite_true] using frame

theorem complete_if_local (c : Call) (callerDevice : Nat) (actual target : Native.Instructions)
    (localAgreement : Native.Obligations actual target) (m : Arena.Memory)
    (checked : check c (mirror c m)=true) :
    ∃ result,invoke c callerDevice actual m=some result ∧ NativeResult c callerDevice actual target m result := by
  rw [←Native.instructions_eq actual target localAgreement]
  exact complete c callerDevice actual m checked

#axiomsClean complete
#axiomsClean complete_if_local
#specNonCircular wordFormula avoiding [invoke,Arena.run,Physical.dynamic,actualSource]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRotaryEmbedding.Binding
