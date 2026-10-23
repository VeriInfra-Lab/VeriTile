import bench.repairs.tritonbench_g.flash_attn.Proofs.SourceCorrect
namespace VeriTile.Bench.Repairs.FlashAttn
structure WholeNative (a : Host.Arguments) (o : Host.Tensor) (scratch caller : Nat)
    (actual reference : Native.Instructions) (ws : List Work) (initial : Physical.Bytes) : Prop where
  identity : (Host.invoke a o scratch caller actual ws initial).tensor=o
  shape : (Host.invoke a o scratch caller actual ws initial).tensor.shape=a.q.shape
  device : (Host.invoke a o scratch caller actual ws initial).tensor.device=a.q.device
  launchDevice : (Host.invoke a o scratch caller actual ws initial).launchDevice=a.q.device
  restoredDevice : (Host.invoke a o scratch caller actual ws initial).restoredDevice=caller
  output : ∀ b h row col,(Work.output b h row col).active (Host.shape a) → ∀ i : Fin 4,
    (Host.invoke a o scratch caller actual ws initial).bytes .o (4*address o.strides b h row col+i.val)=
      Physical.encode (outputValue (Host.layout a o) (Native.ops reference) (Physical.load initial) a.scale b h row col) i
  logsum : ∀ b h row,(Work.logsum b h row).active (Host.shape a) → ∀ i : Fin 4,
    (Host.invoke a o scratch caller actual ws initial).bytes .l (4*lAddress (Host.shape a) b h row+i.val)=
      Physical.encode (logValue (Host.layout a o) (Native.ops reference) (Physical.load initial) a.scale b h row) i
  frame : ∀ r b,¬footprint (Host.layout a o) r (b/4) →
    (Host.invoke a o scratch caller actual ws initial).bytes r b=initial r b

theorem whole_fp (a : Host.Arguments) (valid : Host.Contract a) (o : Host.Tensor) (scratch caller : Nat)
    (allocation : Host.AllocationFacts a o scratch) (actual reference : Native.Instructions)
    (numeric : Native.Correspondence actual reference) (ws : List Work) (complete : Complete (Host.shape a) ws)
    (initial : Physical.Bytes) : WholeNative a o scratch caller actual reference ws initial := by
  refine ⟨rfl,allocation.outputShape,allocation.outputDevice,rfl,rfl,
    Host.invoke_output a valid o scratch caller allocation actual reference numeric ws complete initial,?_,
    Host.invoke_frame a valid o scratch caller actual ws complete initial⟩
  intro b h row active i
  exact Physical.whole_bytes (Host.layout a o) (Host.contract_shape a valid)
    (Host.output_layout a o scratch allocation) actual reference numeric a.scale ws complete initial
    (.logsum b h row) active i

theorem whole_native (a : Host.Arguments) (valid : Host.Contract a) (o : Host.Tensor) (scratch caller : Nat)
    (allocation : Host.AllocationFacts a o scratch) (native : Native.Instructions)
    (ws : List Work) (complete : Complete (Host.shape a) ws) (initial : Physical.Bytes) :
    WholeNative a o scratch caller native native ws initial :=
  whole_fp a valid o scratch caller allocation native native (Native.reflexive native) ws complete initial
#axiomsClean whole_native
#axiomsClean whole_fp
end VeriTile.Bench.Repairs.FlashAttn
