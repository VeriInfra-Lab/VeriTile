import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Host

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding

/-- Independent native target indexed by arbitrary logical output column.
V is copied without numerical operations; Q and K use their own selected
graphs and exact operand orders over unrestricted raw FP32 words. -/
def wordFormula (s : Shape) (p : Native.Instructions) (m : Memory Native.Word)
    (kind : Kind) (t h d : Nat) : Native.Word :=
  if kind=.v then m .v (s.data .v t h d)
  else
    let low := if d<s.half then d else d-s.half
    Native.graph p kind (m kind.input (s.data kind t h low))
      (m kind.input (s.data kind t h (low+s.half)))
      (m .cos (s.coeff false t low)) (m .sin (s.coeff true t low)) (secondHalf s d)

theorem pairWord_formula (s : Shape) (w : Work) (hw : s.Active w)
    (p : Native.Instructions) (m : Memory Native.Word) (o : Bool) :
    Native.pairWord s p w m o=wordFormula s p m w.kind w.token w.head (s.dim w o) := by
  have low := hw.2.2
  have high : ¬w.pair+s.half<s.half := by omega
  have notSecond : ¬s.half≤w.pair := by omega
  have yesSecond : s.half≤w.pair+s.half := by omega
  cases kw : w.kind <;> cases o <;>
    simp [Native.pairWord,wordFormula,Shape.dim,Shape.input,kw,Kind.input,low,high,secondHalf,notSecond,yesSecond]

theorem dynamic_column (call : Host.Call) (domain : Host.Admitted call) (p : Native.Instructions)
    (m : Physical.Bytes) (binding : Physical.metadata m=call.config.metadata)
    (kind : Kind) (t h d : Nat) (ht : t<call.config.shape.tokens)
    (hh : h<call.config.shape.heads kind) (hd : d<call.config.shape.width) :
    Physical.load (Physical.dynamic call.config.shape p m) kind.output (logicalOutput call.config kind t h d)=
      wordFormula call.config.shape p (Physical.load m) kind t h d := by
  have ce := Host.configuration_binding call m binding
  have cv : (Physical.configuration call.config.shape m).Valid := by simpa [ce] using domain.config
  let w := columnWork call.config.shape kind t h d
  let o := secondHalf call.config.shape d
  have hw := column_work call.config.shape domain.config.shape kind t h d ht hh hd
  have result := Physical.dynamic_result call.config.shape p m cv w hw.1 o
  rw [ce,output_logical,pairWord_formula call.config.shape w hw.1 p (Physical.load m) o] at result
  have dimEq : call.config.shape.dim w o=d := hw.2
  rw [dimEq] at result
  simpa [w,columnWork] using result

theorem readonly_outside (c : Config) (r : Region) (a : Nat)
    (hr : r≠.q ∧ r≠.kcache ∧ r≠.vcache) : ¬Footprint c r a := by
  rintro ⟨kind,t,h,d,ht,hh,hd,rfl,ha⟩
  cases kind <;> simp_all [Kind.output]

structure NativeClaim (call : Host.Call) (callerDevice : Nat) (actual target : Native.Instructions)
    (initial : Physical.Bytes) (result : Host.Returned Physical.Byte) : Prop where
  returnedNone : result.value=none
  identities : result.identities=call.identities
  device : result.deviceRestored=callerDevice
  source : Physical.load result.memory=Native.project
    (actualSource call.config (Native.ops actual) (Native.liftMemory (Physical.load initial)))
  values : ∀ kind t h d,t<call.config.shape.tokens → h<call.config.shape.heads kind → d<call.config.shape.width →
    Physical.load result.memory kind.output (logicalOutput call.config kind t h d)=
      wordFormula call.config.shape target (Physical.load initial) kind t h d
  bytes : ∀ kind t h d,t<call.config.shape.tokens → h<call.config.shape.heads kind → d<call.config.shape.width → ∀ i : Fin 4,
    result.memory kind.output (4*logicalOutput call.config kind t h d+i.val)=
      Physical.encode (wordFormula call.config.shape target (Physical.load initial) kind t h d) i
  frame : ∀ r b,¬Footprint call.config r (b/4) → result.memory r b=initial r b
  readOnly : ∀ r b,r≠.q ∧ r≠.kcache ∧ r≠.vcache → result.memory r b=initial r b

theorem whole_native (call : Host.Call) (domain : Host.Admitted call) (callerDevice : Nat)
    (actual : Native.Instructions) (initial : Physical.Bytes)
    (binding : Physical.metadata initial=call.config.metadata) :
    NativeClaim call callerDevice actual actual initial (Host.selected call callerDevice actual initial) := by
  have memory := Host.selected_memory call callerDevice actual initial
  have ce := Host.configuration_binding call initial binding
  have cv : (Physical.configuration call.config.shape initial).Valid := by simpa [ce] using domain.config
  have values := dynamic_column call domain actual initial binding
  have frame : ∀ r b,¬Footprint call.config r (b/4) →
      Physical.dynamic call.config.shape actual initial r b=initial r b := by
    intro r b outside
    apply Physical.dynamic_frame
    rw [ce]
    intro w hw h
    exact outside (writes_footprint call.config domain.config w hw r (b/4) h)
  refine ⟨rfl,rfl,rfl,?_,?_,?_,?_,?_⟩
  · rw [memory]
    simpa [ce] using Physical.dynamic_source call.config.shape actual initial cv
  · rw [memory]
    exact values
  · intro kind t h d ht hh hd i
    rw [memory,←Physical.original_bytes (Physical.dynamic call.config.shape actual initial)
      kind.output (logicalOutput call.config kind t h d) i,values kind t h d ht hh hd]
  · rw [memory]
    exact frame
  · intro r b hr
    rw [memory]
    exact frame r b (readonly_outside call.config r (b/4) hr)

theorem whole_fp (call : Host.Call) (domain : Host.Admitted call) (callerDevice : Nat)
    (actual target : Native.Instructions) (numeric : Native.Obligations actual target)
    (initial : Physical.Bytes) (binding : Physical.metadata initial=call.config.metadata) :
    NativeClaim call callerDevice actual target initial (Host.selected call callerDevice actual initial) := by
  rw [←Native.instructions_eq actual target numeric]
  exact whole_native call domain callerDevice actual initial binding

#specNonCircular wordFormula avoiding [sourcePair,pairValues,actualSource,Native.selected]
#axiomsClean whole_native
#axiomsClean whole_fp
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FusedRotaryEmbedding
