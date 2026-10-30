import bench.tritonbench_g.triton_attention.Proofs.NativeFrame
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
abbrev Word := BitVec 32

/-- Eleven local instruction families. TF32 input interpretation and the FP32
accumulator are part of the fixed K8 WGMMA leaf, not a variable-length dot. -/
structure Instructions where
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  fma : Word → Word → Word → Word
  neg : Word → Word
  max : Word → Word → Word
  exp2 : Word → Word
  div : Word → Word → Word
  narrow : Word → Word
  widen : Word → Word
  mma8 : Width → (Fin 8 → Word) → (Fin 8 → Word) → Bool → Word → Word

def ops (p : Instructions) : Ops Word :=
  ⟨0,0x3f800000,0xff800000,0x3fb8aa3b,p.add,p.sub,p.mul,p.fma,p.neg,
   p.max,p.exp2,p.div,p.narrow,p.widen,p.mma8⟩

structure LocalConformance (actual reference : Instructions) : Prop where
  add : ∀ a b, actual.add a b = reference.add a b
  sub : ∀ a b, actual.sub a b = reference.sub a b
  mul : ∀ a b, actual.mul a b = reference.mul a b
  fma : ∀ a b c, actual.fma a b c = reference.fma a b c
  neg : ∀ a, actual.neg a = reference.neg a
  max : ∀ a b, actual.max a b = reference.max a b
  exp2 : ∀ a, actual.exp2 a = reference.exp2 a
  div : ∀ a b, actual.div a b = reference.div a b
  narrow : ∀ a, actual.narrow a = reference.narrow a
  widen : ∀ a, actual.widen a = reference.widen a
  mma8 : ∀ w a b live seed, actual.mma8 w a b live seed = reference.mma8 w a b live seed

theorem local_conformance_eq (a b : Instructions) (h : LocalConformance a b) : a = b := by
  cases a; cases b
  have h1 := funext (fun x => funext (h.add x))
  have h2 := funext (fun x => funext (h.sub x))
  have h3 := funext (fun x => funext (h.mul x))
  have h4 := funext (fun x => funext (fun y => funext (h.fma x y)))
  have h5 := funext h.neg
  have h6 := funext (fun x => funext (h.max x))
  have h7 := funext h.exp2
  have h8 := funext (fun x => funext (h.div x))
  have h9 := funext h.narrow
  have h10 := funext h.widen
  have h11 := funext (fun w => funext (fun x => funext (fun y => funext (fun live => funext (h.mma8 w x y live)))))
  congr

def Correct (c : Shape) (reference : Instructions) (owners : Owners)
    (before after : Memory Word) (scale : Word) : Prop :=
  (∀ i : Coord c,
    after .o (offset c i) = (forwardReference c (ops reference) owners (input c before) scale).y
      i.1.val i.2.1.val i.2.2.val ∧
    after .dq (offset c i) = dqReference c (ops reference)
      (wholeBackwardInput c (ops reference) owners (input c before) scale) scale c.blocks
      i.1.val i.2.1.val i.2.2.val ∧
    after .dk (offset c i) = dkReference c (ops reference)
      (wholeBackwardInput c (ops reference) owners (input c before) scale) scale
      i.1.val i.2.1.val i.2.2.val ∧
    after .dv (offset c i) = dvReference c (ops reference)
      (wholeBackwardInput c (ops reference) owners (input c before) scale) scale
      i.1.val i.2.1.val i.2.2.val) ∧
  ∀ r o, ¬Written c r o → after r o = before r o

/-- Whole forward/preprocess/backward native-graph result, conditional only
on local numerical conformance. Scale is the same actual ABI word on both
sides; real attention mathematics is intentionally a separate theorem. -/
theorem whole_native (c : Shape) (actual reference : Instructions) (owners : Owners)
    (s : Memory Word) (scale : Word) (numeric : LocalConformance actual reference) :
    Correct c reference owners s (run c (ops actual) owners s scale) scale := by
  rw [local_conformance_eq actual reference numeric]
  exact ⟨whole_values c (ops reference) owners s scale,whole_frame c (ops reference) owners s scale⟩

theorem reference_witness (p : Instructions) : LocalConformance p p :=
  ⟨fun _ _ => rfl,fun _ _ => rfl,fun _ _ => rfl,fun _ _ _ => rfl,fun _ => rfl,
   fun _ _ => rfl,fun _ => rfl,fun _ _ => rfl,fun _ => rfl,fun _ => rfl,
   fun _ _ _ _ _ => rfl⟩

#axiomsClean local_conformance_eq
#axiomsClean whole_native
#axiomsClean reference_witness
#specNonCircular LocalConformance avoiding [run,forwardReference,wholeBackwardInput,dqReference,dkReference,dvReference,Correct]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
