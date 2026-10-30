import bench.repairs.tritonbench_g.flash_attn.Proofs.Grid
namespace VeriTile.Bench.Repairs.FlashAttn.Native
/-- Scalar implementations include their exceptional-value semantics. The fixed
K8 port is one logical output component of an instruction, never a full dot.
Register-fragment projection is an ordinary pinned compiler/ABI boundary. -/
structure Instructions where
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  fma : Word → Word → Word → Word
  max : Word → Word → Word
  exp2 : Word → Word
  log2 : Word → Word
  div : Word → Word → Word
  narrow : Word → Word
  widen : Word → Word
  mma8 : Width → (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

def ops (p : Instructions) : Ops Word where
  zero:=0
  negInf:=0xff800000
  log2e:=0x3fb8aa3b
  add:=p.add
  sub:=p.sub
  mul:=p.mul
  fma:=p.fma
  max:=p.max
  exp2:=p.exp2
  log2:=p.log2
  div:=p.div
  narrow:=p.narrow
  widen:=p.widen
  mma8:=p.mma8

structure Correspondence (actual reference : Instructions) : Prop where
  add : ∀ a b,actual.add a b=reference.add a b
  sub : ∀ a b,actual.sub a b=reference.sub a b
  mul : ∀ a b,actual.mul a b=reference.mul a b
  fma : ∀ a b c,actual.fma a b c=reference.fma a b c
  max : ∀ a b,actual.max a b=reference.max a b
  exp2 : ∀ a,actual.exp2 a=reference.exp2 a
  log2 : ∀ a,actual.log2 a=reference.log2 a
  div : ∀ a b,actual.div a b=reference.div a b
  narrow : ∀ a,actual.narrow a=reference.narrow a
  widen : ∀ a,actual.widen a=reference.widen a
  mma8 : ∀ w a b seed,actual.mma8 w a b seed=reference.mma8 w a b seed

theorem correspondence_eq (a b : Instructions) (h : Correspondence a b) : a=b := by
  cases a;cases b
  have h1:=funext (fun x=>funext (h.add x))
  have h2:=funext (fun x=>funext (h.sub x))
  have h3:=funext (fun x=>funext (h.mul x))
  have h4:=funext (fun x=>funext (fun y=>funext (h.fma x y)))
  have h5:=funext (fun x=>funext (h.max x))
  have h6:=funext h.exp2
  have h7:=funext h.log2
  have h8:=funext (fun x=>funext (h.div x))
  have h9:=funext h.narrow
  have h10:=funext h.widen
  have h11:=funext (fun w=>funext (fun x=>funext (fun y=>funext (h.mma8 w x y))))
  congr

theorem whole_native (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x)
    (actual reference : Instructions) (numeric : Correspondence actual reference) (m : Memory Word)
    (scale : Word) (ws : List Work) (complete : Complete x.shape ws) (w : Work) (active : w.active x.shape) :
    execute x (ops actual) scale ws m (w.target x).1 (w.target x).2=
      w.result x (ops reference) m scale := by
  rw [correspondence_eq actual reference numeric]
  exact call_value x hs ho (ops reference) scale ws complete m w active

theorem whole_frame (x : Layout) (hs : x.shape.Valid) (actual : Instructions) (m : Memory Word)
    (scale : Word) (ws : List Work) (complete : Complete x.shape ws) (r : Region) (a : Nat)
    (outside : ¬footprint x r a) : execute x (ops actual) scale ws m r a=m r a :=
  call_frame x hs (ops actual) scale ws complete m r a outside

theorem reflexive (p : Instructions) : Correspondence p p :=
  ⟨fun _ _=>rfl,fun _ _=>rfl,fun _ _=>rfl,fun _ _ _=>rfl,fun _ _=>rfl,
    fun _=>rfl,fun _=>rfl,fun _ _=>rfl,fun _=>rfl,fun _=>rfl,fun _ _ _ _=>rfl⟩
#axiomsClean correspondence_eq
#axiomsClean whole_native
#axiomsClean whole_frame
#axiomsClean reflexive
end VeriTile.Bench.Repairs.FlashAttn.Native
