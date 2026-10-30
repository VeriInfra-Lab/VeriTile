import bench.tritonbench_g.block_sparse_attn.Proofs.Memory
namespace VeriTile.Bench.Repairs.BlockSparseAttn.Native

structure Instructions where
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  fma : Word → Word → Word → Word
  max : Word → Word → Word
  exp : Word → Word
  div : Word → Word → Word
  mma8 : Instruction → (Fin 8 → Word) → (Fin 8 → Word) → Word → Word
def ops (p : Instructions) : Ops Word where
  zero:=0
  negInf:=0xff800000
  add:=p.add
  sub:=p.sub
  mul:=p.mul
  fma:=p.fma
  max:=p.max
  exp:=p.exp
  div:=p.div
  mma8:=p.mma8

/-- Only seven scalar implementations and one fixed-width instruction
component are numerical premises. The exp implementation includes its local
FP32 log2(e) multiply and scalar ex2 instruction. -/
structure Correspondence (actual reference : Instructions) : Prop where
  add : ∀ a b,actual.add a b=reference.add a b
  sub : ∀ a b,actual.sub a b=reference.sub a b
  mul : ∀ a b,actual.mul a b=reference.mul a b
  fma : ∀ a b c,actual.fma a b c=reference.fma a b c
  max : ∀ a b,actual.max a b=reference.max a b
  exp : ∀ a,actual.exp a=reference.exp a
  div : ∀ a b,actual.div a b=reference.div a b
  mma8 : ∀ instruction a b seed,actual.mma8 instruction a b seed=reference.mma8 instruction a b seed

theorem correspondence_eq (a b : Instructions) (h : Correspondence a b) : a=b := by
  cases a;cases b
  have h1:=funext (fun x=>funext (h.add x))
  have h2:=funext (fun x=>funext (h.sub x))
  have h3:=funext (fun x=>funext (h.mul x))
  have h4:=funext (fun x=>funext (fun y=>funext (h.fma x y)))
  have h5:=funext (fun x=>funext (h.max x))
  have h6:=funext h.exp
  have h7:=funext (fun x=>funext (h.div x))
  have h8:=funext (fun instruction=>funext (fun x=>funext (fun y=>funext (h.mma8 instruction x y))))
  congr

theorem reflexive (p : Instructions) : Correspondence p p :=
  ⟨fun _ _=>rfl,fun _ _=>rfl,fun _ _=>rfl,fun _ _ _=>rfl,
    fun _ _=>rfl,fun _=>rfl,fun _ _=>rfl,fun _ _ _ _=>rfl⟩

abbrev Bytes := FlashAttn.Physical.Bytes
abbrev encode := FlashAttn.Physical.encode
abbrev load := FlashAttn.Physical.load
def byteStep (x : Layout) (p : Ops Word) (scale : Word) (w : Work) (m : Bytes) : Bytes :=
  FlashAttn.Physical.byteStore m (.o,w.target x.shape) (rowValue x p (load m) scale w)
def byteExecute (x : Layout) (p : Ops Word) (scale : Word) : List Work → Bytes → Bytes
  | [],m=>m
  | w::ws,m=>byteExecute x p scale ws (byteStep x p scale w m)

theorem execute_load (x : Layout) (p : Ops Word) (scale : Word) (ws : List Work) (m : Bytes) :
    load (byteExecute x p scale ws m)=execute x p scale ws (load m) := by
  induction ws generalizing m with
  | nil=>rfl
  | cons w ws ih=>
    change load (byteExecute x p scale ws (byteStep x p scale w m))=_
    rw [ih]
    simp only [byteStep,FlashAttn.Physical.load_store,execute,step]

theorem output_bytes (x : Layout) (actual reference : Instructions) (numeric : Correspondence actual reference)
    (scale : Word) (ws : List Work) (complete : Complete x.shape ws) (m : Bytes)
    (w : Work) (active : w.active x.shape) (i : Fin 4) :
    byteExecute x (ops actual) scale ws m .o (4*w.target x.shape+i.val)=
      encode (rowValue x (ops reference) (load m) scale w) i := by
  rw [correspondence_eq actual reference numeric]
  have he:=call_value x (ops reference) scale ws complete (load m) w active
  rw [←execute_load] at he
  have hb:=congrArg (fun z=>encode z i) he
  exact (FlashAttn.Physical.original_byte _ _ _ i).symm.trans hb

theorem byte_frame (x : Layout) (p : Ops Word) (scale : Word) (ws : List Work)
    (complete : Complete x.shape ws) (m : Bytes) (r : FlashAttn.Region) (b : Nat)
    (outside : ¬footprint x.shape r (b/4)) : byteExecute x p scale ws m r b=m r b := by
  have general (ws : List Work) (m : Bytes) (out : ∀ w∈ws,(r,b/4)≠(.o,w.target x.shape)) :
      byteExecute x p scale ws m r b=m r b := by
    induction ws generalizing m with
    | nil=>rfl
    | cons w ws ih=>
      change byteExecute x p scale ws (byteStep x p scale w m) r b=_
      rw [ih _ (fun u hu=>out u (by simp [hu]))]
      simp [byteStep,FlashAttn.Physical.byteStore,out w (by simp)]
  apply general
  intro w hw he
  exact outside ⟨w,(schedule_mem x.shape w).mp (complete.mem_iff.mp hw),he⟩

#axiomsClean correspondence_eq
#axiomsClean reflexive
#axiomsClean execute_load
#axiomsClean output_bytes
#axiomsClean byte_frame
end VeriTile.Bench.Repairs.BlockSparseAttn.Native
