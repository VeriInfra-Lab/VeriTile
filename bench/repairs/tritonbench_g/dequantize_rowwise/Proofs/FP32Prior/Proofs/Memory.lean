import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Grid
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Bytes
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Memory
open VeriTile.Triton FP.Structural
open Bytes
noncomputable section
set_option maxHeartbeats 6000000

abbrev Half := Fin 2 → Byte

def read32 (m : Memory) (address : Nat) : Word := fun b=>m (address+b.val)
def read16 (m : Memory) (address : Nat) : Half := fun b=>m (address+b.val)
def pad (h : Half) : Word := fun b=>if bound:b.val<2 then h ⟨b.val,bound⟩ else 0
def low (w : Word) : Half := fun b=>w ⟨b.val,lt_trans b.isLt (by decide)⟩

@[simp] theorem low_pad (h : Half) : low (pad h)=h := by
  funext b
  simp [low,pad,b.isLt]

/-- The FP32 host scalar 1.0/127 is bound to its compiled bits separately.
This is an exact raw constant, not an assumed arithmetic accuracy law. -/
def inverse : Word := ![4,2,1,60]

structure Numerics where
  multiply32 : Word → Word → Word
  convert16 : Word → Half

def compute (ops : Numerics) (scale x : Word) : Half :=
  ops.convert16 (ops.multiply32 (ops.multiply32 scale x) inverse)

def backend (rest : Algebra Word) (ops : Numerics) : Algebra Word :=
  {rest with
    fp32Load:=id
    literal:=fun _ _ _=>inverse
    cast:=fun _ _ _ x=>pad (ops.convert16 x)
    binary:=fun precision dtype operation x y=>match operation with
      | .mul=>ops.multiply32 x y
      | _=>rest.binary precision dtype operation x y}

theorem typed_scalar (rest : Algebra Word) (ops : Numerics) (scale x : Word) :
    Execution.scalar (backend rest ops) scale x=pad (compute ops scale x) := rfl

abbrev Event (rows cols : Nat) := Coord rows cols × Fin 2

def events (rows cols : Nat) : List (Event rows cols) :=
  ((List.finRange rows).product (List.finRange cols)).product (List.finRange 2)

theorem events_complete (e : Event rows cols) : e∈events rows cols := by
  rcases e with ⟨⟨r,c⟩,b⟩
  exact List.mem_product.mpr ⟨List.mem_product.mpr ⟨List.mem_finRange r,List.mem_finRange c⟩,List.mem_finRange b⟩

def key (base : Nat) (e : Event rows cols) : Nat := base+2*linear e.1+e.2.val

theorem key_injective (base : Nat) : Function.Injective (@key rows cols base) := by
  intro a b equal
  have ha:=a.2.isLt
  have hb:=b.2.isLt
  have indexEq : linear a.1=linear b.1 := by unfold key at equal;omega
  apply Prod.ext (linear_injective indexEq)
  apply Fin.ext
  unfold key at equal
  rw [indexEq] at equal
  omega

def xAddress (base : Nat) (q : Coord rows cols) : Nat := base+4*linear q
def scaleAddress (base : Nat) (q : Coord rows cols) : Nat := base+4*q.1.val

def value (ops : Numerics) (xb sb : Nat) (m : Memory) (q : Coord rows cols) : Half :=
  compute ops (read32 m (scaleAddress sb q)) (read32 m (xAddress xb q))

def snapshot (ops : Numerics) (xb sb ob : Nat) (initial : Memory)
    (order : List (Event rows cols)) (current : Memory) : Memory :=
  Scatter.run order (key ob) (fun e=>value ops xb sb initial e.1 e.2) current

/-- A byte-store serialization with actual current-memory loads. The complete
native row's load-all/store order is connected to this by the unchanged-input
lemma below; no output formula is built into a memory primitive. -/
def live (ops : Numerics) (xb sb ob : Nat) : List (Event rows cols) → Memory → Memory
  | [],m=>m
  | e::tail,m=>live ops xb sb ob tail (Scatter.write m (key ob e) (value ops xb sb m e.1 e.2))

def InputsSame {rows cols : Nat} (xb sb : Nat) (initial current : Memory) : Prop :=
  ∀ q : Coord rows cols,
    read32 current (xAddress xb q)=read32 initial (xAddress xb q) ∧
    read32 current (scaleAddress sb q)=read32 initial (scaleAddress sb q)

def Separated {rows cols : Nat} (xb sb ob : Nat) : Prop :=
  ∀ (e : Event rows cols) (q : Coord rows cols) (b : Fin 4),
    key ob e≠xAddress xb q+b.val ∧ key ob e≠scaleAddress sb q+b.val

theorem write_inputs (sep : @Separated rows cols xb sb ob)
    (initial current : Memory) (same : @InputsSame rows cols xb sb initial current)
    (e : Event rows cols) (v : Byte) :
    @InputsSame rows cols xb sb initial (Scatter.write current (key ob e) v) := by
  intro q
  constructor
  · refine Eq.trans ?_ (same q).1
    funext b
    simp only [read32,Scatter.write,Ne.symm (sep e q b).1,↓reduceIte]
  · refine Eq.trans ?_ (same q).2
    funext b
    simp only [read32,Scatter.write,Ne.symm (sep e q b).2,↓reduceIte]

theorem value_same (ops : Numerics) (initial current : Memory)
    (same : @InputsSame rows cols xb sb initial current) (q : Coord rows cols) :
    value ops xb sb current q=value ops xb sb initial q := by
  simp only [value,(same q).1,(same q).2]

theorem live_snapshot (ops : Numerics) (sep : @Separated rows cols xb sb ob)
    (initial : Memory) (order : List (Event rows cols)) (current : Memory)
    (same : @InputsSame rows cols xb sb initial current) :
    live ops xb sb ob order current=snapshot ops xb sb ob initial order current := by
  induction order generalizing current with
  | nil=>rfl
  | cons e tail ih=>
    change live ops xb sb ob tail (Scatter.write current (key ob e) (value ops xb sb current e.1 e.2))=_
    rw [value_same ops initial current same]
    exact ih _ (write_inputs sep initial current same e _)

theorem live_readback (ops : Numerics) (sep : @Separated rows cols xb sb ob)
    (initial : Memory) (order : List (Event rows cols)) (q : Coord rows cols)
    (complete : ∀ b : Fin 2,(q,b)∈order) :
    read16 (live ops xb sb ob order initial) (ob+2*linear q)=value ops xb sb initial q := by
  rw [live_snapshot ops sep initial order initial (fun _=>⟨rfl,rfl⟩)]
  funext b
  exact Scatter.readback_injective order (key ob) (fun e=>value ops xb sb initial e.1 e.2)
    initial (q,b) (complete b) (key_injective ob)

theorem live_frame (ops : Numerics) (sep : @Separated rows cols xb sb ob)
    (initial : Memory) (order : List (Event rows cols)) (address : Nat)
    (outside : ∀ e∈order,address≠key ob e) :
    live ops xb sb ob order initial address=initial address := by
  rw [live_snapshot ops sep initial order initial (fun _=>⟨rfl,rfl⟩)]
  exact Scatter.frame _ _ _ _ address outside

def launch (ops : Numerics) (xb sb ob : Nat) (rows cols : Nat) (initial : Memory) : Memory :=
  live ops xb sb ob (events rows cols) initial

theorem launch_readback (ops : Numerics) (sep : @Separated rows cols xb sb ob)
    (initial : Memory) (q : Coord rows cols) :
    read16 (launch ops xb sb ob rows cols initial) (ob+2*linear q)=value ops xb sb initial q :=
  live_readback ops sep initial _ q (fun b=>events_complete (q,b))

theorem launch_frame (ops : Numerics) (sep : @Separated rows cols xb sb ob)
    (initial : Memory) (address : Nat) (outside : ∀ e : Event rows cols,address≠key ob e) :
    launch ops xb sb ob rows cols initial address=initial address :=
  live_frame ops sep initial _ address (fun e _=>outside e)

#axiomsClean typed_scalar
#axiomsClean live_snapshot
#axiomsClean launch_readback
#axiomsClean launch_frame
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Memory
