import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeInstructions
import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativePhysical

namespace VeriTile.Bench.Repairs.ContextAttnMistral.Recurrence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section

/-- Initial logical coordinates only; no execution state or output reader. -/
structure Inputs (α : Type) where
  (query key value : Nat → Nat → Nat → α)

abbrev Row := TileIndex [128]
abbrev Matrix (D : Nat) := TileIndex [128,D]
structure Accumulator (α : Type) (D : Nat) where
  center : Row → α
  denominator : Row → α
  value : Matrix D → α

def precision : Option ComputeDType := some .fp32

def zero (o : FPBackend.Ops α) : α := o.scalar.literal precision .real 0

def bin (o : FPBackend.Ops α) (op : Binary) (a b : α) : α :=
  o.scalar.binary precision .real op a b

def exp (o : FPBackend.Ops α) (x : α) : α := o.scalar.unary precision .exp x

/-- The compiler folds the source literal initialization zeros-minus-infinity
to the exact negative-infinity word; no finite rounded operand is omitted. -/
def initial (o : FPBackend.Ops α) (D : Nat) : Accumulator α D :=
  ⟨fun _ => o.scalar.negInf, fun _ => zero o, fun _ => zero o⟩

def query (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h block : Nat) : Matrix c.layout.D → α := fun i =>
  if block*128+i.1.val<c.length b then input.query (c.start b+block*128+i.1.val) h i.2.1.val else zero o

def key (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h tile : Nat) : TileIndex [c.layout.D,128] → α := fun i =>
  if tile*128+i.2.1.val<c.length b then
    input.key (c.start b+tile*128+i.2.1.val) (h/c.layout.kvGroup) i.1.val else zero o

def value (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h tile : Nat) : Matrix c.layout.D → α := fun i =>
  if tile*128+i.1.val<c.length b then
    input.value (c.start b+tile*128+i.1.val) (h/c.layout.kvGroup) i.2.1.val else zero o

/-- Two finite masks, full visited key tiles and the original rounded scale.
Neither an ideal hard mask nor zero-cancellation is substituted. -/
def scores (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h block tile : Nat) : Matrix 128 → α := fun i =>
  let row := block*128+i.1.val
  let col := tile*128+i.2.1.val
  let dot := o.dot precision (batch:=[]) (m:=128) (k:=c.layout.D) (n:=128)
    (query o c input b h block) (key o c input b h tile) i
  let scaled := bin o .mul (bin o .add (zero o) dot)
    (o.scalar.literal precision .real (Host.scale c.layout.D))
  let causal := if col≤row then scaled else o.scalar.literal precision .real (-1000000000)
  if (col : Int)>(row : Int)-c.layout.window then causal else o.scalar.literal precision .real (-1000000000)

/-- One source iteration. The normalizer is product/add before native staging;
its FMA contraction and both dot accumulators are retained by NativePhysical.
The p.to(v.dtype) and output same-format casts remain explicit identities. -/
def update (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h block tile : Nat) (old : Accumulator α c.layout.D) : Accumulator α c.layout.D :=
  let qk := scores o c input b h block tile
  let reduced := o.scalar.reduceMax precision ⟨1,by decide⟩ Bool.false qk
  let center : Row → α := fun i =>
    if o.eq precision .real (reduced i) (o.scalar.literal precision .real (-1000000000)) then zero o else reduced i
  let p : Matrix 128 → α := fun i => exp o (bin o .sub (qk i) (center (i.1,())))
  let total := o.scalar.reduceSum precision ⟨1,by decide⟩ Bool.false p
  let nextCenter : Row → α := fun i => bin o .max (old.center i) (center i)
  let alpha : Row → α := fun i => exp o (bin o .sub (old.center i) (nextCenter i))
  let beta : Row → α := fun i => exp o (bin o .sub (center i) (nextCenter i))
  let denominator : Row → α := fun i =>
    let n := bin o .add (bin o .mul (alpha i) (old.denominator i)) (bin o .mul (beta i) (total i))
    if o.eq precision .real n (zero o) then o.scalar.literal precision .real (1/1000000000) else n
  let pScaled : Matrix 128 → α := fun i =>
    o.scalar.cast precision .real .real (bin o .mul (p i) (bin o .div (beta (i.1,())) (denominator (i.1,()))))
  let accumulator : Matrix c.layout.D → α := fun i =>
    bin o .mul (old.value i)
      (bin o .mul (bin o .div (old.denominator (i.1,())) (denominator (i.1,()))) (alpha (i.1,())))
  let pv := o.dot precision (batch:=[]) (m:=128) (k:=128) (n:=c.layout.D)
    pScaled (value o c input b h tile)
  ⟨nextCenter,denominator,fun i => bin o .add (accumulator i) (pv i)⟩

def tiles (c : Host.Call) (b block : Nat) : Nat := if block*128<c.length b then block+1 else 0

/-- Independent online-attention recurrence over initially supplied inputs. -/
def accumulated (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h block : Nat) : Nat → Accumulator α c.layout.D
  | 0 => initial o c.layout.D
  | n+1 => update o c input b h block n (accumulated o c input b h block n)

def expected (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h row : Nat) (d : Fin c.layout.D) : α :=
  o.scalar.cast precision .real .real
    ((accumulated o c input b h (row/128) (tiles c b (row/128))).value
      (⟨row%128,Nat.mod_lt _ (by decide)⟩,d,()))

/-- The source's imperative tile loop has the same independent recurrence.
This is a loop invariant, not an equality assumed of a final tensor. -/
def loop (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h block start : Nat) (state : Accumulator α c.layout.D) : Nat → Accumulator α c.layout.D
  | 0 => state
  | n+1 => loop o c input b h block (start+1) (update o c input b h block start state) n

theorem loop_recurrence (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h block start count : Nat) :
    loop o c input b h block start (accumulated o c input b h block start) count =
      accumulated o c input b h block (start+count) := by
  induction count generalizing start with
  | zero => simp [loop]
  | succ n ih =>
    rw [loop,← accumulated,ih]
    congr 1
    omega

def sourceValue (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h row : Nat) (d : Fin c.layout.D) : α :=
  o.scalar.cast precision .real .real
    ((loop o c input b h (row/128) 0 (initial o c.layout.D) (tiles c b (row/128))).value
      (⟨row%128,Nat.mod_lt _ (by decide)⟩,d,()))

theorem source_expected (o : FPBackend.Ops α) (c : Host.Call) (input : Inputs α)
    (b h row : Nat) (d : Fin c.layout.D) : sourceValue o c input b h row d=expected o c input b h row d := by
  unfold sourceValue expected
  rw [show initial o c.layout.D=accumulated o c input b h (row/128) 0 from rfl,loop_recurrence]
  simp only [Nat.zero_add]

#axiomsClean loop_recurrence
#axiomsClean source_expected
#specNonCircular expected avoiding [FP.Structural.exec,FPHost.Composition.schedule,sourceValue,loop]
end
end VeriTile.Bench.Repairs.ContextAttnMistral.Recurrence
