import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Core

namespace VeriTile.Bench.Repairs.LlamaFF
open VeriTile.Bench.Repairs.LlamaFF
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

abbrev Address := Nat × Nat
abbrev Byte := BitVec 8
abbrev Stored (F H : Type) := Sum Byte (Sum F H)
abbrev Memory (F H : Type) := Address → Stored F H

structure StorageOps (F H : Type) extends Ops F where
  hzero : H
  widen : H → F
  narrow : F → H
  e5Half : Byte → H
  e5Single : Byte → F

structure View3 where
  storage : Nat
  offset : Nat
  sb : Nat
  sm : Nat
  sn : Nat
def View3.address (v : View3) (b m n : Nat) : Address :=
  (v.storage,v.offset+b*v.sb+m*v.sm+n*v.sn)
structure View2 where
  storage : Nat
  offset : Nat
  sn : Nat
  sk : Nat
def View2.address (v : View2) (n k : Nat) : Address :=
  (v.storage,v.offset+n*v.sn+k*v.sk)
structure View1 where
  storage : Nat
  offset : Nat
  stride : Nat
def View1.address (v : View1) (k : Nat) : Address := (v.storage,v.offset+k*v.stride)

/-- Single projection input/output views. Int8 storage is raw E5M2 payload. -/
structure Buffers where
  x : View3
  rms : View1
  weight : View2
  weight3 : View2
  output : View3
  useByte : Bool
def Buffers.Separate (b : Buffers) : Prop :=
  b.x.storage≠b.output.storage ∧ b.rms.storage≠b.output.storage ∧ b.weight.storage≠b.output.storage ∧ b.weight3.storage≠b.output.storage

def read32 (a : StorageOps F H) (m : Memory F H) (p : Address) : F :=
  match m p with | .inr (.inl x) => x | _ => a.zero
def read16 (a : StorageOps F H) (m : Memory F H) (p : Address) : H :=
  match m p with | .inr (.inr x) => x | _ => a.hzero
def read8 (m : Memory F H) (p : Address) : Byte :=
  match m p with | .inl x => x | _ => 0
def loadRms (a : StorageOps F H) (byte : Bool) (m : Memory F H) (p : Address) : F :=
  a.widen (if byte then a.e5Half (read8 m p) else read16 a m p)
def loadWeight (a : StorageOps F H) (byte : Bool) (m : Memory F H) (p : Address) : F :=
  a.widen (if byte then a.narrow (a.e5Single (read8 m p)) else read16 a m p)
def rowInput (a : StorageOps F H) (b : Buffers) (m : Memory F H)
    (batch row : Nat) : RowInput F where
  x := fun k => a.widen (read16 a m (b.x.address batch row k))
  rms := fun k => loadRms a b.useByte m (b.rms.address k)
  weight := fun n k => loadWeight a b.useByte m (b.weight.address n k)
  weight3 := fun n k => loadWeight a b.useByte m (b.weight3.address n k)
def encode (a : StorageOps F H) (b : Buffers) (v : F) : Stored F H :=
  .inr (.inr (a.narrow v))
abbrev Cell (s : Shape) := Fin s.B × Fin s.M × Fin s.N
def outputAddress (b : Buffers) (x : Cell s) : Address :=
  (b.output).address x.1 x.2.1 x.2.2
def InputsAgree (b : Buffers) (m m' : Memory F H) : Prop :=
  ∀ p,(p.1≠b.output.storage) → m p=m' p
def cellValue (a : StorageOps F H) (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (m : Memory F H) (x : Cell s) : Stored F H :=
  encode a b (value a.toOps p (rowInput a b m x.1 x.2.1)
    (plans x.1 x.2.1) (reductions x.1 x.2.1)
    x.2.2)
def wordTarget (a : StorageOps F H) (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (initial : Memory F H) (x : Cell s) : Stored F H :=
  encode a b (referenceValue a.toOps p (rowInput a b initial x.1 x.2.1)
    (plans x.1 x.2.1) (reductions x.1 x.2.1)
    x.2.2)
def write (m : Memory F H) (p : Address) (v : Stored F H) : Memory F H :=
  fun q => if q=p then v else m q
def run (a : StorageOps F H) (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) : List (Cell s) → Memory F H → Memory F H
  | [],m => m
  | x::xs,m => run a p b plans reductions xs (write m (outputAddress b x) (cellValue a p b plans reductions m x))

theorem rowInput_current (a : StorageOps F H) (b : Buffers) (sep : b.Separate)
    (m m' : Memory F H) (same : InputsAgree b m m') (batch row : Nat) :
    rowInput a b m batch row=rowInput a b m' batch row := by
  have hx k := same (b.x.address batch row k) sep.1
  have hr k := same (b.rms.address k) sep.2.1
  have hw n k := same (b.weight.address n k) sep.2.2.1
  have hw3 n k := same (b.weight3.address n k) sep.2.2.2
  simp only [rowInput,read32,read16,read8,loadRms,loadWeight,hx,hr,hw,hw3]
theorem cellValue_current (a : StorageOps F H) (p : Parameters F) (b : Buffers) (sep : b.Separate)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (m m' : Memory F H)
    (same : InputsAgree b m m') (x : Cell s) :
    cellValue a p b plans reductions m x=wordTarget a p b plans reductions m' x := by
  unfold cellValue wordTarget
  rw [rowInput_current a b sep m m' same,value_reference]
theorem write_inputs (b : Buffers) (m : Memory F H) (x : Cell s) (v : Stored F H) :
    InputsAgree b (write m (outputAddress b x) v) m := by
  intro p hp
  have h : p≠outputAddress b x := by
    intro he
    exact hp (congrArg Prod.fst he)
  simp [write,h]
theorem run_frame (a : StorageOps F H) (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (events : List (Cell s)) (m : Memory F H)
    (q : Address) (outside : ∀ x : Cell s,q≠outputAddress b x) :
    run a p b plans reductions events m q=m q := by
  induction events generalizing m with
  | nil => rfl
  | cons x xs ih => rw [run,ih]; simp [write,outside]
theorem run_output (a : StorageOps F H) (p : Parameters F) (b : Buffers) (sep : b.Separate)
    (injective : Function.Injective (outputAddress b : Cell s → Address))
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (events : List (Cell s))
    (initial m : Memory F H) (same : InputsAgree b m initial) (x : Cell s) :
    run a p b plans reductions events m (outputAddress b x)=
      if x∈events then wordTarget a p b plans reductions initial x else m (outputAddress b x) := by
  induction events generalizing m with
  | nil => simp [run]
  | cons e es ih =>
    have next : InputsAgree b (write m (outputAddress b e) (cellValue a p b plans reductions m e)) initial := by
      intro q hq
      rw [write_inputs b m e _ q hq]
      exact same q hq
    rw [run,ih _ next]
    by_cases tail : x∈es
    · simp [tail]
    · by_cases eq : x=e
      · subst e
        simp [tail,write,cellValue_current a p b sep plans reductions m initial same]
      · have different : outputAddress b x≠outputAddress b e := fun h => eq (injective h)
        simp [tail,eq,write,different]

/-- Runtime schedule certificate concerns identities only. Arbitrary program
interleavings are allowed because all reads are from preserved input storage. -/
structure Schedule (s : Shape) where
  events : List (Cell s)
  complete : ∀ x,x∈events
  unique : events.Nodup
theorem whole_word_and_frame (a : StorageOps F H) (p : Parameters F) (b : Buffers) (sep : b.Separate)
    (injective : Function.Injective (outputAddress b : Cell s → Address))
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (schedule : Schedule s) (initial : Memory F H) :
    (∀ x : Cell s,run a p b plans reductions schedule.events initial (outputAddress b x)=
      wordTarget a p b plans reductions initial x) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) → run a p b plans reductions schedule.events initial q=initial q) := by
  constructor
  · intro x
    rw [run_output a p b sep injective plans reductions schedule.events initial initial (fun _ _ => rfl),if_pos (schedule.complete x)]
  · exact run_frame a p b plans reductions schedule.events initial

#axiomsClean rowInput_current
#axiomsClean cellValue_current
#axiomsClean run_frame
#axiomsClean run_output
#axiomsClean whole_word_and_frame
end
end VeriTile.Bench.Repairs.LlamaFF
