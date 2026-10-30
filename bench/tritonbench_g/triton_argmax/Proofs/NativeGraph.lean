import bench.tritonbench_g.triton_argmax.Proofs.Admission
import bench.tritonbench_g.triton_argmax.Proofs.SegmentedExecution
import bench.tritonbench_g.triton_argmax.Proofs.ScalarMacroGraph
import bench.tritonbench_g.triton_argmax.Proofs.ReductionIR

namespace VeriTile.Bench.Repairs.TritonArgmax.NativeGraph
open Source Memory Reduction Revision004
noncomputable section

structure DirectPrimitives (d : MidFormat) where
  cast : Word → MidWord d
  greater : MidWord d → MidWord d → Bool
  equal : MidWord d → MidWord d → Bool
  padding : MidWord d

/-- Only scalar instruction ports. No reduction, loop, memory, or result port. -/
structure Primitives where
  direct : (d : MidFormat) → DirectPrimitives d
  integer : (kind : Formats.IntegerKind) → ScalarBinding.IntegerPrimitives kind

def scalars (p : Primitives) : (a : RealScratch.Adapter) → Scalars a.format
  | .directCast d => ⟨ScalarBinding.gt32, ScalarBinding.eq32, 0xff800000,
      (p.direct d.format).cast, (p.direct d.format).greater,
      (p.direct d.format).equal, (p.direct d.format).padding⟩
  | .integerSafe k => ScalarBinding.scalars k (p.integer k)

def nativeStore (p : Primitives) : (a : RealScratch.Adapter) → Word → MidWord a.format
  | .directCast d, x => (p.direct d.format).cast x
  | .integerSafe k, x =>
      ((ConversionMacro.SaturatingConversion.mk k).emittedStore (p.integer k) x).cast
        (Formats.format_width k)

def ZeroClamps (p : Primitives) : Prop :=
  ∀ k, ConversionMacro.ZeroClamp k (p.integer k)

theorem native_store (p : Primitives) (zero : ZeroClamps p)
    (a : RealScratch.Adapter) (x : Word) :
    nativeStore p a x = (scalars p a).storeCast x := by
  cases a with
  | directCast d => rfl
  | integerSafe k =>
    simp only [nativeStore, ConversionMacro.emitted_source _ _ (zero k)]
    rfl

/-! Native load representation: narrow words occupy at least 32-bit registers;
signed loads extend the sign, unsigned loads extend with zero, and bool loads
normalize the stored byte. Every register bit participates in comparisons. -/
namespace Register

def bits (d : MidFormat) : Nat := max 32 (8*d.bytes)
abbrev Value (a : RealScratch.Adapter) := BitVec (bits a.format)

def encode : (a : RealScratch.Adapter) → MidWord a.format → Value a
  | .directCast d, word => word.setWidth (bits d.format)
  | .integerSafe k, word =>
    if k = .boolean then (if word.toNat = 0 then 0 else 1)
    else if k.signed then word.signExtend (bits (Formats.format k))
    else word.setWidth (bits (Formats.format k))

def StoredValue : RealScratch.Adapter → Type
  | .directCast d => MidWord d.format
  | .integerSafe _ => Int

def view : (a : RealScratch.Adapter) → Value a → StoredValue a
  | .directCast d, word => word.setWidth (8*d.format.bytes)
  | .integerSafe k, word => if k.signed then word.toInt else Int.ofNat word.toNat

def decode : (a : RealScratch.Adapter) → MidWord a.format → StoredValue a
  | .directCast _, word => word
  | .integerSafe k, word => Formats.decodeStored k word

theorem view_encode (a : RealScratch.Adapter) (word : MidWord a.format) :
    view a (encode a word) = decode a word := by
  cases a with
  | directCast d =>
    exact (BitVec.setWidth_setWidth_of_le word (Nat.le_max_right 32 (8*d.format.bytes))).trans
      (BitVec.setWidth_eq word)
  | integerSafe k =>
    cases k with
    | i8 | i16 | i32 | i64 => exact BitVec.toInt_signExtend_of_le (by decide)
    | u8 | u16 | u32 | u64 =>
      exact congrArg Int.ofNat (BitVec.toNat_setWidth_of_le (b := word) (by decide))
    | boolean =>
      change (if word.toNat = 0 then (0 : BitVec 32) else 1).toInt =
        if word.toNat = 0 then 0 else 1
      split <;> rfl

def storedGreater (p : Primitives) : (a : RealScratch.Adapter) → StoredValue a → StoredValue a → Bool
  | .directCast d => (p.direct d.format).greater
  | .integerSafe _ => (orderedGt : Int → Int → Bool)

def storedEqual (p : Primitives) : (a : RealScratch.Adapter) → StoredValue a → StoredValue a → Bool
  | .directCast d => (p.direct d.format).equal
  | .integerSafe _ => (orderedEq : Int → Int → Bool)

def greater (p : Primitives) (a : RealScratch.Adapter) (x y : Value a) : Bool :=
  storedGreater p a (view a x) (view a y)

def equal (p : Primitives) (a : RealScratch.Adapter) (x y : Value a) : Bool :=
  storedEqual p a (view a x) (view a y)

theorem compare_greater (p : Primitives) (a : RealScratch.Adapter) (x y : MidWord a.format) :
    greater p a (encode a x) (encode a y) = (scalars p a).gtMid x y := by
  simp only [greater,view_encode]
  cases a <;> rfl

theorem compare_equal (p : Primitives) (a : RealScratch.Adapter) (x y : MidWord a.format) :
    equal p a (encode a x) (encode a y) = (scalars p a).eqMid x y := by
  simp only [equal,view_encode]
  cases a <;> rfl

theorem tree_indices (p : Primitives) (a : RealScratch.Adapter)
    (tree : Reduction.Tree n) (words : Fin n → MidWord a.format) :
    tree.pick (greater p a) (equal p a) (fun i => encode a (words i)) =
      tree.pick (scalars p a).gtMid (scalars p a).eqMid words := by
  rw [← Formats.pick_map]
  congr 1
  · funext x y; exact compare_greater p a x y
  · funext x y; exact compare_equal p a x y

theorem selected (p : Primitives) (a : RealScratch.Adapter)
    (program : ReductionIR.Program (bits a.format) n) (words : Fin n → MidWord a.format) :
    (program.nativeRun (greater p a) (equal p a) (fun i => encode a (words i))).2.toNat =
      (program.tree.pick (scalars p a).gtMid (scalars p a).eqMid words).val := by
  rw [(program.native_evaluated _ _ _).2]
  exact congrArg Fin.val (tree_indices p a program.tree words)

#axiomsClean view_encode
#axiomsClean selected
end Register

structure GlobalPrograms (d : MidFormat) (o : Checked.Observation) where
  first : Nat → ReductionIR.Program 32 (Checked.block o)
  second : ReductionIR.Program (Register.bits d) (Checked.midBlock o)

def GlobalPrograms.config (g : GlobalPrograms d o) : GlobalConfig :=
  ⟨o.input.numel,Checked.block o,Checked.blocks o,Checked.midBlock o,
   fun pid => (g.first pid).tree,g.second.tree⟩

def first (p : Primitives) (a : RealScratch.Adapter) (g : GlobalPrograms a.format o)
    (mem : Memory.Primitives M) (buffers : GlobalMemory.Buffers) (s : M) (pid : Nat) : M :=
  let ops := scalars p a
  let loaded := g.config.loaded (GlobalMemory.input mem buffers s) ops.negInf32 pid
  let selected := (g.first pid).nativeRun ops.gt32 ops.eq32 loaded
  mem.write 8
    (mem.write a.format.bytes s (address buffers.midValue a.format.bytes pid)
      (nativeStore p a selected.1))
    (address buffers.midIndex 8 pid)
    (BitVec.ofNat 64 (selected.2.toNat + pid*Checked.block o))

def second (p : Primitives) (a : RealScratch.Adapter) (g : GlobalPrograms a.format o)
    (mem : Memory.Primitives M) (buffers : GlobalMemory.Buffers) (s : M) : M :=
  let ops := scalars p a
  let loaded := fun lane : Fin (Checked.midBlock o) =>
    if lane.val < Checked.blocks o then
      mem.read a.format.bytes s (address buffers.midValue a.format.bytes lane.val)
    else ops.negInfMid
  let selected := (g.second.nativeRun (Register.greater p a) (Register.equal p a) (fun i => Register.encode a (loaded i))).2.toNat
  mem.write 8 s (address buffers.output 8 0)
    (mem.read 8 s (address buffers.midIndex 8 selected))

def global (p : Primitives) (a : RealScratch.Adapter) (g : GlobalPrograms a.format o)
    (mem : Memory.Primitives M) (buffers : GlobalMemory.Buffers) (s : M) : M :=
  second p a g mem buffers
    ((SegmentedExecution.schedule 0 (Checked.blocks o)).foldl (first p a g mem buffers)
      (Host.initializeIndices mem buffers (Checked.midBlock o) s))

theorem first_source (p : Primitives) (zero : ZeroClamps p) (a : RealScratch.Adapter)
    (g : GlobalPrograms a.format o) (mem : Memory.Primitives M)
    (buffers : GlobalMemory.Buffers) (s : M) (pid : Nat) :
    first p a g mem buffers s pid =
      GlobalMemory.firstLive mem (scalars p a) g.config buffers s pid := by
  have selected := (g.first pid).native_evaluated (scalars p a).gt32 (scalars p a).eq32
    (g.config.loaded (GlobalMemory.input mem buffers s) (scalars p a).negInf32 pid)
  simp only [first, selected.1, selected.2, native_store p zero]
  rfl

theorem second_source (p : Primitives) (a : RealScratch.Adapter)
    (g : GlobalPrograms a.format o) (mem : Memory.Primitives M)
    (buffers : GlobalMemory.Buffers) (s : M) :
    second p a g mem buffers s =
      GlobalMemory.secondProgram mem (scalars p a) g.config buffers s := by
  have selected := Register.selected p a g.second
    (fun lane : Fin (Checked.midBlock o) =>
      if lane.val < Checked.blocks o then
        mem.read a.format.bytes s (address buffers.midValue a.format.bytes lane.val)
      else (scalars p a).negInfMid)
  simp only [second,selected]
  rfl

theorem global_source (p : Primitives) (zero : ZeroClamps p) (a : RealScratch.Adapter)
    (g : GlobalPrograms a.format o) (mem : Memory.Primitives M)
    (buffers : GlobalMemory.Buffers) (s : M) :
    global p a g mem buffers s =
      GlobalMemory.run mem (scalars p a) g.config buffers
        (Host.initializeIndices mem buffers g.config.midBlock s) := by
  have each : first p a g mem buffers = GlobalMemory.firstLive mem (scalars p a) g.config buffers :=
    funext fun state => funext fun pid => first_source p zero a g mem buffers state pid
  simp only [global,each,second_source,GlobalMemory.run,GlobalMemory.firstGrid,SegmentedExecution.schedule_zero]
  rfl

def firstPair (p : Primitives) (a : RealScratch.Adapter) (g : GlobalPrograms a.format o)
    (bits : Nat → Word) (pid : Nat) :=
  (g.first pid).nativeRun (scalars p a).gt32 (scalars p a).eq32
    (g.config.loaded bits (scalars p a).negInf32 pid)

def firstValue (p : Primitives) (a : RealScratch.Adapter) (g : GlobalPrograms a.format o)
    (bits : Nat → Word) (pid : Nat) :=
  nativeStore p a (firstPair p a g bits pid).1

def firstIndex (p : Primitives) (a : RealScratch.Adapter) (g : GlobalPrograms a.format o)
    (bits : Nat → Word) (pid : Nat) :=
  (firstPair p a g bits pid).2.toNat+pid*Checked.block o

/-- Independent native output graph: both emitted reductions, intermediate
conversion, typed padding, and the initialized-index gather are retained. -/
def globalTarget (p : Primitives) (a : RealScratch.Adapter) (g : GlobalPrograms a.format o)
    (bits : Nat → Word) : Nat :=
  let loaded := fun lane : Fin (Checked.midBlock o) =>
    if lane.val < Checked.blocks o then firstValue p a g bits lane.val else (scalars p a).negInfMid
  let selected := (g.second.nativeRun (Register.greater p a) (Register.equal p a) (fun i => Register.encode a (loaded i))).2.toNat
  if selected < Checked.blocks o then firstIndex p a g bits selected else 0

theorem global_target (p : Primitives) (zero : ZeroClamps p) (a : RealScratch.Adapter)
    (g : GlobalPrograms a.format o) (bits : Nat → Word) :
    globalTarget p a g bits = Source.globalIndex (scalars p a) g.config bits := by
  have value (pid : Nat) : firstValue p a g bits pid =
      Source.stage1Value (scalars p a) g.config bits pid := by
    have step := (g.first pid).native_evaluated (scalars p a).gt32 (scalars p a).eq32
      (g.config.loaded bits (scalars p a).negInf32 pid)
    simp only [firstValue,firstPair,step.1,native_store p zero]
    rfl
  have index (pid : Nat) : firstIndex p a g bits pid =
      Source.stage1Index (scalars p a) g.config bits pid := by
    have step := (g.first pid).native_evaluated (scalars p a).gt32 (scalars p a).eq32
      (g.config.loaded bits (scalars p a).negInf32 pid)
    simp only [firstIndex,firstPair,step.2]
    rfl
  unfold globalTarget
  simp only [value,index]
  have step := Register.selected p a g.second
    (fun lane : Fin (Checked.midBlock o) => if lane.val < Checked.blocks o then
      Source.stage1Value (scalars p a) g.config bits lane.val else (scalars p a).negInfMid)
  rw [step]
  rfl

structure DimensionPrograms where
  tileProgram : Nat → Nat → Nat → ReductionIR.Program 32 128

def DimensionPrograms.config (g : DimensionPrograms) (o : Checked.Observation) (ax : Nat) : DimConfig :=
  ⟨Checked.outer o ax,Checked.extent o ax,Checked.inner o ax,
   fun row k tile => (g.tileProgram row k tile).tree⟩

def dimLocal (g : DimensionPrograms) (c : DimConfig)
    (bits : Nat → Word) (row k tile : Nat) : Word × Nat :=
  let loaded := c.loaded bits 0xff800000 row k tile
  let selected := (g.tileProgram row k tile).nativeRun ScalarBinding.gt32 ScalarBinding.eq32 loaded
  (selected.1,tile*128+selected.2.toNat)

def dimStep (g : DimensionPrograms) (c : DimConfig)
    (bits : Nat → Word) (row k : Nat) (current : Word × Nat) (tile : Nat) : Word × Nat :=
  let tileProgram := dimLocal g c bits row k tile
  if ScalarBinding.gt32 tileProgram.1 current.1 then tileProgram else current

def dimIndex (g : DimensionPrograms) (c : DimConfig)
    (bits : Nat → Word) (row k : Nat) : Nat :=
  ((List.range ((c.extent+127)/128)).foldl (dimStep g c bits row k) (0xff800000,0)).2

def dimension (g : DimensionPrograms) (o : Checked.Observation) (ax : Nat)
    (mem : Memory.Primitives M) (buffers : DimMemory.Buffers) (s : M) : M :=
  let c := g.config o ax
  (SegmentedExecution.dimJobs c).foldl (fun state job =>
    if job.1 < c.outer then mem.write 8 state (address buffers.output 8 (job.1*c.inner+job.2))
      (BitVec.ofNat 64 (dimIndex g c (DimMemory.input mem buffers state) job.1 job.2))
    else state) s

def dimScalars : Scalars (RealScratch.FloatingKind.fp32.format) :=
  ⟨ScalarBinding.gt32,ScalarBinding.eq32,0xff800000,id,
    ScalarBinding.gt32,ScalarBinding.eq32,0xff800000⟩

theorem dimension_index (g : DimensionPrograms) (o : Checked.Observation) (ax : Nat)
    (bits : Nat → Word) (row k : Nat) :
    dimIndex g (g.config o ax) bits row k =
      Source.dimIndex dimScalars (g.config o ax) bits row k := by
  have tileProgram : ∀ tile, dimLocal g (g.config o ax) bits row k tile =
      Source.dimLocal dimScalars (g.config o ax) bits row k tile := by
    intro tile
    have selected := (g.tileProgram row k tile).native_evaluated ScalarBinding.gt32 ScalarBinding.eq32
      ((g.config o ax).loaded bits 0xff800000 row k tile)
    simp only [dimLocal,selected.1,selected.2]
    rfl
  have step : dimStep g (g.config o ax) bits row k =
      Source.dimStep dimScalars (g.config o ax) bits row k := by
    funext current tile
    simp only [dimStep,tileProgram,Source.dimStep,dimScalars]
  simp only [dimIndex,Source.dimIndex,Source.dimTiles,Source.blockN,dimScalars,step,
    show (g.config o ax).extent+128-1 = (g.config o ax).extent+127 by omega]

theorem dimension_source (g : DimensionPrograms) (o : Checked.Observation) (ax : Nat)
    (mem : Memory.Primitives M) (buffers : DimMemory.Buffers) (s : M) :
    dimension g o ax mem buffers s =
      DimMemory.run mem dimScalars (g.config o ax) buffers s := by
  simp only [dimension,dimension_index,SegmentedExecution.dimJobs_eq]
  rfl

#axiomsClean native_store
#axiomsClean global_source
#axiomsClean global_target
#axiomsClean dimension_source
end
end VeriTile.Bench.Repairs.TritonArgmax.NativeGraph
