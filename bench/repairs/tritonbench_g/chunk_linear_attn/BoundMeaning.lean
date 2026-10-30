import bench.repairs.tritonbench_g.chunk_linear_attn.NativeTerms
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.NativeBound
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
open NativeTerms
def bindScale (word : BitVec 32) : Term → Term
  | .bits b => .bits b
  | .literal x => if x=1 then .bits word else .literal x
  | .add a b => .add (bindScale word a) (bindScale word b)
  | .mul a b => .mul (bindScale word a) (bindScale word b)
  | .cast a => .cast (bindScale word a)
  | .load a => .load (bindScale word a)
  | .dot k a b => .dot k (fun e => bindScale word (a e)) (fun e => bindScale word (b e))
def boundMeaning (decode : BitVec 32 → ℝ) (word : BitVec 32) (t : Term) :=
  (bindScale word t).meaning decode
def scaleMeaning (decode : BitVec 32 → ℝ) (word : BitVec 32) (x : ℝ) := if x=1 then decode word else x

variable (decode : BitVec 32 → ℝ) (word : BitVec 32)

def mapCell (c : Cell Term) : Cell ℝ := .mk .real (boundMeaning decode word (c.read .real))
def mapMemory (m : Memory Term) : Memory ℝ := fun r o => mapCell decode word (m r o)
def mapMatrix {m n : Nat} (a : Matrix Term m n) : Matrix ℝ m n := fun i => boundMeaning decode word (a i)

@[simp] theorem map_read (m : Memory Term) (r : RegionName) (o : Nat) :
    (mapMemory decode word m r o).read .real=boundMeaning decode word ((m r o).read .real) := rfl
@[simp] theorem map_mk (x : Term) : mapCell decode word (.mk .real x)=.mk .real ((boundMeaning decode word x)) := rfl
@[simp] theorem map_zero : (boundMeaning decode word (zero algebra))=zero Real.algebra := by simp [boundMeaning,bindScale,zero,algebra,Term.meaning,Real.algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra]
@[simp] theorem map_scalar (x : ℝ) : (boundMeaning decode word (scalar algebra x))=scalar Real.algebra (scaleMeaning decode word x) := by
  unfold boundMeaning scalar algebra bindScale scaleMeaning
  split <;> rfl
@[simp] theorem map_add (x y : Term) : (boundMeaning decode word (add algebra x y))=add Real.algebra ((boundMeaning decode word x)) ((boundMeaning decode word y)) := rfl
@[simp] theorem map_mul (x y : Term) : (boundMeaning decode word (mul algebra x y))=mul Real.algebra ((boundMeaning decode word x)) ((boundMeaning decode word y)) := rfl
@[simp] theorem map_cast (x : Term) : (boundMeaning decode word (cast algebra x))=cast Real.algebra ((boundMeaning decode word x)) := rfl
@[simp] theorem map_if (p : Prop) [Decidable p] (a b : Term) :
    (boundMeaning decode word (if p then a else b))=if p then (boundMeaning decode word a) else (boundMeaning decode word b) := by split <;> rfl
@[simp] theorem map_dot {m k n : Nat} (a : Matrix Term m k) (b : Matrix Term k n) (i : TileIndex [m,n]) :
    (boundMeaning decode word (dot dotPrimitive a b i))=dot Real.dotPrimitive (mapMatrix decode word a) (mapMatrix decode word b) i := rfl
@[simp] theorem map_rawLoad (m : Memory Term) (r : RegionName) (base rows cols sr sc ro co : Nat)
    {br bc : Nat} (i : TileIndex [br,bc]) :
    (boundMeaning decode word (rawLoad algebra m r base rows cols sr sc ro co i))=
      rawLoad Real.algebra (mapMemory decode word m) r base rows cols sr sc ro co i := by
  simp only [rawLoad,map_if,map_zero,map_read]
@[simp] theorem map_load (m : Memory Term) (r : RegionName) (base rows cols sr sc ro co : Nat)
    {br bc : Nat} (i : TileIndex [br,bc]) :
    (boundMeaning decode word (load algebra m r base rows cols sr sc ro co i))=
      load Real.algebra (mapMemory decode word m) r base rows cols sr sc ro co i := map_rawLoad decode word m r base rows cols sr sc ro co i

 theorem map_fold {α β ι : Type} (f : α → β) (a : ι → α → α) (b : ι → β → β)
    (h : ∀ i x, f (a i x)=b i (f x)) (xs : List ι) (x : α) :
    f (xs.foldl (fun x i => a i x) x)=xs.foldl (fun x i => b i x) (f x) := by
  induction xs generalizing x with
  | nil => rfl
  | cons i xs ih => simpa only [List.foldl_cons,h] using ih (a i x)
 theorem map_range {α β : Type} (f : α → β) (a : Nat → α → α) (b : Nat → β → β)
    (h : ∀ i x, f (a i x)=b i (f x)) (n : Nat) (x : α) :
    f (foldRange a 0 n 1 x)=foldRange b 0 n 1 (f x) := by
  induction n with
  | zero => simp only [foldRange_zero]
  | succ n ih => simp only [foldRange_succ,h,ih]
 theorem map_scatter {ι : Type} (xs : List ι) (dest : RegionName) (active : ι → Prop)
    [DecidablePred active] (offset : ι → Nat) (value : ι → Cell Term) (m : Memory Term) :
    mapMemory decode word (scatter xs dest active offset value m)=
      scatter xs dest active offset (fun i => mapCell decode word (value i)) (mapMemory decode word m) := by
  apply map_fold
  intro i mem
  funext r o
  by_cases hi : active i <;> by_cases hr : r=dest ∧ o=offset i <;>
    simp [hi,hr,mapMemory]

@[simp] theorem map_stateInitial (p : Parameters) (pid : PID) (m : Memory Term) (rev seed : Bool) :
    mapMatrix decode word (stateInitial algebra p pid m rev seed)=
      stateInitial Real.algebra p pid (mapMemory decode word m) rev seed := by
  funext i
  simp only [mapMatrix,stateInitial,map_if,map_cast,map_rawLoad,map_zero]
@[simp] theorem map_stateLeft (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (rev : Bool) (n : Nat) :
    mapMatrix decode word (stateLeft algebra p pid m scale rev n)=
      stateLeft Real.algebra p pid (mapMemory decode word m) (scaleMeaning decode word scale) rev n := by
  funext i
  cases rev <;> simp only [mapMatrix,stateLeft,Bool.false_eq_true,if_false,if_true,map_load,map_cast,map_mul,map_scalar]
@[simp] theorem map_stateRight (p : Parameters) (pid : PID) (m : Memory Term) (rev : Bool) (n : Nat) :
    mapMatrix decode word (stateRight algebra p pid m rev n)=
      stateRight Real.algebra p pid (mapMemory decode word m) rev n := by
  funext i
  cases rev <;> simp only [mapMatrix,stateRight,Bool.false_eq_true,if_false,if_true,map_load,map_cast]
@[simp] theorem map_stateNext (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (rev : Bool)
    (n : Nat) (h : HValues p Term) :
    mapMatrix decode word (stateNext algebra dotPrimitive p pid m scale rev n h)=
      stateNext Real.algebra Real.dotPrimitive p pid (mapMemory decode word m) (scaleMeaning decode word scale) rev n (mapMatrix decode word h) := by
  funext i
  simp only [mapMatrix,stateNext,map_add,map_dot,map_stateLeft,map_stateRight]
@[simp] theorem map_stateValues (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (rev seed : Bool) (n : Nat) :
    mapMatrix decode word (stateValues algebra dotPrimitive p pid m scale rev seed n)=
      stateValues Real.algebra Real.dotPrimitive p pid (mapMemory decode word m) (scaleMeaning decode word scale) rev seed n := by
  induction n with
  | zero => exact map_stateInitial decode word p pid m rev seed
  | succ n ih => simp only [stateValues,map_stateNext,ih]
@[simp] theorem map_writeHistory (p : Parameters) (pid : PID) (rev : Bool) (n : Nat)
    (h : HValues p Term) (m : Memory Term) :
    mapMemory decode word (writeHistory algebra p pid rev n h m)=
      writeHistory Real.algebra p pid rev n (mapMatrix decode word h) (mapMemory decode word m) := by
  simp only [writeHistory,map_scatter,map_mk,map_cast,mapMatrix]
@[simp] theorem map_writeFinal (p : Parameters) (pid : PID) (rev final : Bool)
    (h : HValues p Term) (m : Memory Term) :
    mapMemory decode word (writeStateFinal algebra p pid rev final h m)=
      writeStateFinal Real.algebra p pid rev final (mapMatrix decode word h) (mapMemory decode word m) := by
  cases final <;> simp only [writeStateFinal,Bool.false_eq_true,if_false,if_true,map_scatter,map_mk,map_cast,mapMatrix]
 theorem map_stateMemory (p : Parameters) (pid : PID) (input current : Memory Term)
    (scale : ℝ) (rev seed final : Bool) :
    mapMemory decode word (stateMemory algebra dotPrimitive p pid input current scale rev seed final)=
      stateMemory Real.algebra Real.dotPrimitive p pid (mapMemory decode word input) (mapMemory decode word current) (scaleMeaning decode word scale) rev seed final := by
  simp only [stateMemory,map_writeFinal,map_stateValues]
  congr 1
  apply map_range
  intro n mem
  simp only [map_writeHistory,map_stateValues]

#axiomsClean map_stateMemory
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.NativeBound
