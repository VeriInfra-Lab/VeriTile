import bench.tritonbench_g.chunk_linear_attn.NativeTerms
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable (decode : BitVec 32 → ℝ)

def mapCell (c : Cell Term) : Cell ℝ := .mk .real ((c.read .real).meaning decode)
def mapMemory (m : Memory Term) : Memory ℝ := fun r o => mapCell decode (m r o)
def mapMatrix {m n : Nat} (a : Matrix Term m n) : Matrix ℝ m n := fun i => (a i).meaning decode

@[simp] theorem map_read (m : Memory Term) (r : RegionName) (o : Nat) :
    (mapMemory decode m r o).read .real=((m r o).read .real).meaning decode := rfl
@[simp] theorem map_mk (x : Term) : mapCell decode (.mk .real x)=.mk .real (x.meaning decode) := rfl
@[simp] theorem map_zero : (zero algebra).meaning decode=zero Real.algebra := rfl
@[simp] theorem map_scalar (x : ℝ) : (scalar algebra x).meaning decode=scalar Real.algebra x := rfl
@[simp] theorem map_add (x y : Term) : (add algebra x y).meaning decode=add Real.algebra (x.meaning decode) (y.meaning decode) := rfl
@[simp] theorem map_mul (x y : Term) : (mul algebra x y).meaning decode=mul Real.algebra (x.meaning decode) (y.meaning decode) := rfl
@[simp] theorem map_cast (x : Term) : (cast algebra x).meaning decode=cast Real.algebra (x.meaning decode) := rfl
@[simp] theorem map_if (p : Prop) [Decidable p] (a b : Term) :
    (if p then a else b).meaning decode=if p then a.meaning decode else b.meaning decode := by split <;> rfl
@[simp] theorem map_dot {m k n : Nat} (a : Matrix Term m k) (b : Matrix Term k n) (i : TileIndex [m,n]) :
    (dot dotPrimitive a b i).meaning decode=dot Real.dotPrimitive (mapMatrix decode a) (mapMatrix decode b) i := rfl
@[simp] theorem map_rawLoad (m : Memory Term) (r : RegionName) (base rows cols sr sc ro co : Nat)
    {br bc : Nat} (i : TileIndex [br,bc]) :
    (rawLoad algebra m r base rows cols sr sc ro co i).meaning decode=
      rawLoad Real.algebra (mapMemory decode m) r base rows cols sr sc ro co i := by
  simp only [rawLoad,map_if,map_zero,map_read]
@[simp] theorem map_load (m : Memory Term) (r : RegionName) (base rows cols sr sc ro co : Nat)
    {br bc : Nat} (i : TileIndex [br,bc]) :
    (load algebra m r base rows cols sr sc ro co i).meaning decode=
      load Real.algebra (mapMemory decode m) r base rows cols sr sc ro co i := map_rawLoad decode m r base rows cols sr sc ro co i

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
    mapMemory decode (scatter xs dest active offset value m)=
      scatter xs dest active offset (fun i => mapCell decode (value i)) (mapMemory decode m) := by
  apply map_fold
  intro i mem
  funext r o
  by_cases hi : active i <;> by_cases hr : r=dest ∧ o=offset i <;>
    simp [hi,hr,mapMemory]

@[simp] theorem map_stateInitial (p : Parameters) (pid : PID) (m : Memory Term) (rev seed : Bool) :
    mapMatrix decode (stateInitial algebra p pid m rev seed)=
      stateInitial Real.algebra p pid (mapMemory decode m) rev seed := by
  funext i
  simp only [mapMatrix,stateInitial,map_if,map_cast,map_rawLoad,map_zero]
@[simp] theorem map_stateLeft (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (rev : Bool) (n : Nat) :
    mapMatrix decode (stateLeft algebra p pid m scale rev n)=
      stateLeft Real.algebra p pid (mapMemory decode m) scale rev n := by
  funext i
  cases rev <;> simp only [mapMatrix,stateLeft,Bool.false_eq_true,if_false,if_true,map_load,map_cast,map_mul,map_scalar]
@[simp] theorem map_stateRight (p : Parameters) (pid : PID) (m : Memory Term) (rev : Bool) (n : Nat) :
    mapMatrix decode (stateRight algebra p pid m rev n)=
      stateRight Real.algebra p pid (mapMemory decode m) rev n := by
  funext i
  cases rev <;> simp only [mapMatrix,stateRight,Bool.false_eq_true,if_false,if_true,map_load,map_cast]
@[simp] theorem map_stateNext (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (rev : Bool)
    (n : Nat) (h : HValues p Term) :
    mapMatrix decode (stateNext algebra dotPrimitive p pid m scale rev n h)=
      stateNext Real.algebra Real.dotPrimitive p pid (mapMemory decode m) scale rev n (mapMatrix decode h) := by
  funext i
  simp only [mapMatrix,stateNext,map_add,map_dot,map_stateLeft,map_stateRight]
@[simp] theorem map_stateValues (p : Parameters) (pid : PID) (m : Memory Term) (scale : ℝ) (rev seed : Bool) (n : Nat) :
    mapMatrix decode (stateValues algebra dotPrimitive p pid m scale rev seed n)=
      stateValues Real.algebra Real.dotPrimitive p pid (mapMemory decode m) scale rev seed n := by
  induction n with
  | zero => exact map_stateInitial decode p pid m rev seed
  | succ n ih => simp only [stateValues,map_stateNext,ih]
@[simp] theorem map_writeHistory (p : Parameters) (pid : PID) (rev : Bool) (n : Nat)
    (h : HValues p Term) (m : Memory Term) :
    mapMemory decode (writeHistory algebra p pid rev n h m)=
      writeHistory Real.algebra p pid rev n (mapMatrix decode h) (mapMemory decode m) := by
  simp only [writeHistory,map_scatter,map_mk,map_cast,mapMatrix]
@[simp] theorem map_writeFinal (p : Parameters) (pid : PID) (rev final : Bool)
    (h : HValues p Term) (m : Memory Term) :
    mapMemory decode (writeStateFinal algebra p pid rev final h m)=
      writeStateFinal Real.algebra p pid rev final (mapMatrix decode h) (mapMemory decode m) := by
  cases final <;> simp only [writeStateFinal,Bool.false_eq_true,if_false,if_true,map_scatter,map_mk,map_cast,mapMatrix]
 theorem map_stateMemory (p : Parameters) (pid : PID) (input current : Memory Term)
    (scale : ℝ) (rev seed final : Bool) :
    mapMemory decode (stateMemory algebra dotPrimitive p pid input current scale rev seed final)=
      stateMemory Real.algebra Real.dotPrimitive p pid (mapMemory decode input) (mapMemory decode current) scale rev seed final := by
  simp only [stateMemory,map_writeFinal,map_stateValues]
  congr 1
  apply map_range
  intro n mem
  simp only [map_writeHistory,map_stateValues]

#axiomsClean map_stateMemory
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.NativeTerms
