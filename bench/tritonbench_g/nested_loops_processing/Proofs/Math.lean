import bench.tritonbench_g.nested_loops_processing.Proofs.Source

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing
open VeriTile.Bench.TritonBenchG.NestedLoopsProcessing
noncomputable section
open scoped Classical

def lanes : List Lane := [(0,0),(0,1),(1,0),(1,1)]

def zeroExtend (zero : α) (count : Nat) (input : Nat → α) : Nat → α :=
  fun i=>if i<count then input i else zero

/-- Independent whole-allocation target. Reuse the previously proved ordered
copy oracle, with specified zero reads beyond the actual allocation. Only the
actual output allocation is observed; unwritten cells retain their initial value. -/
def target (zero : α) (count cols : Nat) (input initial : Nat → α) : Nat → α :=
  CopyMath.expected (zeroExtend zero count input) initial cols 1

def put (count address : Nat) (value : α) (memory : Nat → α) : Nat → α :=
  if address<count then Function.update memory address value else memory

def fullTile (cols shift : Nat) (tile : Tile α) (memory : Nat → α) : Nat → α :=
  lanes.foldl (fun m lane=>Function.update m (address cols shift lane) (tile lane)) memory

def maskedTile (count cols shift : Nat) (tile : Tile α) (memory : Nat → α) : Nat → α :=
  lanes.foldl (fun m lane=>put count (address cols shift lane) (tile lane) m) memory

def uniqueTile (count cols shift : Nat) (tile : Tile α) (memory : Nat → α) : Nat → α :=
  lanes.foldl (fun m lane=>if activeLane cols lane then put count (address cols shift lane) (tile lane) m else m) memory

theorem put_idempotent (count address : Nat) (value : α) (memory : Nat → α) :
    put count address value (put count address value memory)=put count address value memory := by
  by_cases h:address<count <;> simp [put,h]

theorem active_lane_injective (cols shift : Nat) (positive : 0<cols) (a b : Lane)
    (ha : activeLane cols a) (hb : activeLane cols b)
    (same : address cols shift a=address cols shift b) : a=b := by
  rcases a with ⟨ar,ac⟩
  rcases b with ⟨br,bc⟩
  fin_cases ar <;> fin_cases ac <;> fin_cases br <;> fin_cases bc <;>
    simp [activeLane,address] at * <;> omega

theorem input_tile_duplicate (zero : α) (count : Nat) (input : Nat → α) (shift : Nat) :
    inputTile zero count 1 input shift (1,0)=inputTile zero count 1 input shift (0,1) := by
  simp [inputTile,address]

theorem unique_tile_eq_masked (count cols shift : Nat) (positive : 0<cols)
    (tile : Tile α) (duplicate : cols=1 → tile (1,0)=tile (0,1)) (memory : Nat → α) :
    uniqueTile count cols shift tile memory=maskedTile count cols shift tile memory := by
  by_cases one:cols=1
  · subst cols
    have equal:=duplicate rfl
    simp [uniqueTile,maskedTile,lanes,activeLane,address,equal,put_idempotent]
  · have wide:1<cols:=by omega
    simp [uniqueTile,maskedTile,activeLane,wide]

def Inside (count : Nat) (a b : Nat → α) : Prop := ∀ i,i<count → a i=b i

theorem put_simulates (count addr : Nat) (value : α) (a b : Nat → α) (h : Inside count a b) :
    Inside count (put count addr value a) (Function.update b addr value) := by
  intro i hi
  by_cases inside:addr<count
  · by_cases same:i=addr
    · subst i;simp [put,inside]
    · simp [put,inside,Function.update_of_ne same,h i hi]
  · have different:i≠addr:=by omega
    simp [put,inside,Function.update_of_ne different,h i hi]

theorem masked_tile_simulates (count cols shift : Nat) (tile : Tile α)
    (a b : Nat → α) (h : Inside count a b) :
    Inside count (maskedTile count cols shift tile a) (fullTile cols shift tile b) := by
  have fold : ∀ list : List Lane,∀ a b : Nat → α,Inside count a b →
      Inside count (list.foldl (fun m l=>put count (address cols shift l) (tile l) m) a)
        (list.foldl (fun m l=>Function.update m (address cols shift l) (tile l)) b) := by
    intro list
    induction list with
    | nil=>intro a b h;exact h
    | cons l ls ih=>intro a b h;exact ih _ _ (put_simulates _ _ _ _ _ h)
  exact fold lanes a b h

theorem old_offset_nat (cols shift row col : Nat) :
    (CopyMath.offset cols 1 shift row col).toNat=row*cols+(col+shift) := by
  have equal:CopyMath.offset cols 1 shift row col=((row*cols+(col+shift):Nat):Int) := by
    simp only [CopyMath.offset,Nat.cast_add,Nat.cast_mul,Int.mul_one]
  rw [equal]
  exact Int.toNat_natCast _

theorem full_tile_is_copy (zero : α) (count cols t : Nat) (input memory : Nat → α) :
    fullTile cols (2*t) (inputTile zero count cols input (CopyMath.inputShift t)) memory=
      CopyMath.copyTile (zeroExtend zero count input) cols 1 memory t := by
  simp [fullTile,lanes,inputTile,address,CopyMath.copyTile,zeroExtend,
    List.range_succ,List.foldl_append,old_offset_nat]

def logicalRun (zero : α) (count cols : Nat) (input initial : Nat → α) : Nat → α :=
  (sourceStores (inputTile zero count cols input) zero).foldl
    (fun memory store=>uniqueTile count cols store.1 store.2 memory) initial

theorem logical_readback (zero : α) (count cols : Nat) (positive : 0<cols)
    (input initial : Nat → α) :
    Inside count (logicalRun zero count cols input initial) (target zero count cols input initial) := by
  have fold : ∀ list : List Nat,∀ a b : Nat → α,Inside count a b →
      Inside count
        (list.foldl (fun memory t=>uniqueTile count cols (2*t)
          (inputTile zero count cols input (CopyMath.inputShift t)) memory) a)
        (list.foldl (CopyMath.copyTile (zeroExtend zero count input) cols 1) b) := by
    intro list
    induction list with
    | nil=>intro a b h;exact h
    | cons t ts ih=>
      intro a b h
      apply ih
      change Inside count (uniqueTile count cols (2*t)
        (inputTile zero count cols input (CopyMath.inputShift t)) a)
        (CopyMath.copyTile (zeroExtend zero count input) cols 1 b t)
      rw [unique_tile_eq_masked count cols (2*t) positive
        (inputTile zero count cols input (CopyMath.inputShift t))
        (by intro one;subst cols;exact input_tile_duplicate zero count input _) a]
      rw [←full_tile_is_copy zero count cols t input b]
      exact masked_tile_simulates _ _ _ _ a b h
  unfold logicalRun
  rw [source_store_schedule,List.foldl_map]
  exact fold (List.range 24) initial initial (fun _ _=>rfl)

/-- Signed int32 arange interpretation, including modular wrap. The Real
target uses this integer interpretation, not an unbounded natural arange. -/
def arangeWord (i : Nat) : Word := BitVec.ofNat 32 i
def signedValue (word : Word) : Int := word.toInt
def arangeReal (i : Nat) : ℝ := (signedValue (arangeWord i) : ℝ)

#axiomsClean active_lane_injective
#axiomsClean unique_tile_eq_masked
#axiomsClean logical_readback
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing
