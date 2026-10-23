import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.NativeAddress
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Memory

namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Native
open VeriTile.Triton
open Bytes
noncomputable section
set_option maxHeartbeats 6000000
abbrev Bytes := DequantizeRowwise.FP32Prior.Bytes.Memory

abbrev signed64 := NativeAddress.signed64
abbrev int32 := NativeAddress.int32
theorem signed64_exact (n : Int) (lo : -(2^63) ≤ n) (hi : n < 2^63) : signed64 n = n :=
  NativeAddress.signed64_exact n lo hi
theorem int32_exact (n : Int) (lo : -(2^31) ≤ n) (hi : n < 2^31) : int32 n = n :=
  NativeAddress.int32_exact n lo hi

/-- Ordinary positive-block/grid admission, plus the concrete successful
allocation extents. No extra native-safe dimension precondition is added. -/
structure Bounds (p : Parameters) (xb sb ob scaleCount : Nat) : Prop where
  grid : p.rows<2^31
  width : p.width≤1048576
  scaleRows : p.rows≤ scaleCount
  xSize : 4*(p.rows*p.cols)≤2^63-1
  xCapacity : xb+4*(p.rows*p.cols)≤2^64
  scaleCapacity : sb+4*scaleCount≤2^64
  outputCapacity : ob+2*(p.rows*p.cols)≤2^64

abbrev pointer := NativeAddress.pointer
theorem pointer_exact (bytes base count index : Nat) (positive : 0 < bytes)
    (capacity : base + bytes * count ≤ 2^64) (bound : index < count) :
    pointer bytes base index = base + bytes * index :=
  NativeAddress.pointer_exact bytes base count index positive capacity bound

def offset (cols pid lane : Nat) : Int :=
  signed64 (signed64 (signed64 (int32 pid)*(cols : Int))+signed64 (int32 lane))

/-- Actual widened mask and flat element offset, followed by pointer-byte
scaling. Inactive lanes never dereference their computed addresses. -/
def lane (p : Parameters) (xb sb ob pid column : Nat) : Option (Nat × Nat × Nat) :=
  if signed64 (int32 column)<(p.cols : Int) then
    some (pointer 4 xb (offset p.cols pid column),
          pointer 4 sb (signed64 (int32 pid)),
          pointer 2 ob (offset p.cols pid column))
  else none

theorem lane_exact (p : Parameters) (bounds : Bounds p xb sb ob scaleCount)
    (pid column : Nat) (hr : pid<p.rows) (hd : column<p.width) :
    lane p xb sb ob pid column=
      if column<p.cols then some (xb+4*(pid*p.cols+column),sb+4*pid,ob+2*(pid*p.cols+column))
      else none := by
  have pc : p.cols≤1048576 := le_trans p.covers bounds.width
  have rp : pid<2^31 := lt_trans hr bounds.grid
  have lp : column<1048576 := lt_of_lt_of_le hd bounds.width
  have rowEq : signed64 (int32 pid)=(pid : Int) := by
    rw [int32_exact _ (by omega) (by omega),signed64_exact _ (by omega) (by omega)]
  have colEq : signed64 (int32 column)=(column : Int) := by
    rw [int32_exact _ (by omega) (by omega),signed64_exact _ (by omega) (by omega)]
  by_cases active : column<p.cols
  · have indexed : pid*p.cols+column<p.rows*p.cols := linear_bound (⟨pid,hr⟩,⟨column,active⟩)
    have small : pid*p.cols+column<2^61 := by have := bounds.xSize;omega
    have product : signed64 ((pid : Int)*(p.cols : Int))=((pid*p.cols : Nat) : Int) := by
      rw [←Nat.cast_mul]
      exact signed64_exact _ (by omega) (by omega)
    have sum : signed64 (((pid*p.cols : Nat) : Int)+(column : Int))=((pid*p.cols+column : Nat) : Int) := by
      rw [←Nat.cast_add]
      exact signed64_exact _ (by omega) (by omega)
    have mask : (column : Int)<(p.cols : Int) := by exact_mod_cast active
    simp only [lane,colEq,mask,↓reduceIte,active,offset,rowEq,colEq,product,sum]
    rw [pointer_exact 4 xb (p.rows*p.cols) _ (by decide) bounds.xCapacity indexed,
      pointer_exact 4 sb scaleCount pid (by decide) bounds.scaleCapacity (lt_of_lt_of_le hr bounds.scaleRows),
      pointer_exact 2 ob (p.rows*p.cols) _ (by decide) bounds.outputCapacity indexed]
  · have mask : ¬(column : Int)<(p.cols : Int) := by exact_mod_cast active
    simp only [lane,colEq,mask,↓reduceIte,active]

abbrev RawEvent (p : Parameters) := (Fin p.rows × Fin p.width) × Fin 2

def selected (p : Parameters) (e : RawEvent p) : Option (Memory.Event p.rows p.cols) :=
  if active:e.1.2.val<p.cols then some ((e.1.1,⟨e.1.2.val,active⟩),e.2) else none

def events (p : Parameters) : List (RawEvent p) :=
  ((List.finRange p.rows).product (List.finRange p.width)).product (List.finRange 2)

theorem selected_complete (p : Parameters) (e : Memory.Event p.rows p.cols) :
    e∈(events p).filterMap (selected p) := by
  let raw : RawEvent p := ((e.1.1,⟨e.1.2.val,lt_of_lt_of_le e.1.2.isLt p.covers⟩),e.2)
  apply List.mem_filterMap.mpr
  refine ⟨raw,?_,?_⟩
  · exact List.mem_product.mpr ⟨List.mem_product.mpr ⟨List.mem_finRange _,List.mem_finRange _⟩,List.mem_finRange _⟩
  · simp [selected,raw,e.1.2.isLt]

/-- Byte serialization of each literal native FP16 store. Each load uses the
current shared backing. Output/source separation permits this serialization of
the vector load/store instructions without replacing the evolving memory. -/
def step (ops : Memory.Numerics) (p : Parameters) (xb sb ob : Nat)
    (e : RawEvent p) (m : Bytes) : Bytes :=
  match lane p xb sb ob e.1.1.val e.1.2.val with
  | none=>m
  | some (x,s,o)=>Scatter.write m (o+e.2.val)
      (Memory.compute ops (Memory.read32 m s) (Memory.read32 m x) e.2)

def execute (ops : Memory.Numerics) (p : Parameters) (xb sb ob : Nat)
    (order : List (RawEvent p)) (m : Bytes) : Bytes :=
  order.foldl (fun m e=>step ops p xb sb ob e m) m

theorem execute_selected (ops : Memory.Numerics) (p : Parameters)
    (bounds : Bounds p xb sb ob scaleCount) (order : List (RawEvent p)) (m : Bytes) :
    execute ops p xb sb ob order m=
      Memory.live ops xb sb ob (order.filterMap (selected p)) m := by
  induction order generalizing m with
  | nil=>rfl
  | cons e tail ih=>
    simp only [execute,List.foldl_cons]
    rw [show tail.foldl (fun m e=>step ops p xb sb ob e m) (step ops p xb sb ob e m)=
      Memory.live ops xb sb ob (tail.filterMap (selected p)) (step ops p xb sb ob e m) from ih _]
    rw [step,lane_exact p bounds _ _ e.1.1.isLt e.1.2.isLt]
    by_cases active : e.1.2.val<p.cols
    · simp [active,List.filterMap_cons,selected,Memory.live,Memory.key,
        Memory.value,Memory.scaleAddress,Memory.xAddress,linear]
    · simp [active,List.filterMap_cons,selected]

def launch (ops : Memory.Numerics) (p : Parameters) (xb sb ob : Nat) (m : Bytes) :=
  execute ops p xb sb ob (events p) m

theorem launch_readback (ops : Memory.Numerics) (p : Parameters)
    (bounds : Bounds p xb sb ob scaleCount) (sep : @Memory.Separated p.rows p.cols xb sb ob)
    (m : Bytes) (q : Coord p.rows p.cols) :
    Memory.read16 (launch ops p xb sb ob m) (ob+2*linear q)=Memory.value ops xb sb m q := by
  rw [launch,execute_selected ops p bounds]
  exact Memory.live_readback ops sep m _ q (fun b=>selected_complete p (q,b))

theorem launch_frame (ops : Memory.Numerics) (p : Parameters)
    (bounds : Bounds p xb sb ob scaleCount) (sep : @Memory.Separated p.rows p.cols xb sb ob)
    (m : Bytes) (address : Nat) (outside : ∀ e : Memory.Event p.rows p.cols,address≠Memory.key ob e) :
    launch ops p xb sb ob m address=m address := by
  rw [launch,execute_selected ops p bounds]
  exact Memory.live_frame ops sep m _ address (fun e _=>outside e)

#axiomsClean lane_exact
#axiomsClean selected_complete
#axiomsClean execute_selected
#axiomsClean launch_readback
#axiomsClean launch_frame
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Native
