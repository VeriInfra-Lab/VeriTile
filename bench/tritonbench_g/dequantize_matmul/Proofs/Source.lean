import bench.tritonbench_g.dequantize_matmul.Proofs.Plan
import bench.tritonbench_g.dequantize_matmul.Proofs.Rect

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 3000000

inductive Cell (α : Type) where
  | int8 (word : BitVec 8)
  | fp32 (word : α)
  | opaque (token : Nat)

def Cell.float (zero : α) : Cell α → α
  | .fp32 x => x
  | _ => zero
def Cell.signed : Cell α → BitVec 8
  | .int8 q => q
  | _ => 0

structure Shape where
  M : Nat
  K : Nat
  N : Nat
  deriving DecidableEq
structure Location where
  storage : Nat
  offset : Nat
  deriving DecidableEq
def Location.window (p : Location) (rows cols : Nat) : Window :=
  ⟨p.storage,p.offset,rows,cols⟩
def Location.addr (p : Location) (index : Nat) : Address := (p.storage,p.offset+index)

structure Buffers where
  a : Location
  b : Location
  scale : Location
  fpb : Location
  out : Location

structure Tile where
  rows : Nat
  columns : Nat
  rowsPositive : 0<rows
  columnsPositive : 0<columns

/-- Fixed by the selected library implementation/ABI, before seeing outputs.
Completeness is a permutation of K indices, not a numerical mm contract. -/
structure LibraryPlan (s : Shape) where
  output : Nat → Nat → Plan s.K
  complete : ∀ r<s.M,∀ c<s.N,(output r c).Complete

def dequantRect (s : Shape) (b : Buffers) (tile : Tile) : Rect :=
  ⟨b.fpb.window s.K s.N,tile.rows,tile.columns⟩
def outputRect (s : Shape) (b : Buffers) : Rect :=
  ⟨b.out.window s.M s.N,1,1⟩

def dequantValue (a : Arithmetic α) (s : Shape) (b : Buffers)
    (m : Memory (Cell α)) (k n : Nat) : Cell α :=
  .fp32 (a.store (a.mul (a.signed8 (m (b.b.addr (k*s.N+n))).signed)
    (a.load ((m (b.scale.addr n)).float a.zero))))
def dequantRun (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (m : Memory (Cell α)) : Memory (Cell α) :=
  (dequantRect s b tile).run (dequantValue a s b) m

def mmValue (a : Arithmetic α) (s : Shape) (b : Buffers) (plan : LibraryPlan s)
    (m : Memory (Cell α)) (r n : Nat) : Cell α :=
  .fp32 (a.store ((plan.output r n).eval a
    (fun k => (m (b.a.addr (r*s.K+k.val))).float a.zero)
    (fun k => (m (b.fpb.addr (k.val*s.N+n))).float a.zero)))
def mmRun (a : Arithmetic α) (s : Shape) (b : Buffers) (plan : LibraryPlan s)
    (m : Memory (Cell α)) : Memory (Cell α) :=
  (outputRect s b).run (mmValue a s b plan) m
def run (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (plan : LibraryPlan s) (m : Memory (Cell α)) : Memory (Cell α) :=
  mmRun a s b plan (dequantRun a s b tile m)

structure Separate (b : Buffers) : Prop where
  scratchA : b.a.storage≠b.fpb.storage
  scratchB : b.b.storage≠b.fpb.storage
  scratchScale : b.scale.storage≠b.fpb.storage
  outputA : b.a.storage≠b.out.storage
  outputScratch : b.fpb.storage≠b.out.storage

theorem dequant_stable (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (sep : Separate b) : (dequantRect s b tile).ReadsOutside (dequantValue a s b) := by
  intro entry current same r c
  unfold dequantValue
  rw [same (b.b.addr (r*s.N+c)) sep.scratchB,
    same (b.scale.addr c) sep.scratchScale]

theorem mm_stable (a : Arithmetic α) (s : Shape) (b : Buffers) (plan : LibraryPlan s)
    (sep : Separate b) : (outputRect s b).ReadsOutside (mmValue a s b plan) := by
  intro entry current same r c
  have ha : (fun k : Fin s.K => (current (b.a.addr (r*s.K+k.val))).float a.zero)=
      (fun k : Fin s.K => (entry (b.a.addr (r*s.K+k.val))).float a.zero) := by
    funext k; rw [same _ sep.outputA]
  have hd : (fun k : Fin s.K => (current (b.fpb.addr (k.val*s.N+c))).float a.zero)=
      (fun k : Fin s.K => (entry (b.fpb.addr (k.val*s.N+c))).float a.zero) := by
    funext k; rw [same _ sep.outputScratch]
  simp only [mmValue,ha,hd]

theorem dequant_output (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (sep : Separate b) (initial : Memory (Cell α)) (k n : Nat)
    (hk : k<s.K) (hn : n<s.N) :
    dequantRun a s b tile initial (b.fpb.addr (k*s.N+n))=dequantValue a s b initial k n := by
  simpa only [dequantRun,dequantRect,Location.window,Window.addr,Location.addr,Nat.add_assoc] using
    (dequantRect s b tile).output (dequantValue a s b) (dequant_stable a s b tile sep)
      initial tile.rowsPositive tile.columnsPositive k n hk hn

theorem dequant_keeps_a (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (sep : Separate b) (initial : Memory (Cell α)) (index : Nat) :
    dequantRun a s b tile initial (b.a.addr index)=initial (b.a.addr index) := by
  apply (dequantRect s b tile).frame _ (dequant_stable a s b tile sep)
  intro h
  exact sep.scratchA h.1

/-- The independent word computation uses the initial operands and the fixed
plan. The program theorem derives it from actual fp_b stores and later loads. -/
def resultValue (a : Arithmetic α) (s : Shape) (b : Buffers) (plan : LibraryPlan s)
    (initial : Memory (Cell α)) (r n : Nat) : α :=
  a.store ((plan.output r n).eval a
    (fun k => (initial (b.a.addr (r*s.K+k.val))).float a.zero)
    (fun k => (dequantValue a s b initial k.val n).float a.zero))

theorem producer_to_consumer (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (plan : LibraryPlan s) (sep : Separate b) (initial : Memory (Cell α))
    (r n : Nat) (hn : n<s.N) :
    mmValue a s b plan (dequantRun a s b tile initial) r n=
      .fp32 (resultValue a s b plan initial r n) := by
  have ha : (fun k : Fin s.K => ((dequantRun a s b tile initial) (b.a.addr (r*s.K+k.val))).float a.zero)=
      (fun k : Fin s.K => (initial (b.a.addr (r*s.K+k.val))).float a.zero) := by
    funext k; rw [dequant_keeps_a a s b tile sep]
  have hd : (fun k : Fin s.K => ((dequantRun a s b tile initial) (b.fpb.addr (k.val*s.N+n))).float a.zero)=
      (fun k : Fin s.K => (dequantValue a s b initial k.val n).float a.zero) := by
    funext k; rw [dequant_output a s b tile sep initial k.val n k.isLt hn]
  simp only [mmValue,resultValue,ha,hd]

theorem output (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (plan : LibraryPlan s) (sep : Separate b) (initial : Memory (Cell α))
    (r n : Nat) (hr : r<s.M) (hn : n<s.N) :
    run a s b tile plan initial (b.out.addr (r*s.N+n))=
      .fp32 (resultValue a s b plan initial r n) := by
  have h := (outputRect s b).output (mmValue a s b plan) (mm_stable a s b plan sep)
    (dequantRun a s b tile initial) (by change 0 < (1 : Nat); decide)
      (by change 0 < (1 : Nat); decide) r n hr hn
  have h' : mmRun a s b plan (dequantRun a s b tile initial) (b.out.addr (r*s.N+n))=
      mmValue a s b plan (dequantRun a s b tile initial) r n := by
    simpa only [mmRun,outputRect,Location.window,Window.addr,Location.addr,Nat.add_assoc] using h
  exact h'.trans (producer_to_consumer a s b tile plan sep initial r n hn)

theorem frame (a : Arithmetic α) (s : Shape) (b : Buffers) (tile : Tile)
    (plan : LibraryPlan s) (sep : Separate b) (initial : Memory (Cell α))
    (point : Address) (outsideOutput : ¬(b.out.window s.M s.N).inside point)
    (outsideScratch : ¬(b.fpb.window s.K s.N).inside point) :
    run a s b tile plan initial point=initial point := by
  rw [run,mmRun,(outputRect s b).frame _ (mm_stable a s b plan sep) _ _ outsideOutput,
    dequantRun,(dequantRect s b tile).frame _ (dequant_stable a s b tile sep) _ _ outsideScratch]

#axiomsClean dequant_output
#axiomsClean producer_to_consumer
#axiomsClean output
#axiomsClean frame
end
end VeriTile.Bench.Repairs.DequantizeMatmul
