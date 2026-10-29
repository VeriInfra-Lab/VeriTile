import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceForward

/-! An independent literal FP32 graph for one source program. Every cast,
initial addition, division, sqrt, and the exact local reduction operand tile
is retained. This graph consumes the original source memory, not final values.
-/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceForwardGraph
open VeriTile.Triton FP.Structural SourceModel
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def lit (m : Model α) (x : ℝ) : α := m.scalar.literal (some .fp32) .real x
def bin (m : Model α) (op : Binary) : α → α → α := m.scalar.binary (some .fp32) .real op
def cast (m : Model α) : α → α := m.scalar.cast (some .fp32) .real .real
def sum (m : Model α) (B : Nat) (v : Fin B → α) : α :=
  m.scalar.reduceSum (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false
    (fun i => v i.1) PUnit.unit

def xload (m : Model α) (N : Nat) (x : Nat → α) (j : Nat) : α :=
  cast m (if j<N then x j else lit m 0)

def mean (m : Model α) (N B : Nat) (x : Nat → α) : α :=
  bin m .div (sum m B (fun j => bin m .add (lit m 0) (xload m N x j.val))) (lit m N)

def centered (m : Model α) (N B : Nat) (x : Nat → α) (j : Nat) : α :=
  if j<N then bin m .sub (xload m N x j) (mean m N B x) else lit m 0

def variance (m : Model α) (N B : Nat) (x : Nat → α) : α :=
  bin m .div (sum m B (fun j => bin m .add (lit m 0)
    (bin m .mul (centered m N B x j.val) (centered m N B x j.val)))) (lit m N)

def inverse (m : Model α) (N B : Nat) (eps : ℝ) (x : Nat → α) : α :=
  bin m .div (lit m 1) (m.scalar.unary (some .fp32) .sqrt
    (bin m .add (variance m N B x) (lit m eps)))

def output (m : Model α) (N B : Nat) (eps : ℝ) (x w bias : Nat → α) (j : Nat) : α :=
  cast m (bin m .add
    (bin m .mul (bin m .mul (bin m .sub (xload m N x j) (mean m N B x)) (inverse m N B eps x))
      (m.scalar.fp32Load (w j))) (m.scalar.fp32Load (bias j)))

def row [Inhabited α] (s : State α) (stride : Nat) : Nat → α :=
  fun j => (s.mem "X" (s.pids 0*stride+j)).read .real

def vector [Inhabited α] (s : State α) (r : RegionName) : Nat → α :=
  fun j => (s.mem r j).read .real

def final [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ) (s : State α) : State α :=
  let x := row s stride
  let saved := (s.write "Mean" (s.pids 0) (.mk .real (mean m N B x))).write
    "Rstd" (s.pids 0) (.mk .real (inverse m N B eps x))
  (TileShape.allIndices [B]).foldl (fun state i =>
    if i.1.val<N then state.write "Y" (s.pids 0*stride+i.1.val)
      (.mk .real (output m N B eps x (vector s "W") (vector s "Bias") i.1.val)) else state) saved

theorem masked_memory_congr {α ι : Type} (r : RegionName) (offset : ι → Nat)
    (val : ι → Cell α) (active : ι → Bool) (l : List ι) (s t : State α) (h : s.mem=t.mem) :
    (l.foldl (fun state i => if active i then state.write r (offset i) (val i) else state) s).mem=
    (l.foldl (fun state i => if active i then state.write r (offset i) (val i) else state) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    split
    · apply ih
      simp only [State.write,h]
    · exact ih s t h

theorem fold_memory_congr {α ι : Type} (f g : State α → ι → State α)
    (l : List ι) (s t : State α) (initial : s.mem=t.mem)
    (next : ∀ a b i, a.mem=b.mem → (f a i).mem=(g b i).mem) :
    (l.foldl f s).mem=(l.foldl g t).mem := by
  induction l generalizing s t with
  | nil => exact initial
  | cons i rest ih => exact ih (f s i) (g t i) (next s t i initial)

/-- Exact complete source memory, not merely one output observation. -/
theorem source_memory [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ)
    (positive : 0<N) (covers : N≤B) (s t : State α)
    (h : FP.Structural.exec (engine m)
      (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) s=some t) :
    t.mem=(final m stride N B eps s).mem := by
  simp [FP.Structural.exec,TypedSource.forward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    range_one _ _ N B positive covers,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat] at h
  simp only [show (0.0 : ℝ)=0 by norm_num,show (1.0 : ℝ)=1 by norm_num] at h
  subst t
  simp only [final,row,vector,output,inverse,variance,centered,mean,sum,xload,lit,bin,cast]
  apply fold_memory_congr
  · dsimp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,State.write]
    rfl
  · intro a b i hab
    dsimp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
    by_cases hi : i.1.val<N <;> simp [hi,State.write,hab] <;> rfl

#axiomsClean source_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceForwardGraph
