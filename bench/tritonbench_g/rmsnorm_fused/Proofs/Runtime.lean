import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.rmsnorm_fused.Proofs.FPGrid

namespace VeriTile.Bench.Repairs.RmsnormFused.Runtime
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def lit (A : Algebra α) (x : ℝ) := A.literal (some .fp32) .real x
def add (A : Algebra α) := A.binary (some .fp32) .real .add
def mul (A : Algebra α) := A.binary (some .fp32) .real .mul
def div (A : Algebra α) := A.binary (some .fp32) .real .div
def cast (A : Algebra α) := A.cast (some .fp32) .real .real
def read (s : State α) (r : RegionName) (o : Nat) := (s.mem r o).read .real
def xload (A : Algebra α) (s : State α) (X : RegionName) (row N k : Nat) :=
  cast A (if k<N then read s X (row*N+k) else lit A 0)
def wload (A : Algebra α) (s : State α) (W : RegionName) (N k : Nat) :=
  A.fp32Load (if k<N then read s W k else s.undef .real W k)
def square (A : Algebra α) (s : State α) (X : RegionName) (row N B : Nat) : Values α .real [B] :=
  fun i => add A (lit A 0) (mul A
    (if i.1.val<N then xload A s X row N i.1.val else lit A 0)
    (if i.1.val<N then xload A s X row N i.1.val else lit A 0))
def variance (A : Algebra α) (s : State α) (X : RegionName) (row N B : Nat) :=
  div A (A.reduceSum (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false
    (square A s X row N B) PUnit.unit) (lit A (N : ℝ))
def rstd (A : Algebra α) (s : State α) (X : RegionName) (row N B : Nat) (eps : α) :=
  div A (lit A 1) (A.unary (some .fp32) .sqrt (add A (variance A s X row N B) eps))
def value (A : Algebra α) (s : State α) (X W : RegionName) (row N B : Nat) (eps : α) (k : Nat) :=
  mul A (mul A (xload A s X row N k) (rstd A s X row N B eps)) (wload A s W N k)
def block (A : Algebra α) (s : State α) (X Y W : RegionName) (row N B : Nat) (eps : α) :=
  (TileShape.allIndices [B]).foldl
    (fun st i => if i.1.val<N then st.write Y (row*N+i.1.val) (.mk .real (value A s X W row N B eps i.1.val)) else st) s

abbrev write_congr := @Row.write_congr

/-- The complete unchanged two-loop source, in the wrapper's actual FP32
single-tile regime, executes and emits this typed scalar graph. -/
def bind (s : State α) (eps : α) : State α :=
  s.setReg "eps" (dtype := .real) (shape := []) (fun _ => eps)

theorem complete (A : Algebra α) (s : State α) (X Y W : RegionName)
    (N B : Nat) (eps : α) (hN : 0<N) (hNB : N≤B) :
    ∃ t, exec A (source X Y W N B) (bind s eps)=some t ∧
      t.mem=(block A s X Y W (s.pids 0) N B eps).mem := by
  have hB : B≠0 := by omega
  have hz : (0.0 : ℝ)=0 := by norm_num
  simp [source,template,bind,
    FP.Structural.exec,FP.Structural.run,FP.Structural.step,FP.Structural.range,evalExpr,evalComputeOp,evalOp_unfold,store,
    numeric,numericLt,numericLe,natLt,natLe,bop,ofFloat,toFloat,Region.cast,
    ComputeKernel.surfaceBody,ComputeDType.eraseDType,State.setReg,Option.bind,hN,hB,hz,Nat.not_lt.mpr hNB,
    block,value,rstd,variance,square,xload,wload,lit,add,mul,Runtime.div,cast,read]
  apply write_congr
  rfl

#axiomsClean complete
end
end VeriTile.Bench.Repairs.RmsnormFused.Runtime

namespace VeriTile.Bench.Repairs.RmsnormFused.Runtime
open VeriTile Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def InputsEq (X W : RegionName) (s t : State α) : Prop :=
  (∀ o,s.mem X o=t.mem X o) ∧ (∀ o,s.mem W o=t.mem W o)

theorem value_congr (A : Algebra α) (s t : State α) (X W : RegionName)
    (row N B k : Nat) (eps : α) (h : InputsEq X W s t) (hk : k<N) :
    value A s X W row N B eps k=value A t X W row N B eps k := by
  have hx : ∀ j,xload A s X row N j=xload A t X row N j := by
    intro j
    simp [xload,read,h.1]
  have hs : square A s X row N B=square A t X row N B := by
    funext i
    simp [square,hx]
  simp [value,rstd,variance,hs,hx,wload,hk,read,h.2]

theorem block_frame (A : Algebra α) (s : State α) (X Y W : RegionName)
    (row N B : Nat) (eps : α) (r : RegionName) (o : Nat)
    (outside : r≠Y ∨ ∀ k,k<N → o≠row*N+k) :
    (block A s X Y W row N B eps).mem r o=s.mem r o := by
  apply State.masked_scatter_frame
  rcases outside with h|h
  · exact Or.inl h
  · exact Or.inr (fun i _ hi => h i.1.val hi)

theorem block_inputs (A : Algebra α) (s : State α) (X Y W : RegionName)
    (row N B : Nat) (eps : α) (hX : X≠Y) (hW : W≠Y) :
    InputsEq X W (block A s X Y W row N B eps) s := by
  exact ⟨fun o => block_frame A s X Y W row N B eps X o (.inl hX),
    fun o => block_frame A s X Y W row N B eps W o (.inl hW)⟩

theorem block_value (A : Algebra α) (s : State α) (X Y W : RegionName)
    (row N B k : Nat) (eps : α) (hk : k<N) (hNB : N≤B) :
    read (block A s X Y W row N B eps) Y (row*N+k)=value A s X W row N B eps k := by
  let i : TileIndex [B] := (⟨k,lt_of_lt_of_le hk hNB⟩,PUnit.unit)
  have injective : Function.Injective (fun j : TileIndex [B] => row*N+j.1.val) := by
    intro a b h
    have hf : a.1=b.1 := Fin.ext (Nat.add_left_cancel h)
    exact Prod.ext hf (Subsingleton.elim _ _)
  have hv := State.masked_scatter_readback Y (fun j : TileIndex [B] => row*N+j.1.val)
    (fun j => Cell.mk .real (value A s X W row N B eps j.1.val))
    (fun j => j.1.val<N) s i hk injective
  exact (congrArg (fun cell => cell.read .real) hv).trans (Cell.read_same _ _)

def grid (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : α) :
    List Nat → State α → State α
  | [],s => s
  | row::rest,s => grid A X Y W N B eps rest (block A s X Y W row N B eps)

abbrev writes_congr := @Row.writes_congr

theorem block_congr (A : Algebra α) (s t : State α) (X Y W : RegionName)
    (row N B : Nat) (eps : α) (h : s.mem=t.mem) :
    (block A s X Y W row N B eps).mem=(block A t X Y W row N B eps).mem := by
  apply writes_congr _ _ _ _ _ _ s t h
  intro i hi
  rw [value_congr A s t X W row N B i.1.val eps
    ⟨fun o => congrFun (congrFun h X) o,fun o => congrFun (congrFun h W) o⟩ hi]

theorem grid_congr (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : α)
    (rows : List Nat) (s t : State α) (h : s.mem=t.mem) :
    (grid A X Y W N B eps rows s).mem=(grid A X Y W N B eps rows t).mem := by
  induction rows generalizing s t with
  | nil => exact h
  | cons r rest ih => exact ih _ _ (block_congr A s t X Y W r N B eps h)

def setup (rows row : Nat) (s : State α) : State α :=
  {s with regs := fun _ _ _ => none
          pids := fun a => if a=0 then row else 0
          numPids := fun a => if a=0 then rows else 1}

/-- Every scheduled row executes the entire typed source against the actual
memory left by preceding rows. No per-row output is supplied by the runtime. -/
def run (A : Algebra α) (X Y W : RegionName) (M N B : Nat) (eps : α) :
    List Nat → State α → Option (State α)
  | [],s => some s
  | r::rest,s => (FP.Structural.exec A (source X Y W N B) (bind (setup M r s) eps)).bind
      (run A X Y W M N B eps rest)

theorem source_complete (A : Algebra α) (X Y W : RegionName) (M N B : Nat) (eps : α)
    (hN : 0<N) (hNB : N≤B) (rows : List Nat) (s : State α) :
    ∃ t,run A X Y W M N B eps rows s=some t ∧
      t.mem=(grid A X Y W N B eps rows s).mem := by
  induction rows generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons r rest ih =>
    obtain ⟨t,ht,hm⟩ := complete A (setup M r s) X Y W N B eps hN hNB
    obtain ⟨u,hu,hv⟩ := ih t
    refine ⟨u,by simpa [run,ht] using hu,?_⟩
    have hb : t.mem=(block A s X Y W r N B eps).mem :=
      hm.trans (block_congr A (setup M r s) s X Y W r N B eps rfl)
    exact hv.trans (grid_congr A X Y W N B eps rest t _ hb)

theorem grid_frame (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : α)
    (rows : List Nat) (s : State α) (r : RegionName) (o : Nat)
    (outside : r≠Y ∨ ∀ row∈rows,∀ k,k<N → o≠row*N+k) :
    (grid A X Y W N B eps rows s).mem r o=s.mem r o := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rest ih =>
    rw [grid,ih]
    · apply block_frame
      exact outside.imp_right (fun h k hk => h row (by simp) k hk)
    · exact outside.imp_right (fun h q hq => h q (by simp [hq]))

theorem grid_inputs (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : α)
    (rows : List Nat) (s : State α) (hX : X≠Y) (hW : W≠Y) :
    InputsEq X W (grid A X Y W N B eps rows s) s := by
  exact ⟨fun o => grid_frame A X Y W N B eps rows s X o (.inl hX),
    fun o => grid_frame A X Y W N B eps rows s W o (.inl hW)⟩

abbrev row_offset_ne := @Row.row_offset_ne

theorem grid_value (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : α)
    (rows : List Nat) (s : State α) (r k : Nat) (hr : r∈rows) (hk : k<N)
    (hNB : N≤B) (hX : X≠Y) (hW : W≠Y) (unique : rows.Nodup) :
    read (grid A X Y W N B eps rows s) Y (r*N+k)=value A s X W r N B eps k := by
  induction rows generalizing s with
  | nil => simp at hr
  | cons q rest ih =>
    have un := List.nodup_cons.mp unique
    by_cases inside : r∈rest
    · rw [grid,ih _ inside un.2]
      exact value_congr A _ s X W r N B k eps (block_inputs A s X Y W q N B eps hX hW) hk
    · have same : r=q := (List.mem_cons.mp hr).resolve_right inside
      subst q
      have hf := grid_frame A X Y W N B eps rest
        (block A s X Y W r N B eps) Y (r*N+k) (Or.inr (by
          intro q hq j hj
          exact row_offset_ne N r q k j hk hj (by intro he; subst q; exact un.1 hq)))
      change ((grid A X Y W N B eps rest (block A s X Y W r N B eps)).mem Y (r*N+k)).read .real = _
      rw [hf]
      exact block_value A s X Y W r N B k eps hk hNB

/-- A completed grid is any permutation of every launched row exactly once. -/
theorem whole_value (A : Algebra α) (s : Shape) (v : s.Valid) (X Y W : RegionName)
    (eps : α) (rows : List Nat) (all : rows.Perm s.grid) (m : State α)
    (r k : Nat) (hr : r<s.rows) (hk : k<s.width) (hX : X≠Y) (hW : W≠Y) :
    read (grid A X Y W s.width s.block eps rows m) Y (s.offset r k)=
      value A m X W r s.width s.block eps k := by
  exact grid_value A X Y W s.width s.block eps rows m r k
    (all.mem_iff.mpr ((grid_complete s r).mpr hr)) hk v.2.1 hX hW
    (all.nodup_iff.mpr (grid_unique s))

#axiomsClean block_value
#axiomsClean grid_frame
#axiomsClean whole_value
end
end VeriTile.Bench.Repairs.RmsnormFused.Runtime
