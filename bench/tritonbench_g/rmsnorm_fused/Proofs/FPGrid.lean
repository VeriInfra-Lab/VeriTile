import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.rmsnorm_fused.Proofs.FPRow
import bench.tritonbench_g.rmsnorm_fused.Proofs.Geometry

namespace VeriTile.Bench.Repairs.RmsnormFused.Row
open VeriTile Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def InputsEq (X W : RegionName) (s t : State α) : Prop :=
  (∀ o,s.mem X o=t.mem X o) ∧ (∀ o,s.mem W o=t.mem W o)

theorem value_congr (A : Algebra α) (s t : State α) (X W : RegionName)
    (row N B k : Nat) (eps : ℝ) (h : InputsEq X W s t) (hk : k<N) :
    value A s X W row N B eps k=value A t X W row N B eps k := by
  have hx : ∀ j,xload A s X row N j=xload A t X row N j := by
    intro j
    simp [xload,read,h.1]
  have hs : square A s X row N B=square A t X row N B := by
    funext i
    simp [square,hx]
  simp [value,rstd,variance,hs,hx,wload,hk,read,h.2]

theorem block_frame (A : Algebra α) (s : State α) (X Y W : RegionName)
    (row N B : Nat) (eps : ℝ) (r : RegionName) (o : Nat)
    (outside : r≠Y ∨ ∀ k,k<N → o≠row*N+k) :
    (block A s X Y W row N B eps).mem r o=s.mem r o := by
  apply State.masked_scatter_frame
  rcases outside with h|h
  · exact Or.inl h
  · exact Or.inr (fun i _ hi => h i.1.val hi)

theorem block_inputs (A : Algebra α) (s : State α) (X Y W : RegionName)
    (row N B : Nat) (eps : ℝ) (hX : X≠Y) (hW : W≠Y) :
    InputsEq X W (block A s X Y W row N B eps) s := by
  exact ⟨fun o => block_frame A s X Y W row N B eps X o (.inl hX),
    fun o => block_frame A s X Y W row N B eps W o (.inl hW)⟩

theorem block_value (A : Algebra α) (s : State α) (X Y W : RegionName)
    (row N B k : Nat) (eps : ℝ) (hk : k<N) (hNB : N≤B) :
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

def grid (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : ℝ) :
    List Nat → State α → State α
  | [],s => s
  | row::rest,s => grid A X Y W N B eps rest (block A s X Y W row N B eps)

theorem writes_congr {ι : Type} (xs : List ι) (Y : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v w : ι → Cell α)
    (s t : State α) (hm : s.mem=t.mem) (hv : ∀ i,active i → v i=w i) :
    (xs.foldl (fun st i => if active i then st.write Y (off i) (v i) else st) s).mem =
      (xs.foldl (fun st i => if active i then st.write Y (off i) (w i) else st) t).mem := by
  induction xs generalizing s t with
  | nil => exact hm
  | cons i xs ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases h : active i
    · simp [h,State.write,hm,hv i h]
    · simp [h,hm]

theorem block_congr (A : Algebra α) (s t : State α) (X Y W : RegionName)
    (row N B : Nat) (eps : ℝ) (h : s.mem=t.mem) :
    (block A s X Y W row N B eps).mem=(block A t X Y W row N B eps).mem := by
  apply writes_congr _ _ _ _ _ _ s t h
  intro i hi
  rw [value_congr A s t X W row N B i.1.val eps
    ⟨fun o => congrFun (congrFun h X) o,fun o => congrFun (congrFun h W) o⟩ hi]

theorem grid_congr (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : ℝ)
    (rows : List Nat) (s t : State α) (h : s.mem=t.mem) :
    (grid A X Y W N B eps rows s).mem=(grid A X Y W N B eps rows t).mem := by
  induction rows generalizing s t with
  | nil => exact h
  | cons r rest ih => exact ih _ _ (block_congr A s t X Y W r N B eps h)

def setup (rows row : Nat) (s : State α) : State α :=
  {s with regs := fun _ _ _ => none
          pids := fun a => if a=0 then row else 0
          numPids := fun a => if a=0 then rows else 1}

theorem grid_frame (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : ℝ)
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

theorem grid_inputs (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : ℝ)
    (rows : List Nat) (s : State α) (hX : X≠Y) (hW : W≠Y) :
    InputsEq X W (grid A X Y W N B eps rows s) s := by
  exact ⟨fun o => grid_frame A X Y W N B eps rows s X o (.inl hX),
    fun o => grid_frame A X Y W N B eps rows s W o (.inl hW)⟩

theorem row_offset_ne (N r q k j : Nat) (hk : k<N) (hj : j<N) (hr : r≠q) :
    r*N+k≠q*N+j := by
  intro h
  have hN : 0<N := by omega
  have same := congrArg (fun a => a/N) h
  have left : (r*N+k)/N=r := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hN,Nat.div_eq_of_lt hk]
    simp
  have right : (q*N+j)/N=q := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hN,Nat.div_eq_of_lt hj]
    simp
  apply hr
  simpa only [left,right] using same

theorem grid_value (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : ℝ)
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
    (eps : ℝ) (rows : List Nat) (all : rows.Perm s.grid) (m : State α)
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
end VeriTile.Bench.Repairs.RmsnormFused.Row
