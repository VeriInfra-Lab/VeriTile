import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.RmsnormFused.Runtime
open VeriTile Triton FP.Structural
noncomputable section
open Classical
variable {α : Type} [Inhabited α]

/-- Retain the store's cell tag in the existing scatter readback. -/
theorem block_cell (A : Algebra α) (s : State α) (X Y W : RegionName)
    (row N B k : Nat) (eps : α) (hk : k<N) (hNB : N≤B) :
    (block A s X Y W row N B eps).mem Y (row*N+k)=
      Cell.mk .real (value A s X W row N B eps k) := by
  let i : TileIndex [B] := (⟨k,lt_of_lt_of_le hk hNB⟩,PUnit.unit)
  have injective : Function.Injective (fun j : TileIndex [B] => row*N+j.1.val) := by
    intro a b h
    exact Prod.ext (Fin.ext (Nat.add_left_cancel h)) (Subsingleton.elim _ _)
  exact State.masked_scatter_readback Y (fun j : TileIndex [B] => row*N+j.1.val)
    (fun j => Cell.mk .real (value A s X W row N B eps j.1.val))
    (fun j => j.1.val<N) s i hk injective

/-- Lift existing grid value/frame composition to exact typed cells. -/
theorem grid_cell (A : Algebra α) (X Y W : RegionName) (N B : Nat) (eps : α)
    (rows : List Nat) (s : State α) (r k : Nat) (hr : r∈rows) (hk : k<N)
    (hNB : N≤B) (hX : X≠Y) (hW : W≠Y) (unique : rows.Nodup) :
    (grid A X Y W N B eps rows s).mem Y (r*N+k)=
      Cell.mk .real (value A s X W r N B eps k) := by
  induction rows generalizing s with
  | nil => simp at hr
  | cons q rest ih =>
    have un := List.nodup_cons.mp unique
    by_cases inside : r∈rest
    · rw [grid,ih _ inside un.2]
      rw [value_congr A _ s X W r N B k eps (block_inputs A s X Y W q N B eps hX hW) hk]
    · have same : r=q := (List.mem_cons.mp hr).resolve_right inside
      subst q
      rw [grid,grid_frame A X Y W N B eps rest (block A s X Y W r N B eps) Y (r*N+k)
        (Or.inr (by
          intro q hq j hj
          exact row_offset_ne N r q k j hk hj (by intro he; subst q; exact un.1 hq)))]
      exact block_cell A s X Y W r N B k eps hk hNB

/-- Only active input values influence an active output, including its row
reduction. Arbitrary exterior cells and inactive undefined weights are ignored. -/
theorem value_congr_active (A : Algebra α) (s t : State α) (X W : RegionName)
    (row N B k : Nat) (eps : α) (hk : k<N)
    (hx : ∀ j,j<N → read s X (row*N+j)=read t X (row*N+j))
    (hw : read s W k=read t W k) :
    value A s X W row N B eps k=value A t X W row N B eps k := by
  have xeq (j : Nat) : xload A s X row N j=xload A t X row N j := by
    by_cases active : j<N
    · simp [xload,active,hx j active]
    · simp [xload,active]
  have squares : square A s X row N B=square A t X row N B := by
    funext j
    simp only [square,xeq]
  simp only [value,rstd,variance,squares,xeq,wload,hk,if_true,hw]
end
end VeriTile.Bench.Repairs.RmsnormFused.Runtime

namespace VeriTile.Bench.Repairs.RmsnormFused.KernelInterface
open VeriTile Triton FP.Structural
noncomputable section

theorem complete_typed {α : Type} [Inhabited α] (A : Algebra α) (shape : Shape)
    (valid : shape.Valid) (epsilon : α) (schedule : List Nat)
    (all : schedule.Perm shape.grid) (initial : State α) :
    ∃ final, Runtime.run A "X" "Y" "W" shape.rows shape.width shape.block
      epsilon schedule initial=some final ∧
      (∀ row col,row<shape.rows → col<shape.width →
        final.mem "Y" (row*shape.width+col)=Cell.mk .real
          (Runtime.value A initial "X" "W" row shape.width shape.block epsilon col)) ∧
      (∀ (region : RegionName) offset,region≠"Y" ∨ shape.rows*shape.width≤offset →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,ran,_,frame⟩ := complete A shape valid epsilon schedule all initial
  obtain ⟨other,ran',memory⟩ := Runtime.source_complete A "X" "Y" "W" shape.rows shape.width
    shape.block epsilon valid.1 valid.2.1 schedule initial
  have same : other=final := Option.some.inj (ran'.symm.trans ran)
  subst other
  refine ⟨final,ran,?_,frame⟩
  intro row col hr hc
  rw [memory]
  exact Runtime.grid_cell A "X" "Y" "W" shape.width shape.block epsilon schedule initial row col
    (all.mem_iff.mpr (List.mem_range.mpr hr)) hc valid.2.1 (by decide) (by decide)
    (all.nodup_iff.mpr List.nodup_range)
end
end VeriTile.Bench.Repairs.RmsnormFused.KernelInterface
