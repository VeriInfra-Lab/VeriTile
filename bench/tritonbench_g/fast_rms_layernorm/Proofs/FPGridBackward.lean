import bench.tritonbench_g.fast_rms_layernorm.Proofs.FPGridForward
import bench.tritonbench_g.fast_rms_layernorm.Proofs.FPBackwardMemory

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPGridBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open FPGridForward (Address Heap entry)
noncomputable section
variable {α : Type} [Inhabited α]

def address (h : Call) (row : Fin h.rows) (j : Fin h.cols) : Address := ("dx",row.val*h.cols+j.val)
def writes (h : Call) (row : Fin h.rows) (a : Address) : Prop := ∃ j : Fin h.cols,a=address h row j
def reads (h : Call) (row : Fin h.rows) (a : Address) : Prop := a.1="x" ∨ a.1="w" ∨ a.1="r" ∨ writes h row a
def expected (M : Algebra α) (h : Call) (heap : Heap α) (row : Fin h.rows) (j : Fin h.cols) : Cell α :=
  FPBackwardReference.output M h
    (FPMemory.raw h (entry heap row.val h.rows) "x" h.cols)
    (FPMemory.raw h (entry heap row.val h.rows) "w" 0)
    (FPMemory.raw h (entry heap row.val h.rows) "dx" h.cols)
    ((heap ("r",row.val)).read .real) ⟨j.val,j.isLt.trans_le h.covered⟩
def execute (M : Algebra α) (h : Call) (row : Fin h.rows) (heap : Heap α) : Option (Heap α) :=
  (FP.Structural.exec (engine M) (Kernels.backward h) (entry heap row.val h.rows)).map
    (fun v a => v.mem a.1 a.2)

theorem separate (h : Call) (a b : Fin h.rows) (ne : a≠b) (addr : Address)
    (ha : writes h a addr) : ¬writes h b addr ∧ ¬reads h b addr := by
  have disjoint : ¬writes h b addr := by
    rintro ⟨j,hb⟩
    obtain ⟨i,ha⟩:=ha
    have eq := congrArg Prod.snd (ha.symm.trans hb)
    exact ne (Fin.ext (Geometry.row_injective h a.val b.val i j eq).1)
  refine ⟨disjoint,?_⟩
  intro hr
  obtain ⟨j,rfl⟩:=ha
  apply disjoint
  simpa [reads,address] using hr

theorem step_complete (M : Algebra α) (h : Call) (row : Fin h.rows) (heap : Heap α) :
    ∃ final,execute M h row heap=some final ∧
      (∀ j,final (address h row j)=expected M h heap row j) ∧
      (∀ addr,¬writes h row addr → final addr=heap addr) := by
  obtain ⟨v,he,hy,hf⟩ := FPBackwardMemory.complete M h (entry heap row.val h.rows)
  refine ⟨fun a => v.mem a.1 a.2,by simp [execute,he],hy,?_⟩
  rintro ⟨q,o⟩ outside
  apply hf
  rintro ⟨same,j,eq⟩
  exact outside ⟨j,Prod.ext same eq⟩

theorem expected_stable (M : Algebra α) (h : Call) (row : Fin h.rows) (a b : Heap α)
    (stable : ∀ addr,reads h row addr → a addr=b addr) (j : Fin h.cols) :
    expected M h a row j=expected M h b row j := by
  have agree (role : RegionName) (stride : Nat)
      (whereRead : ∀ k : Fin h.block,k.val<h.cols → reads h row (role,row.val*stride+k.val)) :
      FPReference.input M h (FPMemory.raw h (entry a row.val h.rows) role stride)=
        FPReference.input M h (FPMemory.raw h (entry b row.val h.rows) role stride) := by
    funext k
    by_cases active : k.val<h.cols
    · simp only [FPReference.input,FPReference.padded,if_pos active,FPMemory.raw,entry]
      rw [stable _ (whereRead k active)]
    · simp [FPReference.input,FPReference.padded,active]
  have hx:=agree "x" h.cols (by intros; exact Or.inl rfl)
  have hw:=agree "w" 0 (by intros; exact Or.inr (Or.inl rfl))
  have hg:=agree "dx" h.cols (by intro k hk; exact Or.inr (Or.inr (Or.inr ⟨⟨k.val,hk⟩,rfl⟩)))
  have hr:=stable ("r",row.val) (Or.inr (Or.inr (Or.inl rfl)))
  unfold expected FPBackwardReference.output FPBackwardReference.rowSum FPBackwardReference.normed FPBackwardReference.weighted
  rw [hx,hw,hg,hr]

def run (M : Algebra α) (h : Call) := GridRun.run (execute M h) (List.finRange h.rows)

theorem complete (M : Algebra α) (h : Call) (heap : Heap α) :
    ∃ final,run M h heap=some final ∧
      (∀ row : Fin h.rows,∀ j,final (address h row j)=expected M h heap row j) ∧
      (∀ addr,(∀ row : Fin h.rows,¬writes h row addr) → final addr=heap addr) := by
  obtain ⟨final,he,ho,hf⟩ := GridRun.complete (execute M h) (fun _ => Fin h.cols) (address h) id
    (expected M h) (reads h) (writes h) (fun row j => ⟨j,rfl⟩) (List.finRange h.rows) (List.nodup_finRange _)
    (fun a _ b _ => separate h a b) (fun w _ => step_complete M h w)
    (fun w _ => expected_stable M h w) heap
  exact ⟨final,he,fun row => ho row (List.mem_finRange row),fun addr out => hf addr (fun row _ => out row)⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPGridBackward
