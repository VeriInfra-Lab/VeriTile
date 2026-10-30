import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealGridForward

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealGridBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.FastRmsLayernorm
open RealGridForward (Address Heap entry read)
noncomputable section

def address (h : Call) (row : Fin h.rows) (j : Fin h.cols) : Address := ("dx",row.val*h.cols+j.val)
def writes (h : Call) (row : Fin h.rows) (a : Address) : Prop := ∃ j : Fin h.cols,a=address h row j
def reads (h : Call) (row : Fin h.rows) (a : Address) : Prop := a.1="x" ∨ a.1="w" ∨ a.1="r" ∨ writes h row a
def raw (h : Call) (heap : Heap) (row : Fin h.rows) (role : RegionName) (stride : Nat) (j : Fin h.block) : ℝ :=
  read (heap (role,row.val*stride+j.val))
def expected (h : Call) (heap : Heap) (row : Fin h.rows) (j : Fin h.cols) : ℝ :=
  RealMath.gradient h.gemma h.cols (read (heap ("r",row.val)))
    (raw h heap row "x" h.cols) (raw h heap row "dx" h.cols) (raw h heap row "w" 0) (RealBridge.lane h j)
def execute (h : Call) (row : Fin h.rows) (heap : Heap) : Option Heap :=
  (VeriTile.Triton.exec (Kernels.backward h).eraseDType.toAlgKernel (entry heap row.val h.rows)).map
    (fun v a => v.mem a.1 a.2)

theorem separate (h : Call) (a b : Fin h.rows) (ne : a≠b) (addr : Address)
    (ha : writes h a addr) : ¬writes h b addr ∧ ¬reads h b addr := by
  have disjoint : ¬writes h b addr := by
    rintro ⟨j,hb⟩
    obtain ⟨i,ha⟩:=ha
    exact ne (Fin.ext (Geometry.row_injective h a.val b.val i j (congrArg Prod.snd (ha.symm.trans hb))).1)
  refine ⟨disjoint,?_⟩
  intro hr
  obtain ⟨j,rfl⟩:=ha
  apply disjoint
  simpa [reads,address] using hr

theorem step_complete (h : Call) (row : Fin h.rows) (heap : Heap) :
    ∃ final,execute h row heap=some final ∧
      (∀ j,read (final (address h row j))=expected h heap row j) ∧
      (∀ addr,¬writes h row addr → final addr=heap addr) := by
  obtain ⟨v,he,hy,hf⟩ := RealRows.backward_values h (entry heap row.val h.rows)
  refine ⟨fun a => v.mem a.1 a.2,by simp [execute,he],?_,?_⟩
  · intro j
    have weights : raw h heap row "w" 0=(fun k : Fin h.block => read (heap ("w",k.val))) := by
      funext k
      simp [raw]
    unfold expected
    rw [weights]
    exact hy j
  · rintro ⟨q,o⟩ outside
    apply hf
    rintro ⟨same,j,eq⟩
    exact outside ⟨j,Prod.ext same eq⟩

theorem expected_stable (h : Call) (row : Fin h.rows) (a b : Heap)
    (stable : ∀ addr,reads h row addr → a addr=b addr) (j : Fin h.cols) :
    expected h a row j=expected h b row j := by
  have agree (role : RegionName) (stride : Nat)
      (whereRead : ∀ k : Fin h.block,k.val<h.cols → reads h row (role,row.val*stride+k.val)) :
      RealMath.padded h.cols (raw h a row role stride)=RealMath.padded h.cols (raw h b row role stride) := by
    funext k
    by_cases active : k.val<h.cols
    · simp only [RealMath.padded,if_pos active,raw]
      rw [stable _ (whereRead k active)]
    · simp [RealMath.padded,active]
  have hx:=agree "x" h.cols (by intros; exact Or.inl rfl)
  have hw:=agree "w" 0 (by intros; exact Or.inr (Or.inl rfl))
  have hg:=agree "dx" h.cols (by intro k hk; exact Or.inr (Or.inr (Or.inr ⟨⟨k.val,hk⟩,rfl⟩)))
  have hr:=stable ("r",row.val) (Or.inr (Or.inr (Or.inl rfl)))
  unfold expected RealMath.gradient
  rw [hx,hw,hg,hr]

def run (h : Call) := GridRun.run (execute h) (List.finRange h.rows)

theorem complete (h : Call) (heap : Heap) :
    ∃ final,run h heap=some final ∧
      (∀ row : Fin h.rows,∀ j,read (final (address h row j))=expected h heap row j) ∧
      (∀ addr,(∀ row : Fin h.rows,¬writes h row addr) → final addr=heap addr) := by
  obtain ⟨final,he,ho,hf⟩ := GridRun.complete (execute h) (fun _ => Fin h.cols) (address h) read
    (expected h) (reads h) (writes h) (fun row j => ⟨j,rfl⟩) (List.finRange h.rows) (List.nodup_finRange _)
    (fun a _ b _ => separate h a b) (fun w _ => step_complete h w)
    (fun w _ => expected_stable h w) heap
  exact ⟨final,he,fun row => ho row (List.mem_finRange row),fun addr out => hf addr (fun row _ => out row)⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealGridBackward
