import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Execution

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Readback
open VeriTile.Triton FP.Structural
open Execution
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 48000
variable {α : Type} [Inhabited α]

def combinedOffset (p : Parameters) (s : State α) (q : TileIndex [3,p.blockSize]) :=
  coordinateOffset p s q.2+q.1.val

def combinedValue (M : Algebra α) (p : Parameters) (s : State α)
    (q : TileIndex [3,p.blockSize]) : Cell α := .mk .real (backwardValue M p s q.1 q.2)

theorem written_scatter (M : Algebra α) (p : Parameters) (s : State α) :
    written M p s=(TileShape.allIndices [3,p.blockSize]).foldl (fun t q=>
      if combinedOffset p s q<3*p.rows then
        t.write gradientRegion (combinedOffset p s q) (combinedValue M p s q) else t) s := by
  rw [TileShape.allIndices_cons]
  simp only [List.foldl_flatMap,List.foldl_map]
  rfl

theorem output_injective (p : Parameters) (s : State α) :
    Function.Injective (combinedOffset p s) := by
  intro a b eq
  have ha:=a.1.isLt
  have hb:=b.1.isLt
  have lane : a.2.1.val=b.2.1.val := by
    unfold combinedOffset coordinateOffset at eq
    omega
  have component : a.1.val=b.1.val := by
    unfold combinedOffset coordinateOffset at eq
    rw [lane] at eq
    omega
  exact Prod.ext (Fin.ext component) (Prod.ext (Fin.ext lane) (Subsingleton.elim _ _))

def row (p : Parameters) (s : State α) (q : TileIndex [p.blockSize]) :=
  q.1.val+p.blockSize*s.pids 0

def backwardExpected (M : Algebra α) (p : Parameters) (s : State α) (r : Nat) (component : Fin 3) : α :=
  Reference.backwardValues M
    (M.fp32Load ((s.mem coordinateRegion (r*3)).read .real))
    (M.fp32Load ((s.mem coordinateRegion (r*3+1)).read .real))
    (M.fp32Load ((s.mem coordinateRegion (r*3+2)).read .real))
    (fun c=>M.fp32Load ((s.mem upstreamRegion (r*p.upstreamRowStride+(p.columnOffset+c.val)*p.upstreamColStride)).read .real))
    (fun c=>M.fp32Load ((s.mem gradientRegion (r*3+c.val)).read .real)) component

theorem backwardValue_active (M : Algebra α) (p : Parameters) (s : State α)
    (component : Fin 3) (q : TileIndex [p.blockSize]) (fit : p.columnOffset+11≤p.outputWidth) (active : row p s q<p.rows) :
    backwardValue M p s component q=backwardExpected M p s (row p s q) component := by
  have coord : coordinateOffset p s q=row p s q*3 := by
    simp only [coordinateOffset,row]; ring
  have up : upstreamOffset p s q=row p s q*p.upstreamRowStride+p.columnOffset*p.upstreamColStride := by
    rfl
  have hc : ∀ c : Fin 3,row p s q*3+c.val<3*p.rows := by
    intro c
    have := c.isLt
    omega
  have logical : logicalOffset p s q=row p s q*p.outputWidth+p.columnOffset := by
    simp only [logicalOffset,row]; ring
  have hg : ∀ c : Fin 11,row p s q*p.outputWidth+p.columnOffset+c.val<p.outputWidth*p.rows := by
    intro c
    have hc := c.isLt
    have hmul := Nat.mul_le_mul_right p.outputWidth (Nat.succ_le_of_lt active)
    rw [Nat.succ_mul,Nat.mul_comm p.rows p.outputWidth] at hmul
    omega
  have address (c : Fin 11) :
      row p s q*p.upstreamRowStride+p.columnOffset*p.upstreamColStride+c.val*p.upstreamColStride=
      row p s q*p.upstreamRowStride+(p.columnOffset+c.val)*p.upstreamColStride := by ring
  have h1 : row p s q*3+1<3*p.rows := by simpa using hc 1
  have h2 : row p s q*3+2<3*p.rows := by simpa using hc 2
  simp only [backwardValue,backwardExpected,coord,up,load,maskedLoad,logical,address,
    show row p s q*3<3*p.rows from by have := hc 0;simp at this;exact this,
    h1,h2,hc,hg,↓reduceIte]

theorem backward_correct (M : Algebra α) (p : Parameters) (s : State α)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ t,FP.Structural.exec M (backwardKernel p) s=some t ∧
      (∀ c q,row p s q<p.rows → t.mem gradientRegion (row p s q*3+c.val)=
        .mk .real (backwardExpected M p s (row p s q) c)) ∧
      (∀ region index,(region≠gradientRegion ∨ ∀ (c : Fin 3) q,
        row p s q<p.rows → index≠row p s q*3+c.val) → t.mem region index=s.mem region index) := by
  refine ⟨(backward M p s).val,(backward M p s).property,?_,?_⟩
  · intro c q active
    have offsetEq : combinedOffset p s (c,q)=row p s q*3+c.val := by
      simp only [combinedOffset,coordinateOffset,row];ring
    have bound : combinedOffset p s (c,q)<3*p.rows := by
      rw [offsetEq]
      have := c.isLt
      omega
    rw [backward_memory,written_scatter,←offsetEq]
    have h:=State.masked_scatter_readback gradientRegion (combinedOffset p s)
      (combinedValue M p s) (fun q=>combinedOffset p s q<3*p.rows)
      s (c,q) bound (output_injective p s)
    simpa [combinedValue,backwardValue_active M p s c q fit active] using h
  · intro region index outside
    rw [backward_memory,written_scatter]
    apply State.masked_scatter_frame
    rcases outside with h|h
    · exact Or.inl h
    · right
      intro q _ active
      have offsetEq : combinedOffset p s q=row p s q.2*3+q.1.val := by
        simp only [combinedOffset,coordinateOffset,row];ring
      have rowActive : row p s q.2<p.rows := by rw [offsetEq] at active;omega
      simpa only [offsetEq] using h q.1 q.2 rowActive

#axiomsClean backward_correct
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Readback
