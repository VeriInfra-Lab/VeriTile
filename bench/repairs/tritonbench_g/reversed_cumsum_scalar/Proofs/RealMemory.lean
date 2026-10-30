import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.RealValues
import VeriTile.Triton.Memory.Frame.Basic
import VeriTile.Triton.KernelLemmas.ScatterStore

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.RealMemory
open VeriTile.Triton RealValues
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def scatter (h : Call) (c : Config) (row q : Nat) (f : Nat → ℝ) (acc : ℝ) (u : BlockState) : BlockState :=
  (TileShape.allIndices [c.BT]).foldl (fun t idx =>
    if Geometry.active h c q idx.1 then
      t.writeMem h.output (h.outputStrides.address h.H row (Geometry.time h c q idx.1))
        (f (q*c.BT+idx.1.val)-chunkPrefix c.BT q f idx.1+acc)
    else t) u

def footprint (h : Call) (row : Nat) : WriteFootprint :=
  fun cell => cell.1=h.output ∧ ∃ t, t<h.T ∧ cell.2=h.outputStrides.address h.H row t

theorem store_eval (h : Call) (c : Config) (row q : Nat) (f : Nat → ℝ) (acc : ℝ) (u : BlockState)
    (hp : u.regs .ptr [c.BT] "p_o"=some (pointer h c row q h.output h.outputStrides))
    (hm : u.regs .bool [c.BT] "active"=some (mask h c q))
    (hv : u.regs .real [c.BT] "b_o"=some (outputTile c.BT q f acc)) :
    stepStmt (.store .real [c.BT] (.ptr (.ref .ptr [c.BT] "p_o"))
      (.ref .real [c.BT] "b_o") (.mask (.ref .bool [c.BT] "active"))) u=
      some (scatter h c row q f acc u) := by
  simp only [stepStmt,evalOp_ref,hp,hm,hv,Option.bind_some,Option.bind_eq_bind,
    Bool.true_and,pointer,mask,decide_eq_true_eq]
  simp only [Option.map_some,Option.bind_some,decide_eq_true_eq]
  rfl

theorem frame (h : Call) (c : Config) (row q : Nat) (start : q*c.BT<h.T)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState) :
    BlockState.WriteWithin (footprint h row) u (scatter h c row q f acc u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  refine ⟨rfl,Geometry.time h c q idx.1,?_,rfl⟩
  rw [Geometry.active_time h c q idx.1 ha]
  exact (Geometry.active_iff h c q idx.1 start).mp ha

theorem cell (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (row q : Nat)
    (hr : row<h.B*h.H) (start : q*c.BT<h.T) (f : Nat → ℝ) (acc : ℝ) (u : BlockState)
    (i : Fin c.BT) (hi : q*c.BT+i.val<h.T) :
    (scatter h c row q f acc u).mem h.output (h.outputStrides.address h.H row (q*c.BT+i.val))=
      MemCell.real (f (q*c.BT+i.val)-chunkPrefix c.BT q f i+acc) := by
  let idx : TileIndex [c.BT] := (i,PUnit.unit)
  have active : Geometry.active h c q i := (Geometry.active_iff h c q i start).mpr hi
  obtain ⟨before,after,he⟩ := List.append_of_mem (TileShape.mem_allIndices [c.BT] idx)
  have hn := TileShape.allIndices_nodup [c.BT]
  rw [he,List.nodup_append,List.nodup_cons] at hn
  have hafter : idx∉after := hn.2.1.1
  unfold scatter
  rw [he,List.foldl_append,List.foldl_cons,if_pos active]
  rw [foldl_store_preserve_cell]
  · simp [BlockState.writeMem,idx,Geometry.active_time h c q i active]
  · rintro ⟨k,⟨⟩⟩ hk ha eq
    have kt : Geometry.time h c q k<h.T := by
      rw [Geometry.active_time h c q k ha]
      exact (Geometry.active_iff h c q k start).mp ha
    have et := Geometry.same_row h layout row hr _ _ kt hi eq.2
    rw [Geometry.active_time h c q k ha] at et
    have ek : k=i := Fin.ext (by omega)
    subst k
    exact hafter hk

theorem preserves_later (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (row q : Nat)
    (hr : row<h.B*h.H) (start : q*c.BT<h.T) (f : Nat → ℝ) (acc : ℝ) (u : BlockState)
    (t : Nat) (ht : t<h.T) (later : (q+1)*c.BT≤t) :
    (scatter h c row q f acc u).mem h.output (h.outputStrides.address h.H row t)=
      u.mem h.output (h.outputStrides.address h.H row t) := by
  unfold scatter
  apply foldl_store_preserve_cell
  rintro ⟨i,⟨⟩⟩ _ ha eq
  have it : Geometry.time h c q i<h.T := by
    rw [Geometry.active_time h c q i ha]
    exact (Geometry.active_iff h c q i start).mp ha
  have et := Geometry.same_row h layout row hr _ _ it ht eq.2
  rw [Geometry.active_time h c q i ha] at et
  have := i.isLt
  rw [Nat.add_mul,Nat.one_mul] at later
  omega

theorem preserves_input (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (row q : Nat)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState) (off : Nat) :
    (scatter h c row q f acc u).readMem h.input off=u.readMem h.input off := by
  unfold scatter
  exact BlockState.scatter_prop_masked_preserves_other_region h.output
    (fun idx : TileIndex [c.BT] => h.outputStrides.address h.H row (Geometry.time h c q idx.1))
    (fun idx => f (q*c.BT+idx.1.val)-chunkPrefix c.BT q f idx.1+acc)
    (fun idx => Geometry.active h c q idx.1) h.input layout.distinct off _ u

#print axioms cell
#print axioms frame
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.RealMemory
