import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawBackendMemory
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeBackendIO

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBackend
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Math Program Backend
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000

def logicalPrepared {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (s : State α) (b h k j : Nat) : α :=
  if h < c.heads ∧ j < c.chunk ∧ k*c.chunk+j < c.sequence then
    prepare c p hasBias useSoftplus lower upper Bool.true
      ((s.mem r.dt (b*c.dtBatch+(k*c.chunk+j)*c.dtTime+h*c.dtHead)).read .fp32)
      ((s.mem r.bias (h*c.biasHead)).read .real)
  else p.model.literal (some .fp32) .real 0.0

def logicalRate {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (s : State α) (h : Nat) : α :=
  p.model.cast (some .fp32) .fp32 .real ((s.mem r.rate (h*c.rateHead)).read .fp32)

/-- An independent logical-coordinate expression, with the profile's actual
ordered sum tree. Products and additions remain explicit model operations. -/
def logicalCumulative {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (s : State α) (b h k : Nat) (j : Fin c.BC) : α :=
  (p.plan.tree ⟨h%c.BH,Nat.mod_lt h c.headPositive⟩ j).eval (p.model.binary (some .fp32) .real .add) p.fused (fun t =>
    p.model.binary (some .fp32) .real .mul
      (logicalPrepared c p r hasBias useSoftplus lower upper s b h k t.val)
      (logicalRate c p r s h))

theorem prepared_logical {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (l : Launch) (hl : c.valid l) (s : State α) (i : TileIndex [c.BH,c.BC]) :
    preparedTile c p r hasBias useSoftplus lower upper l s i =
      logicalPrepared c p r hasBias useSoftplus lower upper s
        l.batch (head c.layout l i.1.val) l.chunk i.2.1.val := by
  have hs := Whole.chunk_start_lt c l hl
  have hm : activeInput c.layout l i.1.val i.2.1.val ↔
      head c.layout l i.1.val < c.heads ∧ i.2.1.val < c.chunk ∧
        l.chunk*c.chunk+i.2.1.val < c.sequence := by
    change (head c.layout l i.1.val < c.heads ∧ i.2.1.val < min c.chunk (c.sequence-l.chunk*c.chunk)) ↔ _
    omega
  have ha : dtAddress c.layout l i.1.val i.2.1.val =
      l.batch*c.dtBatch+(l.chunk*c.chunk+i.2.1.val)*c.dtTime+
        head c.layout l i.1.val*c.dtHead := by
    simp [dtAddress,Whole.Host.layout]
    ring
  by_cases active : activeInput c.layout l i.1.val i.2.1.val
  · have hh := (hm.mp active).1
    simp only [preparedTile,active,decide_true,dtRaw,if_true,biasRaw,hh,ha,
      logicalPrepared,hm.mp active,and_self]
  · have hn := mt hm.mpr active
    simp [preparedTile,active,prepare,logicalPrepared,hn]

theorem tile_frame {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (l : Launch) (s : State α) (rg : RegionName) (off : Nat)
    (outside : ¬ Execution.footprint c.layout l r (rg,off)) :
    (sourceRun c p r hasBias useSoftplus lower upper l s).val.mem rg off = s.mem rg off := by
  have ho : rg ≠ r.out ∨ ∀ i : TileIndex [c.BH,c.BC],
      activeOutput c.layout l i.1.val i.2.1.val → off ≠ outAddress c.layout l i.1.val i.2.1.val := by
    by_cases he : rg = r.out
    · refine Or.inr (fun i hi h => outside (Or.inl ⟨he,i.1,i.2.1,hi,h⟩))
    · exact Or.inl he
  have hs : rg ≠ r.sum ∨ ∀ i : TileIndex [c.BH,c.BC],
      activeOutput c.layout l i.1.val i.2.1.val → off ≠ outAddress c.layout l i.1.val i.2.1.val := by
    by_cases he : rg = r.sum
    · refine Or.inr (fun i hi h => outside (Or.inr ⟨he,i.1,i.2.1,hi,h⟩))
    · exact Or.inl he
  rw [sourceRun_mem]
  simp only [tileMemory]
  rw [scatter_frame _ _ _ _ _ _ _ hs,scatter_frame _ _ _ _ _ _ _ ho]

theorem tile_values {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (fresh : Whole.Fresh r) (hasBias useSoftplus : Bool)
    (lower upper : α) (l : Launch) (hl : c.valid l) (s : State α)
    (i : TileIndex [c.BH,c.BC]) (active : activeOutput c.layout l i.1.val i.2.1.val) :
    (sourceRun c p r hasBias useSoftplus lower upper l s).val.mem
      r.out (outAddress c.layout l i.1.val i.2.1.val) =
        .mk .real (logicalPrepared c p r hasBias useSoftplus lower upper s
          l.batch (head c.layout l i.1.val) l.chunk i.2.1.val) ∧
    (sourceRun c p r hasBias useSoftplus lower upper l s).val.mem
      r.sum (outAddress c.layout l i.1.val i.2.1.val) =
        .mk .real (logicalCumulative c p r hasBias useSoftplus lower upper s
          l.batch (head c.layout l i.1.val) l.chunk i.2.1) := by
  have hh : head c.layout l i.1.val < c.heads := active.1
  have hrow : (⟨head c.layout l i.1.val%c.BH,Nat.mod_lt _ c.headPositive⟩ : Fin c.BH) = i.1 := by
    apply Fin.ext
    simp [head,Whole.Host.layout,Nat.add_mod,Nat.mod_eq_of_lt i.1.isLt]
  rw [sourceRun_mem]
  simp only [tileMemory]
  constructor
  · rw [scatter_frame _ _ _ _ _ _ _ (Or.inl fresh.outSum),scatter_readback c l hl _ _ _ i active]
    rw [prepared_logical c p r hasBias useSoftplus lower upper l hl]
  · rw [scatter_readback c l hl _ _ _ i active]
    congr 1
    unfold logicalCumulative
    rw [hrow]
    congr 1
    funext t
    rw [prepared_logical c p r hasBias useSoftplus lower upper l hl]
    congr 1
    simp only [rateRaw,logicalRate,hh,if_true]
    rw [scatter_frame _ _ _ _ _ _ _ (Or.inl fresh.rateOut)]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBackend
