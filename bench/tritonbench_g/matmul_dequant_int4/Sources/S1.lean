import bench.tritonbench_g.matmul_dequant_int4.Sources.LibraryPlan
import bench.tritonbench_g.matmul_dequant_int4.Sources.Real
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.S1
set_option maxHeartbeats 4000000
noncomputable section

/-- Actual production order: complete dequantization, then the selected public
torch.mm operation with out. No fused JIT invocation is inserted into this API. -/
def run (s : Shape) (dequant : DequantOps α) (library : Library.Ops α β)
    (plan : Library.Selected s) (dequantOrder outputOrder : List Work) (initial : State α) :=
  Library.run s library plan outputOrder (dequantRun s dequant dequantOrder initial)
def reference (s : Shape) (dequant : DequantOps α) (library : Library.Ops α β)
    (plan : Library.Selected s) (initial : State α) (r n : Nat) :=
  library.narrow ((plan.output r n).eval library
    (fun k => initial.values .a (r*s.k+k))
    (fun k => dequantValue s dequant initial ⟨k,n⟩))

theorem whole (s : Shape) (dequant : DequantOps α) (library : Library.Ops α β)
    (plan : Library.Selected s) (dequantGrid : Grid s.k s.n) (outputGrid : Grid s.m s.n)
    (dequantOrder outputOrder : List Work) (hd : dequantGrid.Complete dequantOrder)
    (ho : outputGrid.Complete outputOrder) (initial : State α)
    (r n : Nat) (hr : r<s.m) (hn : n<s.n) :
    (run s dequant library plan dequantOrder outputOrder initial).values .output (r*s.n+n)=
      reference s dequant library plan initial r n := by
  unfold run
  rw [Library.whole_result s library plan outputGrid outputOrder ho _ r n hr hn]
  have ha : (fun k : Fin s.k => (dequantRun s dequant dequantOrder initial).values .a (r*s.k+k))=
      (fun k : Fin s.k => initial.values .a (r*s.k+k)) := by
    funext k
    exact dequant_frame s dequant dequantGrid dequantOrder hd initial .a _ (Or.inl (by decide))
  have hb : (fun k : Fin s.k => (dequantRun s dequant dequantOrder initial).values .dequant (k*s.n+n))=
      (fun k : Fin s.k => dequantValue s dequant initial ⟨k,n⟩) := by
    funext k
    exact dequant_complete s dequant dequantGrid dequantOrder hd initial k n k.isLt hn
  change library.narrow ((plan.output r n).eval library _ _)=_
  rw [ha,hb]
  rfl

theorem frame (s : Shape) (dequant : DequantOps α) (library : Library.Ops α β)
    (plan : Library.Selected s) (dequantGrid : Grid s.k s.n) (outputGrid : Grid s.m s.n)
    (dequantOrder outputOrder : List Work) (hd : dequantGrid.Complete dequantOrder)
    (ho : outputGrid.Complete outputOrder) (initial : State α)
    (r : Region) (a : Nat) (outsideOutput : r≠.output ∨ s.m*s.n≤a)
    (outsideScratch : r≠.dequant ∨ s.k*s.n≤a) :
    (run s dequant library plan dequantOrder outputOrder initial).values r a=initial.values r a := by
  unfold run
  rw [Library.whole_frame s library plan outputGrid outputOrder ho _ r a outsideOutput,
    dequant_frame s dequant dequantGrid dequantOrder hd initial r a outsideScratch]

theorem reference_real (s : Shape) (plan : Library.Selected s) (initial : State ℝ)
    (r n : Nat) (hr : r<s.m) (hn : n<s.n) :
    reference s realDequant Library.realOps plan initial r n=mathematical s initial r n := by
  unfold reference
  change (plan.output r n).eval Library.realOps _ _=_
  rw [Library.Plan.real_complete _ (plan.complete r n hr hn)]
  simp_rw [standalone_real_value]
  exact Library.ofFn_sum (fun k => logicalA s initial r k*dequantizedReal s initial k n) s.k

theorem whole_real (s : Shape) (plan : Library.Selected s)
    (dequantGrid : Grid s.k s.n) (outputGrid : Grid s.m s.n)
    (dequantOrder outputOrder : List Work) (hd : dequantGrid.Complete dequantOrder)
    (ho : outputGrid.Complete outputOrder) (initial : State ℝ)
    (r n : Nat) (hr : r<s.m) (hn : n<s.n) :
    (run s realDequant Library.realOps plan dequantOrder outputOrder initial).values .output (r*s.n+n)=
      mathematical s initial r n := by
  rw [whole s realDequant Library.realOps plan dequantGrid outputGrid dequantOrder outputOrder hd ho initial r n hr hn,
    reference_real s plan initial r n hr hn]

#axiomsClean whole
#axiomsClean frame
#axiomsClean reference_real
#axiomsClean whole_real
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.S1
