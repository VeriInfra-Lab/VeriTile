import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.GPUCorrect
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.GPUFPEquiv
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.TorchMM
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open Host
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 24000

/-- Ordinary external torch.mm(a, fp_b, out=out) API contract, evaluated in
mathematical reals. It is used only after the complete proved GPU dequant grid;
it is not a substitute for either task-owned matmul GPU algorithm. -/
structure RealAPI where
  run : Arguments → State ℝ → Option (State ℝ)
  localSpec : ∀ a, Admissible a .s1 → ∀ s, ∃ final,
    run a s=some final ∧
    Grid.Post (SecondGrid.geometry (second a)) a.c.region s final
      (fun x => .mk .fp16 (Real.sum (fun k =>
        (s.mem a.a.region (x.1.val*a.a.rowStride+k*a.a.columnStride)).read .fp16 *
        (s.mem a.d.region (k*a.d.rowStride+x.2.val*a.d.columnStride)).read .fp16) a.K))

def s1RealRun (api : RealAPI) (a : Arguments) (h : Admissible a .s1) (jobs : List Nat) (s : State ℝ) :=
  (Grid.run (Correct.dequant16Program a h) jobs s).bind (api.run a)

theorem s1_real (api : RealAPI) (a : Arguments) (h : Admissible a .s1)
    (abi : ABI.IndexABI a .s1 .m64n256) (initial : State ℝ) (packed : Packed a initial)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a))))) :
    ABI.SourceExact a .s1 .m64n256 ∧ ∃ middle final,
      Grid.run (Correct.dequant16Program a h) jobs initial=some middle ∧
      api.run a middle=some final ∧ s1RealRun api a h jobs initial=some final ∧
      (∀ x : Geometry.Cell (SecondGrid.geometry (second a)),
        final.mem a.c.region (Geometry.address (SecondGrid.geometry (second a)) x)=
          .mk .fp16 (Real.rawMatmul (RealFirst.inputs (first a .m64n256) initial)
            a.K a.group x.1.val x.2.val)) ∧
      (∀ r o, r≠a.d.region →
        (r≠a.c.region ∨ ∀ x : Geometry.Cell (SecondGrid.geometry (second a)),
          o≠Geometry.address (SecondGrid.geometry (second a)) x) → final.mem r o=initial.mem r o) ∧
      returned .s1 a=some a.c.identity := by
  obtain ⟨hx,middle,hd,dr,df⟩ := Correct.dequant16 a h abi (Or.inr rfl) initial packed jobs schedule
  obtain ⟨final,hm,mr,mf⟩ := api.localSpec a h middle
  refine ⟨hx,middle,final,hd,hm,?_,?_,?_,?_⟩
  · simp [s1RealRun,hd,hm]
  · intro x
    refine (mr x).trans ?_
    apply congrArg (fun v : ℝ => (Cell.mk .fp16 v : Cell ℝ))
    unfold Real.rawMatmul
    apply Real.sum_congr
    intro k hk
    have ha : a.a.region≠a.d.region := by
      have hh := h.separate
      simp only [tensors,List.pairwise_cons,List.mem_cons,List.mem_singleton,forall_eq_or_imp] at hh
      tauto
    have aeq := df a.a.region (x.1.val*a.a.rowStride+k*a.a.columnStride) (Or.inl ha)
    let y : Geometry.Cell (DequantGrid16.geometry (Host.dequant16 a)) := (⟨k,hk⟩,x.2)
    have deq := dr y
    change middle.mem a.d.region (k*a.d.rowStride+x.2.val*a.d.columnStride)=_ at deq
    rw [aeq,deq]
    rfl
  · intro r o hd' hc
    exact (mf r o hc).trans (df r o (Or.inl hd'))
  · simp [returned]

/-- Native torch.mm remains an ordinary external library call. There is no
invented MMA order, tensor result oracle or FP=Real theorem for this API. The
call relation observes the actual input and fully dequantized scratch tensor. -/
structure NativeAPI where
  run : Arguments → State Native.Expr → Option (State Native.Expr)
  call : Arguments → State Native.Expr → State Native.Expr → Prop
  localSpec : ∀ a, Admissible a .s1 → ∀ s, ∃ final,
    run a s=some final ∧ call a s final ∧
    (∀ r o, (r≠a.c.region ∨ ∀ x : Geometry.Cell (SecondGrid.geometry (second a)),
      o≠Geometry.address (SecondGrid.geometry (second a)) x) → final.mem r o=s.mem r o)

def s1NativeRun (api : NativeAPI) (a : Arguments) (h : Admissible a .s1) (jobs : List Nat) (s : State Native.Expr) :=
  (Grid.run (DequantGrid16.program Native.scalar (Host.dequant16 a) (d16_valid a h)
    (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2) jobs s).bind (api.run a)

theorem s1_native (api : NativeAPI) (a : Arguments) (h : Admissible a .s1)
    (abi : ABI.IndexABI a .s1 .m64n256) (initial : State Native.Expr) (packed : Packed a initial) (words : Native.InputWords a initial)
    (plan : Native.Plan) (isa : Native.WordISA)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a))))) :
    ABI.SourceExact a .s1 .m64n256 ∧ ∃ middle final,
      Grid.run (DequantGrid16.program Native.scalar (Host.dequant16 a) (d16_valid a h)
        (distinct a h).2.2.2.2.1 (distinct a h).2.2.2.2.2.1 (distinct a h).2.2.2.2.2.2) jobs initial=some middle ∧
      (∀ pid, pid<Geometry.count (DequantGrid16.geometry (Host.dequant16 a)) → ∀ i,
        Geometry.Active (DequantGrid16.geometry (Host.dequant16 a)) pid i →
        (Native.lower plan ((middle.mem a.d.region (Geometry.offset (DequantGrid16.geometry (Host.dequant16 a)) pid i)).read .fp16) pid).eval isa=
          (Native.lower plan (VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.value Native.scalar
            (Host.dequant16 a) (DequantGrid16.atProgram (Host.dequant16 a) initial pid) i) pid).eval isa) ∧
      api.run a middle=some final ∧ api.call a middle final ∧
      s1NativeRun api a h jobs initial=some final ∧
      (∀ r o, r≠a.d.region →
        (r≠a.c.region ∨ ∀ x : Geometry.Cell (SecondGrid.geometry (second a)),
          o≠Geometry.address (SecondGrid.geometry (second a)) x) → final.mem r o=initial.mem r o) ∧
      returned .s1 a=some a.c.identity := by
  obtain ⟨hx,middle,hd,dr,dgraph,df,coverage⟩ := FPEquiv.dequant16 a .s1 h (Or.inr rfl) abi initial packed words plan isa jobs schedule
  obtain ⟨final,hm,hcall,mf⟩ := api.localSpec a h middle
  refine ⟨hx,middle,final,hd,dgraph,hm,hcall,?_,?_,?_⟩
  · simp [s1NativeRun,hd,hm]
  · intro r o hd' hc
    exact (mf r o hc).trans (df r o (Or.inl hd'))
  · simp [returned]

#axiomsClean s1_real
#axiomsClean s1_native
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.TorchMM
