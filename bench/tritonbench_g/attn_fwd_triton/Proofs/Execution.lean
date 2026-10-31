import bench.tritonbench_g.attn_fwd_triton.Proofs.Loop
import bench.tritonbench_g.attn_fwd_triton.Proofs.Stores

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph
noncomputable section
set_option maxHeartbeats 3000000

def result (p : Config) (BD : Nat) (memory : RegionName→Nat→Cell Expr) (pm ph : Nat) : Carry BD :=
  let f := fixed p BD memory pm ph
  let first := fold p BD (reads memory) f (upper f Bool.false) Bool.false
    (lower f Bool.false) (startCursor f Bool.false) (initialCarry BD)
  fold p BD (reads memory) f (upper f Bool.true) Bool.true
    (lower f Bool.true) (startCursor f Bool.true) first

def reference (p : Config) (BD : Nat) (hD : p.headDim≤BD)
    (memory : RegionName→Nat→Cell Expr) (c : Coordinate p) : Cell Expr :=
  .mk .bf16 (output (result p BD memory (owner p c).1.val (owner p c).2.val) (lane p BD hD c))

theorem before (p : Config) (BD : Nat) (stage : p.stage=3) (s : State Expr) :
    ∃ t, run (engine backend) ((code p BD).take 41) s=some t ∧
      Bound (fixed p BD s.mem (s.pids 0) (s.pids 1))
        (result p BD s.mem (s.pids 0) (s.pids 1)) t ∧ t.mem=s.mem := by
  let f := fixed p BD s.mem (s.pids 0) (s.pids 1)
  obtain ⟨s0,h0,b0,m0⟩ := initial_exact p BD s
  obtain ⟨s1,h1,b1,l1,m1⟩ := setup_exact p BD stage Bool.false f (initialCarry BD) s0 b0
  obtain ⟨s2,h2,b2,m2⟩ := loop_exact p BD stage Bool.false f (initialCarry BD) s1 b1 l1
  have mm1 : s1.mem=s.mem := m1.trans m0
  rw [mm1] at b2
  obtain ⟨s3,h3,b3,l3,m3⟩ := setup_exact p BD stage Bool.true f _ s2 b2
  obtain ⟨s4,h4,b4,m4⟩ := loop_exact p BD stage Bool.true f _ s3 b3 l3
  have mm3 : s3.mem=s.mem := m3.trans (m2.trans mm1)
  rw [mm3] at b4
  refine ⟨s4,?_,b4,m4.trans mm3⟩
  have assembled : (code p BD).take 41=(code p BD).take 27 ++ setupCode p BD Bool.false ++
      loopCode p BD Bool.false ++ setupCode p BD Bool.true ++ loopCode p BD Bool.true := by
    by_cases hd : p.stage=2
    all_goals simp [code,source,AttnFwdTriton.TypedSource.repaired,ComputeKernel.surfaceBody,
      AttnFwdTriton.ControlView.compute,setupCode,loopCode,body,lowerOp,lowerName,upperName,hd]
  rw [assembled]
  simp only [run_append,h0,h1,h2,h3,h4,Option.bind_some]

theorem program (p : Config) (BD : Nat) (stage : p.stage=3) (hn : NonOverlapping p)
    (initial : State Expr) (idx : ProgramIndex p) :
    ∃ t, FP.Structural.exec (engine backend) (source p BD Bool.false) (atProgram p initial idx)=some t ∧
      Frame p BD idx.1.val idx.2.val (atProgram p initial idx) t ∧
      (∀ i : TileIndex [128,BD], outputActive p BD idx.1.val i=Bool.true →
        t.mem p.Out (outputAddress p BD idx.1.val idx.2.val i)=
          Cell.mk .bf16 (output (result p BD initial.mem idx.1.val idx.2.val) i)) := by
  let s := atProgram p initial idx
  obtain ⟨u,hu,hbu,_⟩ := before p BD stage s
  have hbu' : Bound (fixed p BD initial.mem idx.1.val idx.2.val)
      (result p BD initial.mem idx.1.val idx.2.val) u := hbu
  obtain ⟨t,ht⟩ := tail_success backend p BD Bool.false u (bound_ready _ _ _ hbu')
  have hex : FP.Structural.exec (engine backend) (source p BD Bool.false) s=some t := by
    change run (engine backend) (source p BD Bool.false).surfaceBody s=some t
    rw [←code_run backend p BD Bool.false s,←List.take_append_drop 41 (code p BD),run_append]
    change (run (engine backend) ((code p BD).take 41) s).bind
      (run (engine backend) (tail p BD Bool.false))=some t
    rw [hu]
    exact ht
  obtain ⟨other,ho,hf⟩ := executes_framed backend p BD (support BD) Bool.false s
  have eq : other=t := Option.some.inj (ho.symm.trans hex)
  subst other
  exact ⟨t,hex,hf,fun i hi=>tail_read p BD idx initial.mem _ u t hbu' hn ht i hi⟩

theorem reference_eq (p : Config) (BD : Nat) (stage : p.stage=3)
    (hn : NonOverlapping p) (hD : p.headDim≤BD) (initial : State Expr) (c : Coordinate p) :
    AttnFwdTriton.TypedExecution.reference backend p BD (support BD) initial c=
      reference p BD hD initial.mem c := by
  obtain ⟨t,ht,_,hread⟩ := program p BD stage hn initial (owner p c)
  have he : referenceState backend p BD (support BD) initial (owner p c)=t :=
    Option.some.inj ((reference_exec backend p BD (support BD) initial (owner p c)).symm.trans ht)
  unfold AttnFwdTriton.TypedExecution.reference
  rw [he]
  have hr := hread (lane p BD hD c) (lane_active p BD hD c)
  rw [lane_address] at hr
  exact hr

#specNonCircular result avoiding [FP.Structural.exec,FP.Structural.run,AttnFwdTriton.TypedExecution.reference]
#specNonCircular reference avoiding [FP.Structural.exec,FP.Structural.run,AttnFwdTriton.TypedExecution.reference]
#axiomsClean program
#axiomsClean reference_eq
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
