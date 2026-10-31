import bench.tritonbench_g.layer_norm_liger.Proofs.ForwardResult
import bench.tritonbench_g.layer_norm_liger.Proofs.HostResults
import bench.tritonbench_g.layer_norm_liger.Proofs.Frame
import bench.tritonbench_g.layer_norm_liger.Proofs.ClosedFP
namespace VeriTile.Bench.Repairs.LayerNormLiger.Reader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum softwareSum SumPrimitive)
open Logical ClosedFP
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
theorem forward_logical (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ)
    (hb : p.columns≤p.block) (hmy : p.mean≠p.y) (hmr : p.mean≠p.rstd) (hry : p.rstd≠p.y) :
    ∃ t, FP.Structural.exec (withSum (algebra rest) sumPrimitive) (Forward.kernel p) s=some t ∧
      t.mem p.mean (Forward.meanOffset p s)=.mk .fp32 (mean p.columns (rowX p s)) ∧
      t.mem p.rstd (Forward.rstdOffset p s)=.mk .fp32 (rstd p.columns p.epsilon (rowX p s)) ∧
      (∀ i : TileIndex [p.block], i.1.val<p.columns →
        t.mem p.y (Forward.outputOffset p s i)=.mk .fp32
          (y p.columns p.epsilon (rowX p s) (weight p s) (bias p s) i.1.val)) ∧
      (∀ r o, Forward.Outside p s r o → t.mem r o=s.mem r o) := Logical.forward_all_exec rest p s hb hmy hmr hry

theorem backward_logical (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (fresh : FreshDX p) (hwd : p.dw≠p.db) (hxd : p.dx≠p.dw) (hxb : p.dx≠p.db)
    (hb : p.columns≤p.block) (hs : p.columns≤p.dxStride) :
    ∃ t, FP.Structural.exec (withSum (algebra rest) sumPrimitive) (Backward.kernel p) s = some t ∧
      (∀ i : TileIndex [p.block], i.1.val<p.columns →
        t.mem p.dw (s.pids 0*p.dwStride+i.1.val)=.mk .fp32 (groupDW p s i.1.val) ∧
        t.mem p.db (s.pids 0*p.dbStride+i.1.val)=.mk .fp32 (groupDB p s i.1.val)) ∧
      (∀ row, Backward.start p s≤row → row<Backward.stop p s →
        ∀ i : TileIndex [p.block], i.1.val<p.columns →
          t.mem p.dx (row*p.dxStride+i.1.val)=.mk .fp32 (rowDX p s row i.1.val)) := Logical.backward_exec rest p s fresh hwd hxd hxb hb hs

theorem host_dw (p : Backward.Parameters) (s : State ℝ) (programs : Nat)
    (hp : 0<programs) (hr : 0<p.rows)
    (hq : p.rowsPerProgram=Launch.rowsPerProgram p.rows programs) (i : Nat) :
    (∑ pid ∈ Finset.range programs, groupDW p (atPid s pid) i) =
      dw p.rows (fun row => bx p (saved p s) row) (fun row => bdy p (saved p s) row)
        (fun row => Backward.mean p (saved p s) row) (fun row => Backward.rstd p (saved p s) row) i := Logical.host_dw p s programs hp hr hq i

theorem host_db (p : Backward.Parameters) (s : State ℝ) (programs : Nat)
    (hp : 0<programs) (hr : 0<p.rows)
    (hq : p.rowsPerProgram=Launch.rowsPerProgram p.rows programs) (i : Nat) :
    (∑ pid ∈ Finset.range programs, groupDB p (atPid s pid) i) =
      db p.rows (fun row => bdy p (saved p s) row) i := Logical.host_db p s programs hp hr hq i

theorem coverage (p : Backward.Parameters) (s : State ℝ) (programs row : Nat)
    (hp : 0<programs) (hr : row<p.rows)
    (hq : p.rowsPerProgram=Launch.rowsPerProgram p.rows programs) :
    ∃! pid, pid<programs ∧ Backward.start p (atPid s pid)≤row ∧ row<Backward.stop p (atPid s pid) := Logical.dx_coverage p s programs row hp hr hq

theorem forward_bits [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Forward.Parameters) (s : State (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (withSum (Forward.software rest cfg literal rsqrt32) (softwareSum cfg))
      (Forward.kernel p) s=some t ∧ t.mem=(forwardReference cfg literal rsqrt32 p s).mem := ClosedFP.forward_exec rest cfg literal rsqrt32 p s

theorem backward_bits [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Backward.Parameters) (s : State (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (withSum (Forward.software rest cfg literal rsqrt32) (softwareSum cfg))
      (Backward.kernel p) s=some t ∧ t.mem=(backwardReference cfg literal p s).mem := ClosedFP.backward_exec rest cfg literal rsqrt32 p s

theorem backward_frame {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Backward.Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (withSum M R) (Backward.kernel p) s=some t ∧
      ∀ r o, Backward.Outside p s r o → t.mem r o=s.mem r o := Backward.complete_frame M R p s

#axiomsClean forward_logical
#axiomsClean backward_logical
#axiomsClean host_dw
#axiomsClean host_db
#axiomsClean coverage
#axiomsClean forward_bits
#axiomsClean backward_bits
#axiomsClean backward_frame
#specNonCircular Logical.mean avoiding [Forward.kernel, Backward.kernel, Forward.reference, Backward.reference]
#specNonCircular Logical.variance avoiding [Forward.kernel, Backward.kernel, Forward.reference, Backward.reference]
#specNonCircular Logical.y avoiding [Forward.kernel, Backward.kernel, Forward.reference, Backward.reference]
#specNonCircular Logical.dx avoiding [Forward.kernel, Backward.kernel, Forward.reference, Backward.reference]
#specNonCircular Logical.dw avoiding [Forward.kernel, Backward.kernel, Forward.reference, Backward.reference]
#specNonCircular Logical.db avoiding [Forward.kernel, Backward.kernel, Forward.reference, Backward.reference]
#specNonCircular ClosedFP.forwardReference avoiding [Forward.kernel, Backward.kernel, Forward.reference, Backward.reference]
#specNonCircular ClosedFP.backwardReference avoiding [Forward.kernel, Backward.kernel, Forward.reference, Backward.reference]

#auditStmt forward_logical
#auditStmt backward_logical
end
end VeriTile.Bench.Repairs.LayerNormLiger.Reader
