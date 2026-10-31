import bench.tritonbench_g.attn_fwd_triton.Proofs.FlatPointers
import bench.tritonbench_g.attn_fwd_triton.Proofs.Prologue
import bench.tritonbench_g.attn_fwd_triton.Proofs.Initialization
import bench.tritonbench_g.attn_fwd_triton.Proofs.FirstSetup

/-! Complete candidate prologue from arbitrary initial registers and actual
program ids. No loop-entry or loaded-value invariant is assumed. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatPrologue
open VeriTile.Triton Memory Relocation
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

abbrev writes := Prologue.writes

theorem executes (A : FlatAlloc) (s : BlockState) (p : Parameters) (BM BN BD : Nat) :
    let c := config A (Coordinates.configuration s p)
    ∃ t,stepStmts (Coordinates.code p BM BN ++ A.flattenStmts (Pointers.code p BM BN BD) ++ Initialization.code BM BD p.headDim) s = some t ∧
      Iteration.Fixed s t c BM BN BD ∧ Iteration.Carries t (Initialization.initial BM BD) ∧
      FirstSetup.Caller t c BN BD ∧
      t.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c BM BD) ∧ Frame writes s t := by
  dsimp only
  let c := Coordinates.configuration s p
  let rc := config A c
  obtain ⟨sc,hcs,hcv,hcf⟩ := Coordinates.executes s p BM BN
  have hm : sc.regs .int [] "start_m" = some (Tile.scalar c.startM) := hcv.row
  have hqb : sc.regs .int [] "qvk_offset" = some (Tile.scalar (qbase c)) := hcv.queryBase
  have hkb : sc.regs .int [] "k_offset" = some (Tile.scalar (kbase c)) := hcv.keyBase
  have hvb : sc.regs .int [] "v_offset" = some (Tile.scalar (vbase c)) := hcv.valueBase
  have hob : sc.regs .int [] "o_offset" = some (Tile.scalar (obase c)) := hcv.outputBase
  have hqsb : sc.regs .int [] "q_scale_offset" = some (Tile.scalar (qsbase c BM)) := by
    change sc.regs .int [] "q_scale_offset" = some (Tile.scalar
      ((((c.z*p.heads+c.h)*((p.context+BM-1)/BM)) : Nat) : Int))
    rw [show c.z*p.heads+c.h = s.pids 1 from Coordinates.packed_head s p]
    exact hcv.queryScaleBase
  have hksb : sc.regs .int [] "k_scale_offset" = some (Tile.scalar (ksbase c BN)) := by
    change sc.regs .int [] "k_scale_offset" = some (Tile.scalar
      ((((c.z*p.heads+c.h)*((p.context+BN-1)/BN)) : Nat) : Int))
    rw [show c.z*p.heads+c.h = s.pids 1 from Coordinates.packed_head s p]
    exact hcv.keyScaleBase
  obtain ⟨sp,hps,hpv,hpf⟩ := FlatPointers.executes A sc c BM BN BD hm hqb hkb hvb hob hqsb hksb hcv.keyStride hcv.valueStride
  obtain ⟨t,his,hiv,hic,hif⟩ := Initialization.executes sp rc BM BN BD hpv
    ((hpf.regs _ _ "n_ctx" (by simp [Pointers.writes])).trans hcv.context)
    ((hpf.regs _ _ "stride_kn64" (by simp [Pointers.writes])).trans hcv.keyStride)
    ((hpf.regs _ _ "stride_vk64" (by simp [Pointers.writes])).trans hcv.valueStride)
  have hmemory : sp.mem = s.mem := hpf.mem.trans hcf.mem
  refine ⟨t,?_,Prologue.fixed_memory hiv hmemory,hic,?_,?_,?_⟩
  · exact (stepStmts.append_some ((stepStmts.append_some hcs).trans hps)).trans his
  · constructor
    · exact (hif.regs _ _ "start_m" (by simp [Initialization.writes,RealInit.writes])).trans
        ((hpf.regs _ _ "start_m" (by simp [Pointers.writes])).trans hm)
    · exact (hif.regs _ _ "K_ptrs" (by simp [Initialization.writes,RealInit.writes])).trans hpv.key
    · exact (hif.regs _ _ "K_scale_ptr" (by simp [Initialization.writes,RealInit.writes])).trans hpv.scale
    · exact (hif.regs _ _ "V_ptrs" (by simp [Initialization.writes,RealInit.writes])).trans hpv.value
  · exact (hif.regs _ _ "O_block_ptr" (by simp [Initialization.writes,RealInit.writes])).trans hpv.output
  · exact ((hcf.mono (by simp [writes,Prologue.writes])).trans (hpf.mono (by simp [writes,Prologue.writes]))).trans (hif.mono (by simp [writes,Prologue.writes]))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatPrologue
