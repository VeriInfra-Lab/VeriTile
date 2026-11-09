import bench.tritonbench_g.attn_fwd_triton.Proofs.HelperLoops

/-! Transport a full frame through a bijective helper-register view. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualFrames
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open RegisterViews
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

theorem mapped {names : RegName ≃ RegName} {writes : List RegName} {s t : BlockState}
    (frame : Frame writes (pull names s) (pull names t)) : Frame (writes.map names) s t := by
  refine ⟨frame.mem,frame.pids,frame.undef,?_⟩
  intro dtype shape name outside
  have oldOutside : names.symm name ∉ writes := by
    intro inside
    apply outside
    exact List.mem_map.mpr ⟨names.symm name,inside,by simp⟩
  simpa only [pull,Equiv.apply_symm_apply] using
    frame.regs dtype shape (names.symm name) oldOutside

/-- A finite overapproximation of the two helpers' private registers. -/
def workspace : List RegName :=
  ["__veritile_statement_jit_0",
   "__veritile_statement_jit_1",
   "__veritile_statement_jit_2",
   "__veritile_statement_jit_3",
   "__veritile_statement_jit_4",
   "__veritile_statement_jit_5",
   "__veritile_statement_jit_6",
   "__veritile_statement_jit_7",
   "__veritile_statement_jit_8",
   "__veritile_statement_jit_9",
   "__veritile_statement_jit_10",
   "__veritile_statement_jit_11",
   "__veritile_statement_jit_12",
   "__veritile_statement_jit_13",
   "__veritile_statement_jit_14",
   "__veritile_statement_jit_15",
   "__veritile_statement_jit_16",
   "__veritile_statement_jit_17",
   "__veritile_statement_jit_18",
   "__veritile_statement_jit_19",
   "__veritile_statement_jit_20",
   "__veritile_statement_jit_21",
   "__veritile_statement_jit_22",
   "__veritile_statement_jit_23",
   "__veritile_statement_jit_24",
   "__veritile_statement_jit_25",
   "__veritile_statement_jit_26",
   "__veritile_statement_jit_27",
   "__veritile_statement_jit_28",
   "__veritile_statement_jit_29",
   "__veritile_statement_jit_30",
   "__veritile_statement_jit_31",
   "__veritile_statement_jit_32",
   "__veritile_statement_jit_33",
   "__veritile_statement_jit_34",
   "__veritile_statement_jit_35",
   "__veritile_statement_jit_36",
   "__veritile_statement_jit_37",
   "__veritile_statement_jit_local_14_lo",
   "__veritile_statement_jit_local_14_hi",
   "__veritile_statement_jit_local_14_start_n",
   "__veritile_statement_jit_local_14_k_mask",
   "__veritile_statement_jit_local_14_k",
   "__veritile_statement_jit_local_14_k_scale",
   "__veritile_statement_jit_local_14_qk",
   "__veritile_statement_jit_local_14_mask",
   "__veritile_statement_jit_local_14_m_ij",
   "__veritile_statement_jit_local_14_p",
   "__veritile_statement_jit_local_14_l_ij",
   "__veritile_statement_jit_local_14_alpha",
   "__veritile_statement_jit_local_14_v",
   "__veritile_statement_jit_local_33_lo",
   "__veritile_statement_jit_local_33_hi",
   "__veritile_statement_jit_local_33_start_n",
   "__veritile_statement_jit_local_33_k_mask",
   "__veritile_statement_jit_local_33_k",
   "__veritile_statement_jit_local_33_k_scale",
   "__veritile_statement_jit_local_33_qk",
   "__veritile_statement_jit_local_33_mask",
   "__veritile_statement_jit_local_33_m_ij",
   "__veritile_statement_jit_local_33_p",
   "__veritile_statement_jit_local_33_l_ij",
   "__veritile_statement_jit_local_33_alpha",
   "__veritile_statement_jit_local_33_v"]

theorem loop_writes (second : Bool) :
    (Iteration.writes second).map (HelperRegisters.names second) ⊆ workspace := by
  cases second <;> decide

theorem fixed {mem s t : BlockState} {c : Memory.Config} {BD : Nat}
    (before : Iteration.Fixed mem s c 128 64 BD) (frame : Frame workspace s t) :
    Iteration.Fixed mem t c 128 64 BD := by
  refine ⟨frame.mem.trans before.memory,?_,?_,?_,?_,?_,?_,?_⟩
  all_goals first
    | exact (frame.regs _ _ _ (by decide)).trans before.query
    | exact (frame.regs _ _ _ (by decide)).trans before.queryScale
    | exact (frame.regs _ _ _ (by decide)).trans before.rows
    | exact (frame.regs _ _ _ (by decide)).trans before.keys
    | exact (frame.regs _ _ _ (by decide)).trans before.context
    | exact (frame.regs _ _ _ (by decide)).trans before.keyStride
    | exact (frame.regs _ _ _ (by decide)).trans before.valueStride

#axiomsClean mapped
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualFrames
