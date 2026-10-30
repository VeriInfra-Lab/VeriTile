import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.DirectPointers
namespace VeriTile.Bench.Repairs.DequantizeMatmul.Direct
noncomputable section
set_option maxHeartbeats 400000

theorem index_post (c : Request) (config : LaunchConfig) (member : config∈launchConfigs) :
    IndexPost (requestShape c) (config.tile member)
      (canonical2 (columns c.a.info) (columns c.b.info)) (canonical2 (columns c.a.info) (columns c.b.info))
      (canonical1 (columns c.b.info)) := by
  by_cases hk : 0 < columns c.a.info
  · by_cases hn : 0 < columns c.b.info
    · exact (native_admitted _ _ _ _ c.contract hk hn config member).exact
    · have zero : columns c.b.info=0 := by omega
      intro l launched
      have falseBound := launched.2
      change l.programN < gridCount (columns c.b.info) config.blockN at falseBound
      rw [zero] at falseBound
      have positive := (launchConfigs_positive config member).2
      have div : (config.blockN-1)/config.blockN=0 := Nat.div_eq_of_lt (by omega)
      simp [gridCount,div] at falseBound
  · have zero : columns c.a.info=0 := by omega
    intro l launched
    have falseBound := launched.1
    change l.programK < gridCount (columns c.a.info) config.blockK at falseBound
    rw [zero] at falseBound
    have positive := (launchConfigs_positive config member).1
    have div : (config.blockK-1)/config.blockK=0 := Nat.div_eq_of_lt (by omega)
    simp [gridCount,div] at falseBound

theorem checked_word_result (c : Request) (storageBase : Nat → Nat)
    (la : TensorLaws storageBase c.a) (lb : TensorLaws storageBase c.b) (ls : TensorLaws storageBase c.scale)
    (arithmetic : Arithmetic α) (allocate : Allocator α) (before : Memory (Cell α)) (typed : InputStorage c before)
    (caller : Nat → Prop) (alive : caller c.a.key) (blive : caller c.b.key) (slive : caller c.scale.key)
    (tile : Tile) (plan : LibraryPlan (requestShape c)) :
    ∃ result,checkedRun c arithmetic allocate caller alive blive slive tile plan before=.ok result ∧
      result.tensor.device=c.a.info.metadata.device.index ∧
      OutputPost (requestShape c) (provided c) caller before result
        (wordReference arithmetic (logicalInput c arithmetic.zero before) plan) := by
  let result := allocatedHostCall arithmetic allocate (requestShape c) (host c caller alive blive slive)
    (provided c) (output_guard c caller alive blive slive) tile plan before
  refine ⟨result,?_,returned_device c arithmetic allocate caller alive blive slive tile plan before,?_⟩
  · simp [checkedRun,(checkTensors_iff _ _ _ _).mpr c.contract,result,Except.bind,Except.pure,bind,pure]
  · exact host_word_post arithmetic tile plan
      (prepare allocate (requestShape c) (host c caller alive blive slive) (provided c)
        (output_guard c caller alive blive slive) before).allocation
      (logicalInput c arithmetic.zero before)
      (input_represents c storageBase la lb ls arithmetic.zero before typed allocate caller alive blive slive)

#axiomsClean index_post
#axiomsClean checked_word_result
end
end VeriTile.Bench.Repairs.DequantizeMatmul.Direct
