import bench.tritonbench_g.chunk_cumsum_vector.Kernels
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Precision and masking regressions inspect the current repair source,
including its explicit zero padding and guarded tail coordinates. -/
namespace FPVectorCumsumSourceTests
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkCumsumVector
set_option maxRecDepth 24000

def loop (h : Call) (c : Config) : Source.DynamicRange :=
  (Source.dynamicRange? (Kernels.source h c).surfaceBody "i_t").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

theorem load_explicitly_promotes (h : Call) (c : Config) :
    (loop h c).body[7]? = some (.assign .real [c.BT,BS] "b_s"
      (.compute (.alg .fp32 (.castFloat h.inputDType .real
        (.load h.inputDType.toTileDType (.ptr (.ref .ptr [c.BT,BS] "p_s"))
          (.maskOther (.ref .bool [c.BT,BS] "active")
            (.broadcast (.castFloat .real h.inputDType (.const 0)) [c.BT,BS]))))))) := rfl

theorem dot_disables_tf32 (h : Call) (c : Config) :
    (loop h c).body[8]? = some (.assign .real [c.BT,BS] "b_c"
      (.compute (.alg .fp32 (.add .real (.consL (.consSame .nil))
        (.expandDim ⟨0, by simp⟩ (.ref .real [BS] "b_z"))
        (.dot (batch := []) (.ref .real [c.BT,c.BT] "m_s") (.ref .real [c.BT,BS] "b_s")
          {allowTF32 := some Bool.false}))))) := rfl

theorem output_cast_and_both_boundaries (h : Call) (c : Config) :
    (loop h c).body[9]? = some (.store h.outputDType.toTileDType [c.BT,BS]
      (.ptr (.ref .ptr [c.BT,BS] "p_z"))
      (.alg (.castFloat .real h.outputDType
        (.withPrecision .real .fp32 (.ref .real [c.BT,BS] "b_c"))))
      (.mask (.ref .bool [c.BT,BS] "active"))) ∧
    (loop h c).body[4]? = some (.assign .bool [c.BT,BS] "active"
      (.alg (.boolAnd (.consR (.consL .nil))
        (.expandDim ⟨1,by simp⟩ (.ref .bool [c.BT] "active_t"))
        (.expandDim ⟨0,by simp⟩ (.ref .bool [BS] "active_s"))))) := ⟨rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end FPVectorCumsumSourceTests
