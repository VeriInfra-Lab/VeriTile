import bench.tritonbench_g.iv_dependent_matmul.Sources.Contract
import bench.tritonbench_g.iv_dependent_matmul.Sources.Index64

namespace VeriTile.Bench.Repairs.IvDependentMatmul.Contract
open TritonBenchG.IvDependentMatmul Whole

theorem check_iff (r : Request) (defaultFP32 cuda : Bool) :
    check r defaultFP32 cuda=.ok () ↔ Preconditions r defaultFP32 cuda := by
  constructor
  · intro result
    unfold check at result
    split at result
    · contradiction
    rename_i profile
    split at result
    · contradiction
    rename_i blocks
    split at result
    · contradiction
    rename_i control
    split at result
    · contradiction
    rename_i bytes
    split at result
    · contradiction
    rename_i grid
    have df : defaultFP32=Bool.true := by cases defaultFP32 <;> simp_all
    have cd : cuda=Bool.true := by cases cuda <;> simp_all
    refine ⟨df,cd,?_⟩
    have hb : Block r.blockRows ∧ Block r.blockColumns ∧ Block r.blockInner := by tauto
    exact ⟨hb.1,hb.2.1,hb.2.2,by omega,by omega,by omega,by omega,by omega,by omega,by omega⟩
  · rintro ⟨df,cd,h⟩
    have control : ¬max (r.rows+r.blockRows) (max (r.columns+r.blockColumns) (r.inner+4*r.blockInner+3))≥2^63 := by
      have := h.rowsControl; have := h.columnsControl; have := h.innerControl; omega
    have bytes : ¬4*max (r.rows*r.inner) (max (r.inner*r.columns) (r.rows*r.columns))≥2^63 := by
      have := h.inputBytes; have := h.weightBytes; have := h.outputBytes; omega
    have profile : ¬(defaultFP32=Bool.false ∨ cuda=Bool.false) := by simp [df,cd]
    have blocks : ¬(¬Block r.blockRows ∨ ¬Block r.blockColumns ∨ ¬Block r.blockInner) := by
      have := h.rowsBlock; have := h.columnsBlock; have := h.innerBlock; tauto
    simp only [check,if_neg profile,if_neg blocks,if_neg control,if_neg bytes,if_neg (Nat.not_le.mpr h.grid)]

theorem launchable (r : Request) (h : Admitted r) : Index64.Launchable r where
  rowsBlock := by have := h.rowsBlock.2.1; omega
  columnsBlock := by have := h.columnsBlock.2.1; omega
  innerBlock := by have := h.innerBlock.2.1; omega
  grid := h.grid

theorem source_controls (r : Request) (h : Admitted r) :
    r.rows+r.blockRows<2^63 ∧ r.columns+r.blockColumns<2^63 ∧
    r.inner+4*r.blockInner+3<2^63 :=
  ⟨h.rowsControl,h.columnsControl,h.innerControl⟩

theorem active_byte_offset (extent offset : Nat) (checked : 4*extent<2^63) (active : offset<extent) :
    offset<2^63 ∧ 4*offset<2^63 := by omega

theorem prepared (r r' : Request) (h : Admitted r)
    (dimensions : r'.rows=r.rows ∧ r'.columns=r.columns ∧ r'.inner=r.inner)
    (blocks : r'.blockRows=r.blockRows ∧ r'.blockColumns=r.blockColumns ∧ r'.blockInner=r.blockInner) :
    Admitted r' := by
  cases h
  constructor <;> simp_all [programs,tileRows,tileColumns]

#axiomsClean check_iff
#axiomsClean launchable
#axiomsClean source_controls
#axiomsClean active_byte_offset
end VeriTile.Bench.Repairs.IvDependentMatmul.Contract
