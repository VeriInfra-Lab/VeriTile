import bench.repairs.tritonbench_g.attention_fwd_triton2.IO

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.InputValues
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface
noncomputable section
open Classical
variable {α β : Type} [Inhabited α] [Inhabited β]

/-- Only the actual finite storage ports are required; arbitrary other memory
cells and legal aliases remain admitted. -/
def ReadAddress (p : Parameters) (r : RegionName) (off : Nat) : Prop :=
  (r=p.Q ∧ off<words p) ∨ (r=p.K ∧ off<words p) ∨ (r=p.V ∧ off<words p) ∨
  (r=p.QS ∧ off<p.batches*p.heads*((p.length+127)/128)) ∨
  (r=p.KS ∧ off<p.batches*p.heads*((p.length+63)/64))

theorem reader_bound (M : Algebra β) (dot : Dot β) (p : Parameters)
    (encode : α→β) (observe : TileIndex [p.batches,p.heads,p.length,96]→β→α)
    (work : List Program) (vals : Inputs α p) (initial : State β)
    (hin : (io M dot p encode observe work).InputBinding vals initial)
    (r : RegionName) (off : Nat) (backed : ReadAddress p r off) :
    (initial.mem r off).read .real=encode (reader p vals r off) := by
  unfold reader
  split_ifs with hq hk hv hqs hks
  · rcases hq with ⟨rfl,inside⟩
    exact hin (0 : Fin 5) (⟨off,inside⟩,()) trivial
  · rcases hk with ⟨rfl,inside⟩
    exact hin (1 : Fin 5) (⟨off,inside⟩,()) trivial
  · rcases hv with ⟨rfl,inside⟩
    exact hin (2 : Fin 5) (⟨off,inside⟩,()) trivial
  · rcases hqs with ⟨rfl,inside⟩
    exact hin (3 : Fin 5) (⟨off,inside⟩,()) trivial
  · rcases hks with ⟨rfl,inside⟩
    exact hin (4 : Fin 5) (⟨off,inside⟩,()) trivial
  · exact False.elim (by rcases backed with h|h|h|h|h <;> contradiction)

#axiomsClean reader_bound
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.InputValues
