import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.QuantExecution4
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.QuantStores

namespace VeriTile.Bench.Repairs.KvCacheFilling.QuantValues
open VeriTile.Triton FP.Structural QuantProfile CopySource CopyExecution QuantExecution QuantStores
noncomputable section

def rows8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters) (s : State α) (src : Nat) :=
  row8Data a p s src

def rows4 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters) (s : State α) (src : Nat) :=
  row4Data a p s src

structure Observes {α : Type} (p : Parameters) (dst : Nat) (rows : Bool → Row α) (f : State α) : Prop where
  cache : ∀ key i, i < capacity p key →
    f.mem (output key) (dst*capacity p key+i) = .mk .nat ((rows key).values i).toNat
  metadata : ∀ key i, i < 2 → f.mem (metaRegion key) (dst*2+i) =
    .mk .real (if i = 0 then (rows key).scale else (rows key).zero)

def finished {α : Type} (p : Parameters) (dst : Nat) (rows : Bool → Row α) (s : State α) : State α :=
  writeMetadata dst (rows Bool.true) (rows Bool.false)
    (QuantExecution.scatter p dst Bool.false (rows Bool.false).values
      (QuantExecution.scatter p dst Bool.true (rows Bool.true).values s))

theorem observes_finished {α : Type} (p : Parameters) (dst : Nat) (rows : Bool → Row α)
    (s : State α) (covered : ∀ key, capacity p key ≤ lanes p key) :
    Observes p dst rows (finished p dst rows s) := by
  constructor
  · intro key i hi
    unfold finished
    rw [metadata_cache]
    cases key
    · exact scatter_read p dst Bool.false _ _ i hi (covered Bool.false)
    · rw [scatter_frame p dst Bool.false _ _ _ _ (Or.inl (by decide))]
      exact scatter_read p dst Bool.true _ _ i hi (covered Bool.true)
  · intro key i hi
    unfold finished
    cases key <;> simpa using metadata_read dst (rows Bool.true) (rows Bool.false) _ _ i hi

theorem finished_frame {α : Type} (p : Parameters) (dst : Nat) (rows : Bool → Row α)
    (s : State α) (reg : RegionName) (i : Nat)
    (cacheOutside : ∀ key, reg ≠ output key ∨ i < dst*capacity p key ∨ dst*capacity p key+capacity p key ≤ i)
    (metaOutside : ∀ key, reg ≠ metaRegion key ∨ i < dst*2 ∨ dst*2+2 ≤ i) :
    (finished p dst rows s).mem reg i = s.mem reg i := by
  unfold finished
  rw [metadata_frame _ _ _ _ _ _ metaOutside,
    scatter_frame _ _ _ _ _ _ _ (cacheOutside Bool.false),
    scatter_frame _ _ _ _ _ _ _ (cacheOutside Bool.true)]

theorem observe8 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (hk : 0 < p.bk) (hv : 0 < p.bv) (sk : p.dk = p.ck) (sv : p.dv = p.cv)
    (covered : ∀ key, capacity p key ≤ lanes p key) (version : Bool) :
    Observes p dst (rows8 a p s src) (executed8 a p s src dst hs hd hk hv sk sv version).val :=
  observes_finished p dst _ (prepared8 a p s src dst) covered

theorem observe4 {α : Type} [Inhabited α] (a : Ops α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int))
    (hk : 0 < p.bk) (hv : 0 < p.bv)
    (covered : ∀ key, capacity p key ≤ lanes p key) (version : Bool) :
    Observes p dst (rows4 a p s src) (executed4 a p s src dst hs hd hk hv version).val :=
  observes_finished p dst _ (prepared4 a p s src dst) covered

#axiomsClean observes_finished
#axiomsClean finished_frame
#axiomsClean observe8
#axiomsClean observe4
end
end VeriTile.Bench.Repairs.KvCacheFilling.QuantValues
