import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Grid
import Mathlib.Data.List.Nodup

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 4000000

def geometry (s : Shape) (c : Config) : IntScaledMatmul.Revision001.Geometry.Layout :=
  ⟨s.M,s.N,c.BM,c.BN,c.group,s.N,1⟩
namespace Grouped
open IntScaledMatmul.Revision001.Geometry
def valid (s : Shape) (c : Config) : Valid (geometry s c) where
  blockRows := c.positiveM
  blockColumns := c.positiveN
  groupRows := c.groupPositive
  writable := by
    intro x y same
    have hi := outputAddress_injective s (Buffers.mk 0 0 0 0 0 0 0 0 0 0 Bool.false .half .single .half)
    apply hi
    apply Prod.ext
    · rfl
    · simpa [outputAddress,address,geometry] using same

theorem every_cell (s : Shape) (c : Config) (x : MatrixCell s) :
    ∃ pid,pid<count (geometry s c) ∧ Owns (geometry s c) pid x :=
  covers _ (valid s c) x
theorem only_one_owner (s : Shape) (c : Config) (p q : Nat)
    (hp : p<count (geometry s c)) (hq : q<count (geometry s c)) (x : MatrixCell s)
    (hpx : Owns (geometry s c) p x) (hqx : Owns (geometry s c) q x) : p=q :=
  owner_unique _ (valid s c) p q hp hq x hpx hqx
theorem only_one_lane (s : Shape) (c : Config) (pid : Nat) (i j : Lane (geometry s c))
    (hr : outRow (geometry s c) pid i=outRow (geometry s c) pid j)
    (hc : outColumn (geometry s c) pid i=outColumn (geometry s c) pid j) : i=j :=
  lane_injective _ pid i j hr hc
end Grouped

def collect [DecidableEq α] (x : α) (events : List (α × β)) : List β :=
  events.filterMap (fun e => if e.1=x then some e.2 else none)
theorem collect_block [DecidableEq α] (x y : α) (ys : List β) :
    collect x (ys.map (fun z => (y,z)))=if y=x then ys else [] := by
  induction ys with
  | nil => simp [collect]
  | cons z ys ih => by_cases h : y=x <;> simp_all [collect]

theorem collect_absent [DecidableEq α] (xs : List α) (ys : List β) (x : α) (absent : x∉xs) :
    collect x (xs.flatMap (fun y => ys.map (fun z => (y,z))))=[] := by
  induction xs with
  | nil => rfl
  | cons y xs ih =>
    have hx : x∉xs := fun h => absent (List.mem_cons_of_mem _ h)
    have ne : y≠x := by intro h; apply absent; simp [h]
    simp only [List.flatMap_cons,collect,List.filterMap_append]
    change collect x (ys.map (fun z => (y,z)))++collect x (xs.flatMap (fun y => ys.map (fun z => (y,z))))=[]
    rw [collect_block,if_neg ne,ih hx]
    rfl

theorem collect_complete [DecidableEq α] (xs : List α) (ys : List β) (x : α)
    (unique : xs.Nodup) (present : x∈xs) :
    collect x (xs.flatMap (fun y => ys.map (fun z => (y,z))))=ys := by
  induction xs with
  | nil => simp at present
  | cons y xs ih =>
    obtain ⟨ny,nxs⟩ := List.nodup_cons.mp unique
    simp only [List.flatMap_cons,collect,List.filterMap_append]
    change collect x (ys.map (fun z => (y,z)))++collect x (xs.flatMap (fun y => ys.map (fun z => (y,z))))=ys
    by_cases same : y=x
    · rw [collect_block,if_pos same,collect_absent _ _ _ (same ▸ ny),List.append_nil]
    · have hx : x∈xs := by simpa [same,Ne.symm same] using present
      rw [collect_block,if_neg same,ih nxs hx,List.nil_append]

def cells (s : Shape) : List (MatrixCell s) := (List.finRange s.M).product (List.finRange s.N)
def allCommits (s : Shape) (c : Config) : List (Commit s c) :=
  (cells s).flatMap (fun x => (List.finRange c.S).map (fun z => (x,z)))
theorem allCommits_projection (s : Shape) (c : Config) (x : MatrixCell s) :
    projection x (allCommits s c)=List.finRange c.S := by
  apply collect_complete
  · exact (List.nodup_finRange s.M).product (List.nodup_finRange s.N)
  · exact List.mem_product.mpr ⟨by simp,by simp⟩

/-- Any interleaving of the complete source commit set constructs the required
per-cell schedule theorem; completeness is not an output/postcondition oracle. -/
def Schedule.ofPermutation (s : Shape) (c : Config) (events : List (Commit s c))
    (order : events.Perm (allCommits s c)) : Schedule s c where
  events := events
  complete := fun x => by
    have h := order.filterMap (fun e => if e.1=x then some e.2 else none)
    change (projection x events).Perm (projection x (allCommits s c)) at h
    rw [allCommits_projection] at h
    exact h

open IntScaledMatmul.Revision001.Geometry in
structure PhysicalCommit (s : Shape) (c : Config) where
  pid : Fin (count (geometry s c))
  lane : Lane (geometry s c)
  active : Active (geometry s c) pid.val lane
  split : Fin c.S
open IntScaledMatmul.Revision001.Geometry in
def PhysicalCommit.logical (p : PhysicalCommit s c) : Commit s c :=
  (index (geometry s c) p.pid p.lane p.active,p.split)

open IntScaledMatmul.Revision001.Geometry in
theorem physical_covers (s : Shape) (c : Config) (e : Commit s c) :
    ∃ p : PhysicalCommit s c,p.logical=e := by
  obtain ⟨pid,hp,l,hr,hc⟩ := Grouped.every_cell s c e.1
  let p : PhysicalCommit s c := ⟨⟨pid,hp⟩,l,owns_active _ _ _ l hr hc,e.2⟩
  refine ⟨p,?_⟩
  apply Prod.ext
  · apply Prod.ext
    · exact Fin.ext hr
    · exact Fin.ext hc
  · rfl

open IntScaledMatmul.Revision001.Geometry in
theorem physical_unique (s : Shape) (c : Config) (p q : PhysicalCommit s c)
    (same : p.logical=q.logical) : p.pid=q.pid ∧ p.lane=q.lane ∧ p.split=q.split := by
  have hr := congrArg (fun e : Commit s c => e.1.1.val) same
  have hc := congrArg (fun e : Commit s c => e.1.2.val) same
  have hz := congrArg (fun e : Commit s c => e.2) same
  have pid : p.pid.val=q.pid.val := Grouped.only_one_owner s c p.pid q.pid p.pid.isLt q.pid.isLt
    (p.logical.1) ⟨p.lane,rfl,rfl⟩ ⟨q.lane,hr.symm,hc.symm⟩
  have row : outRow (geometry s c) p.pid p.lane=outRow (geometry s c) p.pid q.lane := by
    change outRow (geometry s c) p.pid p.lane=outRow (geometry s c) q.pid q.lane at hr
    simpa only [pid] using hr
  have col : outColumn (geometry s c) p.pid p.lane=outColumn (geometry s c) p.pid q.lane := by
    change outColumn (geometry s c) p.pid p.lane=outColumn (geometry s c) q.pid q.lane at hc
    simpa only [pid] using hc
  exact ⟨Fin.ext pid,Grouped.only_one_lane s c p.pid p.lane q.lane row col,hz⟩

#axiomsClean allCommits_projection
#axiomsClean Schedule.ofPermutation
#axiomsClean physical_covers
#axiomsClean physical_unique
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
