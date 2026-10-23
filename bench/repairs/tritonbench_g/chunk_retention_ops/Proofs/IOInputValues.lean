import bench.repairs.tritonbench_g.chunk_retention_ops.KernelOnly.Formula
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.IOReadLocality

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
noncomputable section
variable {α : Type} [Inhabited α]

/-- A read-only scalar projection used to connect the existing source results.
Its indices remain the source indices; it contains no execution or result. -/
def readInputs (zero : α) (a : Call) (read : RegionName→Nat→α) : Whole001.Inputs α where
  q := fun h t e=>read "q" (h*(a.shape.T*a.shape.K)+t*a.shape.K+e)
  k := fun h t e=>read "k" (h*(a.shape.T*a.shape.K)+t*a.shape.K+e)
  v := fun h t j=>read "v" (h*(a.shape.T*a.shape.V)+t*a.shape.V+j)
  incoming := fun h t j=>read "incoming" (h*(a.shape.T*a.shape.V)+t*a.shape.V+j)
  seed := fun h e j=>if a.inputState then read "h0" (h*a.shape.K*a.shape.V+e*a.shape.V+j) else zero
  finalIncoming := fun h e j=>if a.inputState then read "terminal" (h*a.shape.K*a.shape.V+e*a.shape.V+j) else zero

def readHistory (a : Call) (read : RegionName→Nat→α) (reverse : Bool) : Tile α :=
  fun e j=>read (if reverse then "dhistory" else "h")
    (a.pid.z*(a.shape.chunks*a.shape.K*a.shape.V)+a.pid.y*a.shape.K*a.shape.V+e*a.shape.V+j)

def Reads (a : Call) (xs : Inputs α a.shape) (read : RegionName→Nat→α) : Prop :=
  ∀ port j,inputEnabled a port →
    read (inputs a.shape)[port].buf (inputOffset a port j)=
      (Cell.mk (inputs a.shape)[port].dtype (xs port j)).read .real

theorem Reads.q_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (enabled : inputEnabled a 0)
    (t e : Nat) (ht : t<a.shape.T) (he : e<a.shape.K) :
    (readInputs zero a read).q a.pid.z t e=(logicalInputs zero a xs).q a.pid.z t e := by
  simp only [readInputs,logicalInputs,sample2,dif_pos ht,dif_pos he]
  exact hr ⟨0,by simp [inputs]⟩ (⟨t,ht⟩,⟨e,he⟩,()) enabled

theorem Reads.k_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (enabled : inputEnabled a 1)
    (t e : Nat) (ht : t<a.shape.T) (he : e<a.shape.K) :
    (readInputs zero a read).k a.pid.z t e=(logicalInputs zero a xs).k a.pid.z t e := by
  simp only [readInputs,logicalInputs,sample2,dif_pos ht,dif_pos he]
  exact hr ⟨1,by simp [inputs]⟩ (⟨t,ht⟩,⟨e,he⟩,()) enabled

theorem Reads.v_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (enabled : inputEnabled a 2)
    (t e : Nat) (ht : t<a.shape.T) (he : e<a.shape.V) :
    (readInputs zero a read).v a.pid.z t e=(logicalInputs zero a xs).v a.pid.z t e := by
  simp only [readInputs,logicalInputs,sample2,dif_pos ht,dif_pos he]
  exact hr ⟨2,by simp [inputs]⟩ (⟨t,ht⟩,⟨e,he⟩,()) enabled

theorem Reads.incoming_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (enabled : inputEnabled a 3)
    (t e : Nat) (ht : t<a.shape.T) (he : e<a.shape.V) :
    (readInputs zero a read).incoming a.pid.z t e=(logicalInputs zero a xs).incoming a.pid.z t e := by
  simp only [readInputs,logicalInputs,sample2,dif_pos ht,dif_pos he]
  exact hr ⟨3,by simp [inputs]⟩ (⟨t,ht⟩,⟨e,he⟩,()) enabled

theorem Reads.seed_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (stage : a.stage=.forwardState)
    (e j : Nat) (he : e<a.shape.K) (hj : j<a.shape.V) :
    (readInputs zero a read).seed a.pid.z e j=(logicalInputs zero a xs).seed a.pid.z e j := by
  cases hs : a.inputState
  · simp [readInputs,logicalInputs,hs]
  · simp only [readInputs,logicalInputs,hs,if_true,sample2,dif_pos he,dif_pos hj]
    exact hr ⟨4,by simp [inputs]⟩ (⟨e,he⟩,⟨j,hj⟩,()) ⟨stage,hs⟩

theorem Reads.finalIncoming_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (stage : a.stage=.backwardState)
    (e j : Nat) (he : e<a.shape.K) (hj : j<a.shape.V) :
    (readInputs zero a read).finalIncoming a.pid.z e j=(logicalInputs zero a xs).finalIncoming a.pid.z e j := by
  cases hs : a.inputState
  · simp [readInputs,logicalInputs,hs]
  · simp only [readInputs,logicalInputs,hs,if_true,sample2,dif_pos he,dif_pos hj]
    exact hr ⟨5,by simp [inputs]⟩ (⟨e,he⟩,⟨j,hj⟩,()) ⟨stage,hs⟩

theorem Reads.history_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (reverse : Bool)
    (enabled : if reverse then inputEnabled a 7 else inputEnabled a 6)
    (e j : Nat) (he : e<a.shape.K) (hj : j<a.shape.V) :
    readHistory a read reverse e j=history zero a.shape xs reverse e j := by
  cases reverse
  · simpa only [readHistory,history,Bool.false_eq_true,if_false,sample2,dif_pos he,dif_pos hj]
      using hr ⟨6,by simp [inputs]⟩ (⟨e,he⟩,⟨j,hj⟩,()) enabled
  · simpa only [readHistory,history,if_true,sample2,dif_pos he,dif_pos hj]
      using hr ⟨7,by simp [inputs]⟩ (⟨e,he⟩,⟨j,hj⟩,()) enabled

theorem Reads.qAt_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (ops : Ops α) (enabled : inputEnabled a 0) :
    qAt ops a.shape (readInputs ops.zero a read) a.pid.z=
      qAt ops a.shape (logicalInputs ops.zero a xs) a.pid.z := by
  funext t e
  unfold qAt masked
  split_ifs with active
  · exact hr.q_eq ops.zero enabled t e active.1 active.2
  · rfl

theorem Reads.kAt_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (ops : Ops α) (enabled : inputEnabled a 1) :
    kAt ops a.shape (readInputs ops.zero a read) a.pid.z=
      kAt ops a.shape (logicalInputs ops.zero a xs) a.pid.z := by
  funext t e
  unfold kAt masked
  split_ifs with active
  · exact hr.k_eq ops.zero enabled t e active.1 active.2
  · rfl

theorem Reads.vAt_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (ops : Ops α) (enabled : inputEnabled a 2) :
    vAt ops a.shape (readInputs ops.zero a read) a.pid.z=
      vAt ops a.shape (logicalInputs ops.zero a xs) a.pid.z := by
  funext t e
  unfold vAt masked
  split_ifs with active
  · exact hr.v_eq ops.zero enabled t e active.1 active.2
  · rfl

theorem Reads.gAt_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (ops : Ops α) (enabled : inputEnabled a 3) :
    gAt ops a.shape (readInputs ops.zero a read) a.pid.z=
      gAt ops a.shape (logicalInputs ops.zero a xs) a.pid.z := by
  funext t e
  unfold gAt masked
  split_ifs with active
  · exact hr.incoming_eq ops.zero enabled t e active.1 active.2
  · rfl

theorem Reads.masked_history_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (ops : Ops α) (reverse : Bool)
    (enabled : if reverse then inputEnabled a 7 else inputEnabled a 6) :
    masked ops a.shape.K a.shape.V (readHistory a read reverse)=
      masked ops a.shape.K a.shape.V (history ops.zero a.shape xs reverse) := by
  funext e j
  unfold masked
  split_ifs with active
  · exact hr.history_eq ops.zero reverse enabled e j active.1 active.2
  · rfl

#axiomsClean Reads.qAt_eq
#axiomsClean Reads.masked_history_eq
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
