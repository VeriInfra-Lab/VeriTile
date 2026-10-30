import bench.tritonbench_g.chunk_retention.Proofs.InputBindings

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
noncomputable section
variable {α : Type}

/-- Input-array view used only to transport the existing memory-indexed results. -/
def readInputs (zero : α) (a : Call) (read : RegionName→Nat→α) : Whole001.Inputs α where
  q := fun h t e=>read "q" (h*(a.shape.T*a.shape.K)+t*a.shape.K+e)
  k := fun h t e=>read "k" (h*(a.shape.T*a.shape.K)+t*a.shape.K+e)
  v := fun h t j=>read "v" (h*(a.shape.T*a.shape.V)+t*a.shape.V+j)
  incoming := fun h t j=>read "incoming" (h*(a.shape.T*a.shape.V)+t*a.shape.V+j)
  seed := if a.seed then fun h e j=>read "h0" (h*a.shape.K*a.shape.V+e*a.shape.V+j) else fun _ _ _=>zero
  finalIncoming := if a.seed then fun h e j=>read "terminal" (h*a.shape.K*a.shape.V+e*a.shape.V+j) else fun _ _ _=>zero

def readHistory (c : Shape) (read : RegionName→Nat→α) (reverse : Bool) (h n e j : Nat) : α :=
  read (if reverse then "dh" else "h") (h*(c.chunks*c.K*c.V)+n*c.K*c.V+e*c.V+j)

theorem Reads.q_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (enabled : inputEnabled a 0)
    (h t e : Nat) (hh : h<a.shape.B*a.shape.H) (ht : t<a.shape.T) (he : e<a.shape.K) :
    (readInputs zero a read).q h t e=(logicalInputs zero a xs).q h t e := by
  simp only [readInputs,logicalInputs,sample3,dif_pos hh,dif_pos ht,dif_pos he]
  exact hr.q enabled h t e hh ht he

theorem Reads.k_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (enabled : inputEnabled a 1)
    (h t e : Nat) (hh : h<a.shape.B*a.shape.H) (ht : t<a.shape.T) (he : e<a.shape.K) :
    (readInputs zero a read).k h t e=(logicalInputs zero a xs).k h t e := by
  simp only [readInputs,logicalInputs,sample3,dif_pos hh,dif_pos ht,dif_pos he]
  exact hr.k enabled h t e hh ht he

theorem Reads.v_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (enabled : inputEnabled a 2)
    (h t e : Nat) (hh : h<a.shape.B*a.shape.H) (ht : t<a.shape.T) (he : e<a.shape.V) :
    (readInputs zero a read).v h t e=(logicalInputs zero a xs).v h t e := by
  simp only [readInputs,logicalInputs,sample3,dif_pos hh,dif_pos ht,dif_pos he]
  exact hr.v enabled h t e hh ht he

theorem Reads.incoming_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (enabled : inputEnabled a 3)
    (h t e : Nat) (hh : h<a.shape.B*a.shape.H) (ht : t<a.shape.T) (he : e<a.shape.V) :
    (readInputs zero a read).incoming h t e=(logicalInputs zero a xs).incoming h t e := by
  simp only [readInputs,logicalInputs,sample3,dif_pos hh,dif_pos ht,dif_pos he]
  exact hr.incoming enabled h t e hh ht he

theorem Reads.seed_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (stage : a.stage=.forwardH)
    (h e j : Nat) (hh : h<a.shape.B*a.shape.H) (he : e<a.shape.K) (hj : j<a.shape.V) :
    (readInputs zero a read).seed h e j=(logicalInputs zero a xs).seed h e j := by
  cases hs : a.seed
  · simp [readInputs,logicalInputs,hs]
  · simp only [readInputs,logicalInputs,hs,if_true,sample3,dif_pos hh,dif_pos he,dif_pos hj]
    exact hr.h0 ⟨stage,hs⟩ h e j hh he hj

theorem Reads.finalIncoming_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (stage : a.stage=.backwardH)
    (h e j : Nat) (hh : h<a.shape.B*a.shape.H) (he : e<a.shape.K) (hj : j<a.shape.V) :
    (readInputs zero a read).finalIncoming h e j=(logicalInputs zero a xs).finalIncoming h e j := by
  cases hs : a.seed
  · simp [readInputs,logicalInputs,hs]
  · simp only [readInputs,logicalInputs,hs,if_true,sample3,dif_pos hh,dif_pos he,dif_pos hj]
    exact hr.terminal ⟨stage,hs⟩ h e j hh he hj

theorem Reads.history_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (zero : α) (reverse : Bool)
    (enabled : if reverse then inputEnabled a 7 else inputEnabled a 6)
    (h n e j : Nat) (hh : h<a.shape.B*a.shape.H) (hn : n<a.shape.chunks)
    (he : e<a.shape.K) (hj : j<a.shape.V) :
    readHistory a.shape read reverse h n e j=history zero a.shape xs reverse h n e j := by
  cases reverse
  · simpa only [readHistory,history,Bool.false_eq_true,if_false,sample4,dif_pos hh,dif_pos hn,dif_pos he,dif_pos hj]
      using hr.h enabled h n e j hh hn he hj
  · simpa only [readHistory,history,if_true,sample4,dif_pos hh,dif_pos hn,dif_pos he,dif_pos hj]
      using hr.dh enabled h n e j hh hn he hj

theorem Reads.qAt_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (ops : Ops α) (enabled : inputEnabled a 0)
    (h : Nat) (hh : h<a.shape.B*a.shape.H) :
    qAt ops a.shape (readInputs ops.zero a read) h=qAt ops a.shape (logicalInputs ops.zero a xs) h := by
  funext t e
  unfold qAt masked
  split_ifs with active
  · exact hr.q_eq ops.zero enabled h t e hh active.1 active.2
  · rfl

theorem Reads.kAt_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (ops : Ops α) (enabled : inputEnabled a 1)
    (h : Nat) (hh : h<a.shape.B*a.shape.H) :
    kAt ops a.shape (readInputs ops.zero a read) h=kAt ops a.shape (logicalInputs ops.zero a xs) h := by
  funext t e
  unfold kAt masked
  split_ifs with active
  · exact hr.k_eq ops.zero enabled h t e hh active.1 active.2
  · rfl

theorem Reads.vAt_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (ops : Ops α) (enabled : inputEnabled a 2)
    (h : Nat) (hh : h<a.shape.B*a.shape.H) :
    vAt ops a.shape (readInputs ops.zero a read) h=vAt ops a.shape (logicalInputs ops.zero a xs) h := by
  funext t e
  unfold vAt masked
  split_ifs with active
  · exact hr.v_eq ops.zero enabled h t e hh active.1 active.2
  · rfl

theorem Reads.gAt_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (ops : Ops α) (enabled : inputEnabled a 3)
    (h : Nat) (hh : h<a.shape.B*a.shape.H) :
    gAt ops a.shape (readInputs ops.zero a read) h=gAt ops a.shape (logicalInputs ops.zero a xs) h := by
  funext t e
  unfold gAt masked
  split_ifs with active
  · exact hr.incoming_eq ops.zero enabled h t e hh active.1 active.2
  · rfl

theorem Reads.masked_history_eq {a : Call} {xs : Inputs α a.shape} {read : RegionName→Nat→α}
    (hr : Reads a xs read) (ops : Ops α) (reverse : Bool)
    (enabled : if reverse then inputEnabled a 7 else inputEnabled a 6)
    (h n : Nat) (hh : h<a.shape.B*a.shape.H) (hn : n<a.shape.chunks) :
    masked ops a.shape.K a.shape.V (readHistory a.shape read reverse h n)=
      masked ops a.shape.K a.shape.V (history ops.zero a.shape xs reverse h n) := by
  funext e j
  unfold masked
  split_ifs with active
  · exact hr.history_eq ops.zero reverse enabled h n e j hh hn active.1 active.2
  · rfl

#axiomsClean Reads.qAt_eq
#axiomsClean Reads.masked_history_eq
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
