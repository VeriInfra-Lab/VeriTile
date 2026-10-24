import bench.repairs.tritonbench_g.flash_attn.Proofs.Host
namespace VeriTile.Bench.Repairs.FlashAttn.Source
/-- Raw source pointer expressions, before deriving logical tensor coordinates. -/
def base (s : Shape) (t : Strides) (pidHead : Nat) :=
  (pidHead/s.heads)*t.batch+(pidHead%s.heads)*t.head
def raw (s : Shape) (t : Strides) (pidHead row col : Nat) := base s t pidHead+row*t.row+col*t.dim

theorem raw_logical (s : Shape) (t : Strides) (b h row col : Nat) (hh : h<s.heads) :
    raw s t (b*s.heads+h) row col=address t b h row col := by
  obtain ⟨hd,hm⟩:=flatten_decode s.heads b h hh
  simp [raw,base,hd,hm,address]

theorem own_v_stride (s : Shape) (v : Strides) (pidHead row col : Nat) :
    raw s v pidHead row col=base s v pidHead+row*v.row+col*v.dim := rfl

def scalarResult (x : Layout) (p : Ops α) (m : Memory α) (scale : α) (lane : Lane) : α :=
  match lane.column with
  | none=>logValue x p m scale (lane.pidHead/x.shape.heads) (lane.pidHead%x.shape.heads)
      (lane.pidRow*128+lane.rowLane)
  | some d=>outputValue x p m scale (lane.pidHead/x.shape.heads) (lane.pidHead%x.shape.heads)
      (lane.pidRow*128+lane.rowLane) d

def scalarTarget (x : Layout) (lane : Lane) : Region × Nat :=
  match lane.column with
  | none=>(.l,lane.pidHead*x.shape.length+(lane.pidRow*128+lane.rowLane))
  | some d=>(.o,raw x.shape x.o lane.pidHead (lane.pidRow*128+lane.rowLane) d)

theorem result_binding (x : Layout) (p : Ops α) (m : Memory α) (scale : α) (lane : Lane) :
    scalarResult x p m scale lane=(laneWork x.shape lane).result x p m scale := by
  cases hc : lane.column <;> simp [scalarResult,laneWork,Work.result,hc]

theorem target_binding (x : Layout) (lane : Lane) : scalarTarget x lane=(laneWork x.shape lane).target x := by
  cases hc : lane.column
  · simp only [scalarTarget,hc,laneWork,Work.target,lAddress]
    have h:=Nat.div_add_mod lane.pidHead x.shape.heads
    congr 2
    nlinarith
  · simp [scalarTarget,laneWork,Work.target,hc,raw,base,address]

/-- Launch stores perform their source-specific pointer computation and read
current memory. The row arithmetic body is the explicit lowered graph in Kernels. -/
def one (x : Layout) (p : Ops α) (scale : α) (lane : Lane) (m : Memory α) : Memory α :=
  store m (scalarTarget x lane) (scalarResult x p m scale lane)

def run (x : Layout) (p : Ops α) (scale : α) : List Lane → Memory α → Memory α
  | [],m=>m
  | lane::ls,m=>run x p scale ls (one x p scale lane m)

theorem run_execute (x : Layout) (p : Ops α) (scale : α) (lanes : List Lane) (m : Memory α) :
    run x p scale lanes m=execute x p scale (lanes.map (laneWork x.shape)) m := by
  induction lanes generalizing m with
  | nil=>rfl
  | cons lane lanes ih=>
    simp only [run,List.map_cons,execute,one,target_binding,result_binding,step]
    exact ih _

noncomputable def launch (x : Layout) (p : Ops α) (scale : α) (m : Memory α) : Memory α :=
  run x p scale ((rawLanes x.shape).filter (fun l=>decide ((laneWork x.shape l).active x.shape))) m

theorem launch_execute (x : Layout) (p : Ops α) (scale : α) (m : Memory α) :
    launch x p scale m=execute x p scale (sourceSchedule x.shape) m :=
  run_execute x p scale _ m

#axiomsClean raw_logical
#axiomsClean result_binding
#axiomsClean target_binding
#axiomsClean run_execute
#axiomsClean launch_execute
end VeriTile.Bench.Repairs.FlashAttn.Source
