import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Returns

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage
noncomputable section
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform} {service : Service platform api neg inputs Metadata}
  {c : SavedContext} {base : Base} {scratch : Scratch} {layout : BaseLayout base c inputs}
  {sl : ScratchLayout base scratch c inputs} {initial : State Store Metadata}

def LaunchesRoute.dxValue (r : LaunchesRoute service c base scratch layout sl initial) (i : Nat) :=
  backwardValue (platform.scalar r.first.code.object.primitives) r.first.code.plan
    (Execution.E.saved api c.r c.mu r.first.s2.memory (i/c.columns))
    (Execution.E.rows c.x c.columns r.first.s2.memory api (i/c.columns))
    (Execution.E.rows base.dy c.columns r.first.s2.memory api (i/c.columns))
    (Execution.E.vector c.w r.first.s2.memory api) (i%c.columns)

def LaunchesRoute.dwValue (r : LaunchesRoute service c base scratch layout sl initial) (i : Nat) :=
  finalValue (platform.scalar r.last.code.object.primitives) r.last.code.finalPlan
    (min (Geometry.groups c.columns) c.rows)
    (fun group => Execution.E.rows scratch.dw c.columns r.last.s2.memory api group i) i

def LaunchesRoute.dbValue (r : LaunchesRoute service c base scratch layout sl initial) (i : Nat) :=
  finalValue (platform.scalar r.last.code.object.primitives) r.last.code.finalPlan
    (min (Geometry.groups c.columns) c.rows)
    (fun group => Execution.E.rows scratch.db c.columns r.last.s2.memory api group i) i

theorem launches_live (r : LaunchesRoute service c base scratch layout sl initial)
    (t : Storage.Tensor) (present : initial.live t) : r.final.live t :=
  r.last.preTransition.live _ (r.first.preTransition.live _ present)

theorem launches_dw_native (r : LaunchesRoute service c base scratch layout sl initial) (i : Fin base.dw.count) :
    Storage.logical api neg r.final.memory base.dw.tensor i.val=r.dwValue i.val := by
  rw [dense_logical api neg _ _ _ i.isLt]
  have ho : base.dw.owner=c.x.owner := layout.owners 1
  rw [ho]
  exact Runtime.route_readback r.last 0 i

theorem launches_db_native (r : LaunchesRoute service c base scratch layout sl initial) (i : Fin base.db.count) :
    Storage.logical api neg r.final.memory base.db.tensor i.val=r.dbValue i.val := by
  rw [dense_logical api neg _ _ _ i.isLt]
  have ho : base.db.owner=c.x.owner := layout.owners 2
  rw [ho]
  exact Runtime.route_readback r.last 1 i

theorem launches_dx_native (r : LaunchesRoute service c base scratch layout sl initial)
    (present : initial.live base.dx.tensor) (i : Fin base.dx.count) :
    Storage.logical api neg r.final.memory base.dx.tensor i.val=r.dxValue i.val := by
  have ho : base.dx.owner=c.x.owner := layout.owners 0
  have fresh : ∀ k,Storage.Disjoint ((finalBuffers base layout).buffers.output k).tensor.backing base.dx.tensor.backing := by
    intro k
    fin_cases k
    · exact layout.separated 1 0 (by decide)
    · exact layout.separated 2 0 (by decide)
  have hp := Execution.run_input (finalBuffers base layout).buffers api
    (Execution.finalValues (platform.scalar r.last.code.object.primitives) r.last.code.finalPlan api
      (finalBuffers base layout) (min (Geometry.groups c.columns) c.rows) scratch.dw scratch.db r.last.s2.memory)
    r.last.schedule r.last.s2.memory base.dx ho fresh i
  rw [dense_logical api neg _ _ _ i.isLt]
  apply hp.trans
  have ht := r.last.preTransition.values base.dx.tensor (r.first.preTransition.live _ present) i.val
    (by rw [←base.dx.countEq]; exact i.isLt)
  rw [dense_logical api neg _ _ _ i.isLt,dense_logical api neg _ _ _ i.isLt] at ht
  apply ht.trans
  rw [ho]
  exact Runtime.route_readback r.first 0 i

#axiomsClean launches_dx_native
#axiomsClean launches_dw_native
#axiomsClean launches_db_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
