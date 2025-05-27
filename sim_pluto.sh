export PYTHONPATH=$PYTHONPATH:$(pwd)
export NUPLAN_DATA_ROOT="/home/bydguikong/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/bydguikong/nuplan/dataset/maps"
export WS="/home/bydguikong/nuplan"
export NUPLAN_EXP_ROOT="$WS/exp"

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=pluto_planner
CKPT_N=pluto_1M_aux_cil

CKPT=$CKPT_N.ckpt
# BUILDER=nuplan_mini
# FILTER=mini_demo_scenario
# BUILDER=nuplan_challenge
# FILTER=random14_benchmark
BUILDER=nuplan_mini
FILTER=val14_benchmark
VIDEO_SAVE_DIR=$cwd/videos/$PLANNER.$CKPT_N/$FILTER

CHALLENGE="closed_loop_nonreactive_agents"
# CHALLENGE="closed_loop_reactive_agents"
# CHALLENGE="open_loop_boxes"

python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    worker=sequential \
    number_of_gpus_allocated_per_simulation=0 \        # 全 CPU 推理；ScopePlanner 速度仍可接受
    verbose=true \
    experiment_uid="$PLANNER/$FILTER" \
    planner.$PLANNER.render=true \
    planner.$PLANNER.planner_ckpt="$CKPT_ROOT/$CKPT" \
    +planner.$PLANNER.save_dir=$VIDEO_SAVE_DIR/$CHALLENGE.norule \
    planner.$PLANNER.rule_based_evaluator=false \
    planner.$PLANNER.planner.use_hidden_proj=false \
    planner.$PLANNER.planner.cat_x=true \
    planner.$PLANNER.planner.ref_free_traj=true \
    planner.$PLANNER.planner.num_modes=12


    # worker.threads_per_node=32 \
    # worker=sequential \