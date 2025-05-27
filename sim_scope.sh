export PYTHONPATH=$PYTHONPATH:$(pwd)
###
 # @Author: CGB cai.guanbin@byd.com
 # @Date: 2025-03-17 14:00:47
 # @LastEditors: linmenan 314378011@qq.com
 # @LastEditTime: 2025-04-23 09:39:08
 # @FilePath: /PlanScope/sim_scope.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
export NUPLAN_DATA_ROOT="/home/bydguikong/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/bydguikong/nuplan/dataset/maps"
export WS="/home/bydguikong/nuplan"
export NUPLAN_EXP_ROOT="$WS/exp"
export HYDRA_FULL_ERROR=1

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=scope_planner
CKPT_N=last
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

    # worker=sequential \
python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    scenario_filter.limit_total_scenarios=1 \
    verbose=true \
    worker=ray_distributed worker.threads_per_node=1 \
    number_of_gpus_allocated_per_simulation=1.0 \
    distributed_mode='LOG_FILE_BASED' \
    experiment_uid="$PLANNER/$FILTER" \
    planner.$PLANNER.render=false \
    planner.$PLANNER.planner_ckpt="$CKPT_ROOT/$CKPT" \
    +planner.$PLANNER.save_dir=$VIDEO_SAVE_DIR/$CHALLENGE.norule \
    planner.$PLANNER.rule_based_evaluator=false \
    planner.$PLANNER.planner.cat_x=true \
    planner.$PLANNER.planner.ref_free_traj=true \
    planner.$PLANNER.planner.use_hidden_proj=true \
    planner.$PLANNER.planner.num_modes=12 \
    planner.$PLANNER.planner.future_steps=80 \
    planner.$PLANNER.planner.recursive_decoder=false \
    +planner.$PLANNER.planner.multihead_decoder=false \
    +planner.$PLANNER.planner.wtd_with_history=false \
    +planner.$PLANNER.planner.independent_detokenizer=false 


    # worker=sequential \
    # worker.threads_per_node=12 \