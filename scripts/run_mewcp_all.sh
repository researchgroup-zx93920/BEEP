#!/bin/sh
TASKS=(sgm)
# GRAPHS=(cit-Patents_adj com-youtube com-orkut as-Skitter) # 
# GRAPHS=(com-youtube com-lj com-orkut as-Skitter cit-Patents_adj)
# GRAPHS=(tina)
# GRAPHS=(com-youtube com-lj cit-Patents_adj com-orkut com-Friendter)
# GRAPHS=(com-dblp)
GRAPHS=(soc-pokec)

MODES=(node)
for task in ${TASKS[@]}
do
for temp in ${GRAPHS[@]};
do
for mode in ${MODES[@]};
do
    TEMP_PATH="/home/almasri3/samiran/dataset/gbin/${temp}.bel"
    # TEMP_PATH="../tests/${temp}.bel"
    #TEMP_PATH="../dataset/RGG/rgg_n_2_${temp}_s0.bel"
    echo -e "\n\n************* Processing Data Graph $temp **************"
    ./run_mewcp_1.sh $task $TEMP_PATH $mode
done
done
done
