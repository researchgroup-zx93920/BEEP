#!/bin/sh
TASKS=(sgm)
GRAPHS=(uci) # com-orkut com-Friendter # com-orkut com-Friendter) cit-Patents_adj
# GRAPHS=(../tests/test)
# GRAPHS=(com-youtube com-lj cit-Patents_adj com-orkut com-Friendter)
MODES=(node)
for task in ${TASKS[@]}
do
for temp in ${GRAPHS[@]};
do
for mode in ${MODES[@]};
do
    TEMP_PATH="../../dataset/template_graphs/gbin/${temp}.bel"
    # TEMP_PATH="../tests/${temp}.bel"/
    #TEMP_PATH="../dataset/RGG/rgg_n_2_${temp}_s0.bel"
    echo -e "\n\n************* Processing Data Graph $temp **************"
    ./run_mewcp_1.sh $task $TEMP_PATH $mode
done
done
done