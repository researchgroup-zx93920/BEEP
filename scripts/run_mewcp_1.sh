#!/bin/sh

# ./run_mewcp.sh <path to data graph>
TASK=$1
GRAPH=`pwd`/$2
MODE=$3

echo "Input Graph: $GRAPH"
echo "ProcessBy: $MODE"
echo "Task: $TASK"

# TEMPLATES=(diamond) # fan4 wheel5 cq6m1 cq6) # cq7 cq8 cq9 cq10) 
TEMPLATES=(cq6m1)
# TEMPLATES=(cq5m1 house pyramid fan3)
#TEMPLATES=(cq6m1)
#TEMPLATES=(cq7 cq8 cq9 wheel6 wheel7)
#TEMPLATES=(cq6 wheel5)
#TEMPLATES=(diamond fan3 fan4)
#TEMPLATES=(fan5)
TIMEOUT=10800

cd /home/almasri3/samiran/mewcp-gpu/

for temp in ${TEMPLATES[@]};
do
    TEMP_PATH="/home/almasri3/samiran/dataset/template_graphs/mtx/${temp}_template.mtx"
    #echo -e "Template Path: $TEMP_PATH"
    echo -e "\n\n************* Processing Template $temp **************"
    # echo -e "symopt"
    ./build/exe/src/main.cu.exe -g $GRAPH -o full -t $TEMP_PATH -w -d 3 -m $TASK -p $MODE #| grep -Ei "preprocessing|count|HD|LD"
    # echo -e "baseline"
    # /usr/bin/timeout $TIMEOUT ./build/exe/src/baseline.cu.exe -g $GRAPH -t $TEMP_PATH -d 3 -m $TASK -o full -p $MODE | grep -Ei "preprocessing|count"

    #./build/exe/src/main.cu.exe -g ../dataset/com-youtube.bel -t ../dataset/template_graphs/mtx/cq5_template.mtx -d 1 -m sgm -o full -p edge
    #/usr/bin/timeout $TIMEOUT ./build/exe/src/main.cu.exe -g $GRAPH -d 1 -m $TASK -o full -t $TEMP_PATH -p $MODE #| grep -Ei "preprocessing|count"
done
