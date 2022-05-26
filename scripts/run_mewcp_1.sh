#!/bin/bash

TASK=$1
GRAPH=$2
MODE=$3

# echo "Input Graph: $GRAPH"
# echo "ProcessBy: $MODE"
# echo "Task: $TASK"

# TEMPLATES=(diamond) # fan4 wheel5 cq6m1 cq6) # cq7 cq8 cq9 cq10) 
# TEMPLATES=(fan3)
# TEMPLATES=(cq5m1 house pyramid fan3 cq6m1)
TEMPLATES=(tri diamond cq4 cq5m1 cq5 house pyramid fan3 cq6m1 fan4 wheel5 cq6)
TEMPLATES+=(cq7m1 cq7 cq8m1 cq8 cq9m1 cq9)
# TEMPLATES=(wheel52)
TIMEOUT=6000

cd ../

for temp in ${TEMPLATES[@]};
do
    # TEMP_PATH="/home/almasri3/samiran/dataset/template_graphs/mtx/${temp}_template.mtx"
    TEMP_PATH="/home/almasri3/samiran/dataset/template_graphs/mtx/${temp}_template.mtx"
    #echo -e "Template Path: $TEMP_PATH"
    echo -e "************* Processing Template $temp **************"
    # echo -e "symopt"
    # /usr/bin/timeout $TIMEOUT ./buildD/exe/src/main.cu.exe -g $GRAPH -o full -t $TEMP_PATH -d 3 -m $TASK -p $MODE |grep -Ei "preprocessing|count|HD|LD"
    # /usr/bin/timeout $TIMEOUT ./buildDS/exe/src/main.cu.exe -g $GRAPH -o full -t $TEMP_PATH -d 3 -m $TASK -p $MODE |grep -Ei "preprocessing|count|HD|LD"
    # /usr/bin/timeout $TIMEOUT ./buildDR/exe/src/main.cu.exe -g $GRAPH -o full -t $TEMP_PATH -d 3 -m $TASK -p $MODE |grep -Ei "preprocessing|count|HD|LD"
    # /usr/bin/timeout $TIMEOUT ./buildDSR/exe/src/main.cu.exe -g $GRAPH -o full -t $TEMP_PATH -d 3 -m $TASK -p $MODE |grep -Ei "preprocessing|count|HD|LD"
    /usr/bin/timeout $TIMEOUT ./build/exe/src/main.cu.exe -g $GRAPH -o full -t $TEMP_PATH -d 3 -m $TASK -p $MODE |grep -Ei "kernel time"
done
