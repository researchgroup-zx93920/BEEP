# BEEP
BEEP is a GPU-accelerated subgraph enumerator that performs DFS with dynamic load balancing using a multi-producer-multi-consumer worker queue.
Refer to the paper with all the details here: https://dl.acm.org/doi/10.1145/3605573.3605653

To install and run BEEP please follow this process:

## Minimum Hardware Requirements
1. A X86_64 CPU with 4 cores, 512 GB disk space.
2. CUDA Enabled GPU with compute capability >= 70 (Volta architecture or higher), VRAM >= 32 GB*
3. For single-device version: RAM >= 128 GB
4. For multi-device version: RAM >= 256 GB
5. CUDA toolkit version 9 or higher. (Can work with older versions with separately installed CUB)

*The repo has been developed and tested on GPUs with at least 32GB VRAM and may run out of memory on lower-capacity cards with large graphs.


## Installing Nauty
BEEP uses Nauty as a query preprocessor. Reference: https://pallini.di.uniroma1.it/

Download and build nauty:

```wget https://pallini.di.uniroma1.it/nauty27r1.tar.gz``` (Download tar file)

```tar -xvf nauty27r1.tar.gz``` (Untar into ${BEEP_HOME}/nauty)

```cd ${BEEP_HOME}/nauty```

```./configure```     (add executable permission if needed) 

```make``` (Compile Nauty)

## Compiling BEEP
The makefile is configured to automatically detect the GPU architecture and compile the code with relevant flags.
To compile run: ```make clean all```


## Getting Data Graphs
We have tested and reported the performance BEEP on the following data graphs from the SNAP repository:

com-youtube, cit-patents, soc-pokec, com-orkut, com-friendster

BEEP uses ```.bel``` format for reading data graphs.
We have provided a converter in this repo that can covert the popular formats like .mtx, .txt, .tsv to .bel

Below is a stepwise illustration to convert cit-patents from .txt to .bel (Use ${EXEC} -help for details)
1. Download the undirected cit-patents data graph from snap: ``` wget https://snap.stanford.edu/data/cit-Patents.txt.gz```
2. Unzip: ```gunzip cit-Patents.txt.gz```
3. Run: ```./build/exec/src/main.cu.exe -g <source-graph> -r <dest-graph> -m txt-bel```

## Getting Template/Query Graphs:
BEEP supports query graphs with a central node (see Figure 5 in the paper).
The query graph has to be input in ```.mtx``` format. (Use a similar process to convert any valid query graph to .mtx)
The query graph is treated as a directed graph, hence the mtx file should have forward and backward edges.

For example, a .mtx file for Triangle would be:
```
3 3 6
0 1
0 2
1 0
1 2
2 0
2 1
```

## Running BEEP
Once the query and data graphs are ready, BEEP can be minimally run using

```./build/exe/src/main.cu.exe -g <data-graph-path> -t <query-graph-path> -d <device-id> -m sgm ```

For running on Multiple devices use: (Note: this will execute the code on devices 0, 1, .. num-devices)

```./build/exe/src/main.cu.exe -g <data-graph-path> -t <query-graph-path> -n <num-devices> -m sgm ```

## Help
For more details on the executable run:
```/build/exe/src/main.cu.exe -help```

```
Usage:  ./build/exe/src/main.cu.exe [options]

Options:
    -g <Src graph FileName>       Name of file with input graph (default = )
    -r <Dst graph FileName>       Name of file with dst graph only for conversion (default = )
    -t <Pattern graph filename>   Name of file with template/pattern graph only for subgraph matching
    -w <is small graph>         Use global memory to allocate the undirected graph, otherwise zerocopy memory
    -d <Device Id>                      GPU Device Id (default = 0)
    -m <MainTask>     Name of the task to perform (default = TC)  [For subgraph enumeration "sgm" for subgraph counting "sgc"
    -x                   Print Graph Stats         
    -o <orientGraph>       How to orient undirected to directed graph (default = full)
    -a <allocation>        Data allocation on GPU (default = unified)
    -v <verbosity>        Verbosity
    -k <k>        k
    -s <sort>        Sort Read Edges by src then dst (default = false)
    -p <processBy>        Process by node or edge (default = node)
    -e <process element>         Granulaity of element processor (default = t) <t: Thread, w: warp, b: block, g: grid>
    -q <kclique specs>           Specify KC Specs: (o4b --> graph orient, Partition Size = 4, binary encoeding)  (p4n --> pivoting, Partition Size = 4, NO binary encoding) default: o8b
    -h                       Help
    -c <cutoff>          Used for subgraph matching (To switch between node-per-block vs edge-per-block <deprecated>
```

We support open-source code, feel free to contact the author for any questions.

Author info: Samiran Kawtikwar (samiran2@illinois.edu)

https://samiran-kawtikwar.github.io/

## Reference
Samiran Kawtikwar, Mohammad Almasri, Wen-Mei Hwu, Rakesh Nagi, and Jinjun Xiong. 2023. BEEP: Balanced Efficient subgraph Enumeration in Parallel. In Proceedings of the 52nd International Conference on Parallel Processing (ICPP '23). Association for Computing Machinery, New York, NY, USA, 142–152. https://doi.org/10.1145/3605573.3605653

