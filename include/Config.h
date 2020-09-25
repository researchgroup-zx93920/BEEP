#pragma once

#include "utils.cuh"
#include "defs.cuh"



int     opterr = 1,             /* if error message should be printed */
optind = 1,             /* index into parent argv vector */
optopt,                 /* character checked for validity */
optreset;               /* reset getopt */
char* optarg;                /* argument associated with option */

#define BADCH   (int)'?'
#define BADARG  (int)':'
#define EMSG    ""

/*
* getopt --
*      Parse argc/argv argument vector.
*/
int
getopt(int nargc, char* const nargv[], const char* ostr)
{
    static char* place = EMSG;              /* option letter processing */
    const char* oli;                        /* option letter list index */

    if (optreset || !*place) {              /* update scanning pointer */
        optreset = 0;
        if (optind >= nargc || *(place = nargv[optind]) != '-') {
            place = EMSG;
            return (-1);
        }
        if (place[1] && *++place == '-') {      /* found "--" */
            ++optind;
            place = EMSG;
            return (-1);
        }
    }                                       /* option letter okay? */
    if ((optopt = (int)*place++) == (int)':' ||
        !(oli = strchr(ostr, optopt))) {
        /*
        * if the user didn't specify '-' as an option,
        * assume it means -1.
        */
        if (optopt == (int)'-')
            return (-1);
        if (!*place)
            ++optind;
        if (opterr && *ostr != ':')
            (void)printf("illegal option -- %c\n", optopt);
        return (BADCH);
    }
    if (*++oli != ':') {                    /* don't need argument */
        optarg = NULL;
        if (!*place)
            ++optind;
    }
    else {                                  /* need an argument */
        if (*place)                     /* no white space */
            optarg = place;
        else if (nargc <= ++optind) {   /* no arg */
            place = EMSG;
            if (*ostr == ':')
                return (BADARG);
            if (opterr)
                (void)printf("option requires an argument -- %c\n", optopt);
            return (BADCH);
        }
        else                            /* white space */
            optarg = nargv[optind];
        place = EMSG;
        ++optind;
    }
    return (optopt);                        /* dump back option letter */
}

struct Config {

    const char* srcGraph;
    const char* dstGraph; //for conversion
    MAINTASK mt;
    bool printStats;
    int deviceId;
    AllocationTypeEnum allocation;
    OrientGraphByEnum orient;
    bool verbosity;
    int k;
    bool sortEdges;
};

static MAINTASK parseMainTask(const char* s)
{
    if (strcmp(s, "tsv-mtx"))
        return CONV_TSV_MTX;

    if (strcmp(s, "tsv-bel"))
        return CONV_TSV_BEL;

    if (strcmp(s, "txt-bel"))
        return CONV_TXT_BEL;

    if (strcmp(s, "mtx-bel"))
        return CONV_MTX_BEL;

    if (strcmp(s, "bel-mtx"))
        return CONV_BEL_MTX;

    if (strcmp(s, "tc"))
        return TC;

    if (strcmp(s, "kcore"))
        return KCORE;


    if (strcmp(s, "ktruss"))
        return KTRUSS;

    if (strcmp(s, "kclique"))
        return KCLIQUE;

    if (strcmp(s, "cd"))
        return CROSSDECOMP;

    fprintf(stderr, "Unrecognized -mt option (Main TASK): %s\n", s);
    exit(0);
}

static const char* asString(MAINTASK mt) {
    switch (mt) {
    case CONV_TSV_MTX:            return "tsv-mtx";
    case CONV_TSV_BEL:            return "tsv-bel";
    case CONV_TXT_BEL:            return "txt-bel";
    case CONV_MTX_BEL:            return "mtx-bel";
    case CONV_BEL_MTX:            return "bel-mtx";
    case TC:                return "tc";
    case KCORE:            return "kcore";
    case KTRUSS:            return "ktruss";
    case KCLIQUE:            return "kclique";
    case CROSSDECOMP:   return "cd";

    default:
        fprintf(stderr, "Unrecognized main task\n");
        exit(0);
    }
}

static OrientGraphByEnum parseOrient(const char* s)
{
    if (strcmp(s, "full"))
        return None;
    if (strcmp(s, "upper"))
        return Upper;
    if (strcmp(s, "lower"))
        return Lower;
    if (strcmp(s, "degree"))
        return Degree;
    if (strcmp(s, "degen"))
        return Degeneracy;

    fprintf(stderr, "Unrecognized -o option (Graph Orient): %s\n", s);
    exit(0);
}

static AllocationTypeEnum parseAllocation(const char* s)
{
    if (strcmp(s, "unified"))
        return unified;
    if (strcmp(s, "gpu"))
        return gpu;
    if (strcmp(s, "cpu"))
        return cpuonly;


    fprintf(stderr, "Unrecognized -a option (Allocation): %s\n", s);
    exit(0);
}



static void usage() {
    fprintf(stderr,
        "\nUsage:  ./build/exe/src/main.cu.exe [options]"
        "\n"
        "\nOptions:"
        "\n    -sf <Src graph FileName>       Name of file with input graph (default = )"
        "\n    -df <Dst graph FileName>       Name of file with dst graph only for conversion (default = )"
        "\n    -d <Device Id>                      GPU Device Id (default = 0)"
        "\n    -mt <MainTask>     Name of the task to perform (default = TC)"
        "\n    -x                   Print Graph Stats         "
        "\n    -o <orientGraph>       How to orient undirected to directed graph (default = full)"
        "\n    -a <allocation>        Data allocation on GPU (default = unified)"
        "\n    -s <allocation>        Sort Read Edges by src then dst (default = false)"
        "\n    -h                       Help"
        "\n"
        "\n");
}


static Config parseArgs(int argc, char** argv) {
    Config config;
    config.srcGraph = "D:\\graphs\\as-Skitter2.bel";
    config.dstGraph = "D:\\graphs\\as-Skitter2.bel";
    config.deviceId = 0;
    config.mt = TC;
    config.printStats = false;
    config.orient = None;
    config.allocation = unified;
    config.k = 3;
    config.sortEdges = false;
#ifndef __VS__
    int opt;
    while ((opt = getopt(argc, argv, "sf:df:d:mt:x:o:a:k:h:v:s")) >= 0) {
        switch (opt) {
        case 'sf': config.srcGraph = optarg;                           break;
        case 'df': config.dstGraph = optarg;                     break;
        case 'd': config.deviceId = atoi(optarg);                           break;
        case 'mt': config.mt = parseMainTask(optarg);                       break;
        case 'x': config.printStats = true;                                 break;
        case 'o': config.orient = parseOrient(optarg);                      break;
        case 'a': config.allocation = parseAllocation(optarg);              break;
        case 'v': config.verbosity = atoi(optarg);                     break;
        case 'k': config.k = atoi(optarg);
        case 's': config.sortEdges = true;
        case 'h': usage(); exit(0);
        default: fprintf(stderr, "\nUnrecognized option!\n");
            usage(); exit(0);
        }
    }
#endif
    return config;
}

static void printConfig(Config config) 
{
    printf("    Graph: %s\n", config.srcGraph);
    printf("    DST Graph: %s\n", config.dstGraph);
    printf("        Device Id = %u\n", config.deviceId);
    printf("        Main Task = %s\n", asString(config.mt));
    printf("    k: %u\n", config.k);
}