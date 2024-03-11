NVCC = nvcc
TARGET_EXEC ?= a.out

ARCH := $(shell ./get_SM.sh)
BUILD_DIR ?= ./build
SRC_DIRS ?= ./src
INC_DIRS ?= 
EXE_DIR ?= $(BUILD_DIR)/exec

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu)
SRCS_NAMES := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s -or -name *.cu -printf "%f\n")
OBJS := $(SRCS:%=$(BUILD_DIR)/obj/%.o)
EXES := $(SRCS:%=$(BUILD_DIR)/exe/%.exe)
DEBUG_OBJS := $(SRCS:%=$(BUILD_DIR)/debug_objs/%.o)
DEBUG_EXES := $(SRCS:%=$(BUILD_DIR)/debug_exes/%.exe)
DEPS := $(OBJS:.o=.d)

#INCL_DIRS := $(shell find $(INC_DIRS) -type d) ./include $(FREESTAND_DIR)/include 
INCL_DIRS := #./include $(FREESTAND_DIR)/include 
INC_FLAGS := $(addprefix -I,$(INCL_DIRS))
LDFLAGS := -L./nauty/ -l:nauty.a -lcuda -lgomp
CPPFLAGS ?= $(INC_FLAGS) -g -Wall -pthread -MMD -MP -shared -fPIC -std=c++11 -O3 -mavx -ftree-vectorize -fopt-info-vec
CUDAFLAGS = $(INC_FLAGS) -g -w -Xcompiler -fopenmp -lineinfo -O3 -DCUDA -DNOT_IMPL -arch=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=compute_$(ARCH)
CUDADEBUGFLAGS = $(INC_DIRS) -g -w -G -std=c++11 -DCUDA -DNOT_IMPL -arch=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=compute_$(ARCH)

all: objs release_exes

dbg: debug_objs debug_exes

objs: $(OBJS)

release_exes: $(EXES)

debug_objs: $(DEBUG_OBJS)

debug_exes: $(DEBUG_EXES)

$(BUILD_DIR)/exe/%.exe: $(BUILD_DIR)/obj/%.o
	$(MKDIR_P) $(dir $@)
	$(NVCC) $< -o $@ $(LDFLAGS)

$(BUILD_DIR)/debug_exes/%.exe: $(BUILD_DIR)/debug_objs/%.o
	$(MKDIR_P) $(dir $@)
	$(NVCC) $< -o $@ $(LDFLAGS)

# assembly
$(BUILD_DIR)/obj/%.s.o: %.s
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) -c $< -o $@

# c source
$(BUILD_DIR)/obj/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# c++ source
$(BUILD_DIR)/obj/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# cuda source
$(BUILD_DIR)/obj/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

#cuda debug source
$(BUILD_DIR)/debug_objs/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(CUDADEBUGFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
