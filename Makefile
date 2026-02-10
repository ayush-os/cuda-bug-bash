NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80

EXERCISES = ex_1_1 ex_1_2 ex_1_3 ex_1_4 ex_1_5

all: $(EXERCISES)

ex_1_%: ex_1_%.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $

clean:
	rm -f $(EXERCISES)

test_1_1: ex_1_1
	./ex_1_1
	compute-sanitizer ./ex_1_1

test_1_2: ex_1_2
	./ex_1_2
	ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct_of_peak_sustained_elapsed ./ex_1_2

test_1_3: ex_1_3
	./ex_1_3
	compute-sanitizer ./ex_1_3

test_1_4: ex_1_4
	./ex_1_4
	ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./ex_1_4

test_1_5: ex_1_5
	./ex_1_5
	compute-sanitizer --tool racecheck ./ex_1_5

.PHONY: all clean test_%