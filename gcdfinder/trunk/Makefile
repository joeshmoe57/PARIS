CC=gcc
GPP=g++
NVCC=nvcc
NVFLAGS=
MEMFLAGS=-Xptxas="-v"
ARCHFLAGS=-arch=compute_30 -code=sm_30
CFLAGS=-g -O3 -Wall -std=c99
PFLAGS=-g -O3 -Wall -Werror
LFLAGS=-lpthread -ldl
ALL=seqGCD cudaGCD
INCLUDES = -I.
#TMPDIR= /tmp/rsa
TMPDIR=~/Documents/Thesis
RUN=
SEQOUT=$(TMPDIR)/seq_out$(RUN).txt
CUDAOUT=$(TMPDIR)/cuda_out$(RUN).txt
COMPARE_SIZE=200

all: $(ALL)

compare_time: time_seq time_cuda

time_seq: seqGCD
	(time ./seqGCD 200) > $(SEQOUT) 2> seq_time$(RUN)_200
	(time ./seqGCD 400) > $(SEQOUT) 2> seq_time$(RUN)_400
	(time ./seqGCD 800) > $(SEQOUT) 2> seq_time$(RUN)_800
	(time ./seqGCD 1000) > $(SEQOUT) 2> seq_time$(RUN)_1000

time_cuda: cudaGCD
	(time ./cudaGCD 200) > $(CUDAOUT) 2> cuda_time$(RUN)_200
	(time ./cudaGCD 400) > $(CUDAOUT) 2> cuda_time$(RUN)_400
	(time ./cudaGCD 800) > $(CUDAOUT) 2> cuda_time$(RUN)_800
	(time ./cudaGCD 1000) > $(CUDAOUT) 2> cuda_time$(RUN)_1000

time_seq1: seqGCD
	(time ./seqGCD 2000) > $(SEQOUT) 2> seq_time$(RUN)_2000
	(time ./seqGCD 4000) > $(SEQOUT) 2> seq_time$(RUN)_4000
	(time ./seqGCD 10000) > $(SEQOUT) 2> seq_time$(RUN)_10000

time_cuda1: cudaGCD
	(time ./cudaGCD 2000) > $(CUDAOUT) 2> cuda_time$(RUN)_2000
	(time ./cudaGCD 4000) > $(CUDAOUT) 2> cuda_time$(RUN)_4000
	(time ./cudaGCD 10000) > $(CUDAOUT) 2> cuda_time$(RUN)_10000

verify: $(SEQOUT) $(CUDAOUT)
	diff -q $(SEQOUT) $(CUDAOUT)

$(SEQOUT): seqGCD 
	(time ./seqGCD $(COMPARE_SIZE) > $(SEQOUT) 2> seq_time
	-chmod 660 $(SEQOUT)

$(CUDAOUT): cudaGCD 
	(time ./cudaGCD $(COMPARE_SIZE) )> $(CUDAOUT) 2> cuda_time
	-chmod 660 $(CUDAOUT) 

resuse_20: cudaGCD.cu 
	$(NVCC) $(MEMFLAGS) $(ARCHFLAGS) $^

getKeys.o: getKeys.cpp sqlite3.o libcrypto.a libssl.a cfmemopen.o
	$(GPP) $(PFLAGS) -o $@ -c $^

sqlite3.o: sqlite3.c
	$(CC) $(CFLAGS) -o $@ -c $^ $(LFLAGS)

cfmemopen.o: cfmemopen.c
	$(GPP) $(PFLAGS) -o $@ -c $^ $(LFLAGS)

cudaGCD: cudaGCD.cu cudaGCD.h keyMath.o globals.h getKeys.o keyOutput.o sqlite3.o libcrypto.a libssl.a cfmemopen.o hrt.c
	$(NVCC) $(ARCHFLAGS) -o $@ $(INCLUDES) cudaGCD.cu keyMath.o getKeys.o sqlite3.o libcrypto.a libssl.a cfmemopen.o hrt.c $(LFLAGS)

seqGCD: seqGCD.cpp keyMath.o globals.h getKeys.o keyOutput.o sqlite3.o libcrypto.a libssl.a cfmemopen.o hrt.c
	$(GPP) $(PFLAGS) -o $@ seqGCD.cpp keyMath.o getKeys.o sqlite3.o libcrypto.a libssl.a cfmemopen.o hrt.c $(LFLAGS)

keyOutput.o: keyOutput.cpp keyOutput.h
	$(GPP) $(CFLAGS) -c $<

validate: validate.c keyMath.o 
	$(CC) $(CFLAGS) -o $@ $^

keyMath.o: keyMath.c keyMath.h
	$(CC) $(CFLAGS) -c $<

clean:
	-rm -rf core* *.o *.gch $(ALL)
	-rm $(SEQOUT) $(CUDAOUT) 
