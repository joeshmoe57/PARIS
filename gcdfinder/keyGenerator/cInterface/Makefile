CC=g++
CFLAGS=-g -O3 -Wall
ALL=getKeysTest getKeys.o sqlite3.o cfmemopen.o

all: $(ALL)

getKeysTest: getKeysTest.cpp getKeys.o  sqlite3.o libcrypto.a libssl.a cfmemopen.o
	$(CC) $(CFLAGS) -o $@ $^

getKeys.o: getKeys.cpp sqlite3.o libcrypto.a libssl.a cfmemopen.o
	$(CC) $(CFLAGS) -o $@ -c $^

sqlite3.o: sqlite3.c
	gcc $(CFLAGS) -o $@ -c $^

cfmemopen.o: cfmemopen.c
	$(CC) $(CFLAGS) -o $@ -c $^

clean:
	rm -rf core* *.o *.gch $(ALL)
