.PHONY: run

run: main
	./main

main: main.c
	$(CC) -Wall -Wextra -O2 $^ -lOpenCL -luring -o $@
