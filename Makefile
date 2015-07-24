.PHONY: clean

cubic_eigen: cubic_eigen.cpp
	g++ --std=c++11 -mavx -O3 $^ -o $@

cubic_eigen.S: cubic_eigen.cpp
	g++ --std=c++11 -mavx -O3 $^ -S -o $@

clean:
	rm -f cubic_eigen cubic_eigen.S

