# Makefile for G2P Transformer Tester
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -march=native
TESTER = g2p_tf_tester

all: $(TESTER)

$(TESTER): g2p_tf_tester.cpp
	$(CXX) $(CXXFLAGS) -o $(TESTER) $<

clean:
	rm -f $(TESTER) g2p_tf_model

.PHONY: all clean


