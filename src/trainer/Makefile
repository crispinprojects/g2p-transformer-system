# Makefile for G2P Transformer Trainer 
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -march=native
TARGET = g2p_tf_trainer

all: $(TARGET) 

$(TARGET): g2p_tf_trainer.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) $<


train: $(TARGET)
	./$(TARGET) cmudict-0.7b


clean:
	rm -f $(TARGET) g2p_tf_model

.PHONY: all clean train 


