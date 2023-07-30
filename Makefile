# Makefile for compiling C++ source files

# Compiler to use
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -stdlib=libc++

# Target executable name
TARGET = xor_nn

# Source files
SRCS = main.cpp util/dataset.cpp util/neuron.cpp util/net.cpp util/layer.cpp

# Object files derived from source files
OBJS = $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Rule for compiling object files from C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for linking object files to create the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# Clean rule to remove generated files
clean:
	rm -f $(OBJS) $(TARGET)
