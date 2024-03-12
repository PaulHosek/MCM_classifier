CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra

SRC = Matrix.cpp main.cpp
OBJ = $(SRC:.cpp=.o)
EXECUTABLE = MatrixApp

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXECUTABLE)
