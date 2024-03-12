CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra

SRC = transform_best_basis.cpp
OBJ = $(SRC:.cpp=.o)
EXECUTABLE = best_basis_app

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXECUTABLE)
