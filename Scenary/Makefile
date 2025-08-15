CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
TARGET = main
SOURCES = main.cpp
HEADERS = $(wildcard *.hpp)

# Generate compilation database
compile_commands.json: Makefile
	@echo "Generating compilation database..."
	@echo '[' > compile_commands.json
	@echo '  {' >> compile_commands.json
	@echo '    "directory": "$(shell pwd)",' >> compile_commands.json
	@echo '    "command": "$(CXX) $(CXXFLAGS) -c $(SOURCES)",' >> compile_commands.json
	@echo '    "file": "$(SOURCES)"' >> compile_commands.json
	@echo '  }' >> compile_commands.json
	@echo ']' >> compile_commands.json

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) compile_commands.json

.PHONY: clean

