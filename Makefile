# Makefile

CC = g++
CFLAGS = -O
INCPATH += `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv`

EXEDIR = ./bin
OBJDIR = ./obj
SRCDIR = ./src

TARGET1 = $(EXEDIR)/extract_haar

OBJ1 = $(OBJDIR)/extract_haar.o $(OBJDIR)/haar.o $(OBJDIR)/filelib.o

all: $(TARGET1)

$(TARGET1): $(OBJ1)
	$(CC) $(LIBS) -o $(TARGET1) $^

$(OBJDIR)/extract_haar.o: $(SRCDIR)/extract_haar.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/haar.o: $(SRCDIR)/haar.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/filelib.o: $(SRCDIR)/filelib.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET1) $(OBJDIR)/*.o
