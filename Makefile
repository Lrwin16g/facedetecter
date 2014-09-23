# Makefile

CC = g++
CFLAGS = -O
INCPATH += `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv`

EXEDIR = ./bin
OBJDIR = ./obj
SRCDIR = ./src

TARGET1 = $(EXEDIR)/extract_haar
TARGET2 = $(EXEDIR)/train_haar_adaboost

OBJ1 = $(OBJDIR)/extract_haar.o $(OBJDIR)/haar.o $(OBJDIR)/filelib.o
OBJ2 = $(OBJDIR)/train_haar_adaboost.o $(OBJDIR)/adaboost.o $(OBJDIR)/haar.o $(OBJDIR)/filelib.o

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(OBJ1)
	$(CC) $(LIBS) -o $(TARGET1) $^

$(TARGET2): $(OBJ2)
	$(CC) $(LIBS) -o $(TARGET2) $^

$(OBJDIR)/extract_haar.o: $(SRCDIR)/extract_haar.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/train_haar_adaboost.o: $(SRCDIR)/train_haar_adaboost.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/adaboost.o: $(SRCDIR)/adaboost.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/haar.o: $(SRCDIR)/haar.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/filelib.o: $(SRCDIR)/filelib.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET1) $(TARGET2) $(OBJDIR)/*.o
