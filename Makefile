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
TARGET3 = $(EXEDIR)/create_haar_param
TARGET4 = $(EXEDIR)/train_cascadeclassifier
TARGET5 = $(EXEDIR)/test_haar_adaboost
TARGET6 = $(EXEDIR)/facedetector
TEST1   = $(EXEDIR)/test_integral

OBJ1 = $(OBJDIR)/extract_haar.o $(OBJDIR)/haar.o
OBJ2 = $(OBJDIR)/train_haar_adaboost.o $(OBJDIR)/adaboost.o $(OBJDIR)/haar.o
OBJ3 = $(OBJDIR)/create_haar_param.o $(OBJDIR)/haar.o
OBJ4 = $(OBJDIR)/train_cascadeclassifier.o $(OBJDIR)/cascadeclassifier.o $(OBJDIR)/adaboost.o $(OBJDIR)/haar.o
OBJ5 = $(OBJDIR)/test_integral.o $(OBJDIR)/haar.o
OBJ6 = $(OBJDIR)/test_haar_adaboost.o $(OBJDIR)/adaboost.o $(OBJDIR)/haar.o
OBJ7 = $(OBJDIR)/facedetector.o $(OBJDIR)/cascadeclassifier.o $(OBJDIR)/haar.o $(OBJDIR)/adaboost.o

all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6) $(TEST1)

$(TARGET1): $(OBJ1)
	$(CC) $(LIBS) -o $(TARGET1) $^

$(TARGET2): $(OBJ2)
	$(CC) $(LIBS) -o $(TARGET2) $^

$(TARGET3): $(OBJ3)
	$(CC) $(LIBS) -o $(TARGET3) $^

$(TARGET4): $(OBJ4)
	$(CC) $(LIBS) -o $(TARGET4) $^

$(TARGET5): $(OBJ6)
	$(CC) $(LIBS) -o $(TARGET5) $^

$(TARGET6): $(OBJ7)
	$(CC) $(LIBS) -o $(TARGET6) $^

$(TEST1): $(OBJ5)
	$(CC) $(LIBS) -o $(TEST1) $^

$(OBJDIR)/extract_haar.o: $(SRCDIR)/extract_haar.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/create_haar_param.o: $(SRCDIR)/create_haar_param.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/train_haar_adaboost.o: $(SRCDIR)/train_haar_adaboost.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/train_cascadeclassifier.o: $(SRCDIR)/train_cascadeclassifier.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/cascadeclassifier.o: $(SRCDIR)/cascadeclassifier.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/facedetector.o: $(SRCDIR)/facedetector.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/adaboost.o: $(SRCDIR)/adaboost.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/haar.o: $(SRCDIR)/haar.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/test_haar_adaboost.o: $(SRCDIR)/test_haar_adaboost.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/test_integral.o: ./test/test_integral.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6) $(TEST1) $(OBJDIR)/*.o
