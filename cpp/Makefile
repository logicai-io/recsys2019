flags = -std=c++1y

all: price.o events.o utils.o features.o scores.o counters.o extractor.o
	g++ ${flags} -o ./build/events ./build/events.o ./build/utils.o
	g++ ${flags} -o ./build/price ./build/price.o ./build/features.o ./build/utils.o
	g++ ${flags} -o ./build/scores ./build/scores.o ./build/utils.o ./build/counters.o
	g++ ${flags} -o ./build/extractor ./build/extractor.o ./build/utils.o ./build/features.o

price.o: price.cpp
	g++ ${flags} -c price.cpp -o ./build/price.o

events.o: events.cpp
	g++ ${flags} -c events.cpp -o ./build/events.o

features.o: features.cpp
	g++ ${flags} -c features.cpp -o ./build/features.o

utils.o: utils.cpp
	g++ ${flags} -c utils.cpp -o ./build/utils.o

scores.o: scores.cpp
	g++ ${flags} -c scores.cpp -o ./build/scores.o

counters.o: counters.cpp
	g++ ${flags} -c counters.cpp -o ./build/counters.o

extractor.o: extractor.cpp
	g++ ${flags} -c extractor.cpp -o ./build/extractor.o

clean:
	rm ./build/*
