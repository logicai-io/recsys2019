flags = -std=c++14 

all: price.o events.o utils.o features.o scores.o counters.o extractor.o counters-user.o diff-feature.o
	g++ ${flags} -o ./build/events ./build/events.o utils.o
	g++ ${flags} -o ./build/price ./build/price.o features.o utils.o
	g++ ${flags} -o ./build/scores ./build/scores.o utils.o ./build/counters.o
	g++ ${flags} -o ./build/extractor ./build/extractor.o utils.o features.o
	g++ ${flags} -o ./build/diff-feature ./build/diff-feature.o utils.o ./build/counters-user.o

price.o: price.cpp
	g++ ${flags} -c price.cpp -o ./build/price.o

events.o: events.cpp
	g++ ${flags} -c events.cpp -o ./build/events.o

features.o: features.cpp
	g++ ${flags} -c features.cpp 

utils.o: utils.cpp
	g++ ${flags} -c utils.cpp 

scores.o: scores.cpp
	g++ ${flags} -c scores.cpp -o ./build/scores.o

counters.o: counters.cpp
	g++ ${flags} -c counters.cpp -o ./build/counters.o

counters-user.o: counters-user.cpp
	g++ ${flags} -c counters-user.cpp -o ./build/counters-user.o

diff-feature.o: diff-feature.cpp
	g++ ${flags} -c diff-feature.cpp -o ./build/diff-feature.o


extractor.o: extractor.cpp
	g++ ${flags} -c extractor.cpp -o ./build/extractor.o

clean:
	rm ./build/*
