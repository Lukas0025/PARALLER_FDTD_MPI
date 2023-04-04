subBuild: build
	cd build && make

build:
	mkdir build && cd build && cmake ../sources

clean:
	rm -rf build
