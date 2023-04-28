subBuild: build
	cd build && make

profBuild: build_prof
	cd build_prof && make

build_prof:
	mkdir build_prof && cd build_prof && cmake ../sources

build:
	mkdir build && cd build && cmake ../sources

clean:
	rm -rf build
