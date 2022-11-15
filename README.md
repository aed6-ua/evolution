# Evolution
A neuroevolution simulation framework

## Pre-requisites
This library requires the FreeGLUT library. To install it, download the latest stable release from http://freeglut.sourceforge.net/index.php#download.

Unpack the tarball:
```bash
tar xzvf freeglut-X.Y.Z.tar.gz
```
Create a build directory:
```bash
cd freeglut-X.Y.Z
mkdir build
cd build
```
Run cmake to build the Makefile:
```bash
cmake ..
```
Finally run make and install:
### Linux:
```bash
make
sudo make install
```
### Windows MinGW
For mingw, you'll need to use a terminal with administrator privileges and then run:
```bash
make
make install
```
Then copy the files from C:\Program Files(x86)\freeglut\ into C:\msys64\mingw64

## Build
To build this library go into the extracted folder and create a build directory:
```bash
mkdir build
cd build
```
Run cmake to build the Makefile:

### Linux
```bash
cmake ..
```
Finally just run make to build the executable:
```bash
make
```
### Windows MinGW
For mingw run cmake with the following command:
```bash
cmake .. -G "MinGW Makefiles"
```

## Usage
Just run the executable with the name of the simulation to run:
```bash
./evolution follow [-s]
```
