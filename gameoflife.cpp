#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <unistd.h>

using namespace std;

#define GRID_N 10
#define GRID_M 10
#define GHOST 1
#define GRID_N_GHOST (GRID_N + 2*GHOST)
#define GRID_M_GHOST (GRID_M + 2*GHOST)

void printBoard(array<bool, GRID_N_GHOST*GRID_M_GHOST>* board)
{
    for (int iy = GHOST; iy < GRID_M + GHOST; iy++) {
        for (int ix = GHOST; ix < GRID_N + GHOST; ix++) {
            if ((*board)[iy*GRID_N_GHOST + ix] == true) {
                cout << "0";
            } else {
                cout << ".";
            }
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, char *argv[])
{
    cl::sycl::queue q;
    auto range = cl::sycl::range<2>{GRID_N_GHOST, GRID_M_GHOST};
    string filename;
    int iterations;

    try {
        if (argc < 2) {
            throw "No iteration number or board file provided";
        }
        //  else if (argc < 3) {
        //     throw "No board provided";
        // }
        
        iterations = atoi(argv[1]);
        // filename = argv[2];
    } catch (const char* e) {
        cout << e << endl;
        exit(1);
    }
    
    array <bool, GRID_N_GHOST*GRID_M_GHOST> board{};
    array <bool, GRID_N_GHOST*GRID_M_GHOST>* boardPtr = &board;
    array <bool, GRID_N_GHOST*GRID_M_GHOST> boardNext{};
    array <bool, GRID_N_GHOST*GRID_M_GHOST>* boardNextPtr = &boardNext;

    board[(4+GHOST)*GRID_N_GHOST + (5+GHOST)] = true;
    board[(5+GHOST)*GRID_N_GHOST + (5+GHOST)] = true;
    board[(6+GHOST)*GRID_N_GHOST + (6+GHOST)] = true;
    board[(5+GHOST)*GRID_N_GHOST + (6+GHOST)] = true;
    board[(6+GHOST)*GRID_N_GHOST + (5+GHOST)] = true;
    board[(9+GHOST)*GRID_N_GHOST + (9+GHOST)] = true;
    
    cout << "Step 0:\n";
    printBoard(boardPtr);

    for(int i = 0; i < iterations; i++) {
        
        {
            auto boardBuffer = cl::sycl::buffer<bool, 2>(boardPtr->data(), range);
            auto nextBuffer = cl::sycl::buffer<bool, 2>(boardNextPtr->data(), range);
    
            q.submit([&](cl::sycl::handler& cgh) {
                auto boardAcc = boardBuffer.get_access<cl::sycl::access::mode::read>(cgh);
                auto nextAcc = nextBuffer.get_access<cl::sycl::access::mode::write>(cgh);

                cgh.parallel_for<class GameOfLife>(cl::sycl::range<2>(GRID_N_GHOST, GRID_M_GHOST), [=](cl::sycl::id<2> idx) {
                    int ix = idx[0];
                    int iy = idx[1];

                    if (ix > 0 && iy > 0 && ix < (GRID_N_GHOST - 1) && iy < (GRID_M_GHOST - 1)) {
                        int neighbors = 0;
                        neighbors += boardAcc[iy+1][ix-1];
                        neighbors += boardAcc[iy+1][ix];
                        neighbors += boardAcc[iy+1][ix+1];
                        neighbors += boardAcc[iy][ix-1];
                        neighbors += boardAcc[iy][ix+1];
                        neighbors += boardAcc[iy-1][ix-1];
                        neighbors += boardAcc[iy-1][ix];
                        neighbors += boardAcc[iy-1][ix+1];

                        if (neighbors < 2 || neighbors > 3) {
                            nextAcc[iy][ix] = 0;
                        } else if (neighbors == 3) {
                            nextAcc[iy][ix] = 1;
                        } else {
                            nextAcc[iy][ix] = boardAcc[iy][ix];
                        }
                    }
                });
            }).wait();
        }

        cout << "Step " << i+1 << ":\n";
        printBoard(boardNextPtr);
        auto tmp = boardPtr;
        boardPtr = boardNextPtr;
        boardNextPtr = tmp;
    }

    return 0;
}