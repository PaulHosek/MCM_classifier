g++ -std=c++11 -O3 -Wall ./src/*.cpp -o ./bin/saa.exe
cd bin
saa.exe 50 -i Big5-IPCall_VSmean_Ne5
pause
saa.exe 50 -i Big5-IPCall_VSmean_Ne5 -g -s 
pause