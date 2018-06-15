g++ ./base/Base.cpp -fPIC -shared -o ./release/Base.so -pthread -O3 -march=native
cp ./release/Base.so $CONDA_PREFIX/lib/
echo "Base.so Copied to $CONDA_PREFIX/lib/"