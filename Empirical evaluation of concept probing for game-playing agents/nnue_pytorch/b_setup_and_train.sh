git clone -b development https://github.com/adalsteinnpals/nnue-pytorch

git clone https://github.com/lucasart/c-chess-cli.git
git clone https://github.com/michiguel/Ordo.git
git clone https://github.com/official-stockfish/Stockfish.git


# compile stockfish
cd Stockfish
git checkout 564456a6a824bfca26d6d9af5b35a055eb9fc6c2
cd src
make -j build ARCH=x86-64-modern


cd ../..


# compile c-chess-cli
cd c-chess-cli
python make.py



cd ..


# compile Ordo 
cd Ordo
make
sudo make install

cd ..



# download data
cd nnue-pytorch
bash script_install_requirements.sh


python script_get_training_data.py

for i in "$@"
do
bash train_model.sh $i
done

