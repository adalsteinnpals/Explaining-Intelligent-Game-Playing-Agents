# serialize all models in folder

folder=$1

mkdir -p production_models/$folder/serialized_models
for file in production_models/$folder/checkpoints/epoch=*.ckpt
do
    epochnumber=$(basename $file .ckpt | cut -d'=' -f2)
    file_name="nn-epoch$epochnumber.nnue"
    python serialize.py $file production_models/$folder/serialized_models/$file_name
done