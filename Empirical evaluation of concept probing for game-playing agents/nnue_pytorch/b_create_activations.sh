# for loop run create_activation_dataset2.py


model_name=$1
model_file=model
ckpt_name="epoch=399-step=2441600.ckpt"

# define variable length
#python create_fen_dataset.py
declare -i length=10000
for i in {0..100}
do  
    echo "python p_create_activation_dataset.py --start $((i*length)) --length $length --model_string $model_name --model_file $model_file --ckpt_name $ckpt_name"
    python p_create_activation_dataset.py --start $((i*length)) --length $length --model_string $model_name --model_file $model_file --ckpt_name $ckpt_name
done
