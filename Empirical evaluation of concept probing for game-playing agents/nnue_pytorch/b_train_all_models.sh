

# Loop over all input arguments

for i in "$@"
do
bash b_train_model.sh $i
done

