

model=train_setting_7

b_serialize_all_training_models.sh $model
python p_create_all_activations.py --model_string $model 