# declare ID variable
depth=$1


ID=$2

games=$3


stockfish_path=/home/ap/Stockfish
nnue_path=/home/ap/nnue-pytorch
ordo_path=../PycharmProjects/Ordo/ordo
cli_path=../PycharmProjects/c-chess-cli/c-chess-cli


if [ ! -d "tournaments" ]; then
  mkdir tournaments
fi

if [ ! -d "tournaments/pgn" ]; then
  mkdir tournaments/pgn
fi

if [ ! -d "tournaments/ratings" ]; then
  mkdir tournaments/ratings
fi

# create list of possible models
possible_models=("train_setting_0" "train_setting_2" "train_setting_3" \
            "train_setting_4" \
            "train_setting_5" "train_setting_6"  \
            "train_setting_7" "train_setting_8" )


# make all engines in folder play against each other
for model in "${possible_models[@]}"
do
    file=production_models/$model/serialized_models/nn-epoch399-step.nnue
    engine_name=$(basename $file)
    echo $file
    # get second folder from path
    model=$(echo $file | cut -d'/' -f2)

    # make string -engine cmd=$stockfish_path/src/stockfish name=$engine_name option.EvalFile=$nnue_path/serialized_models/$model/$engine_name depth=$depth  depth=$depth"
    run_command="-engine cmd=$stockfish_path/src/stockfish name=$model option.EvalFile=$nnue_path/$file depth=$depth"
    # add run command to command string
    echo $run_command
    command="$command $run_command"
done

echo $command
    

$cli_path -rounds 1 -concurrency 30 -each option.Hash=32 option.Threads=1 timeout=20 tc=100.0+0.04 -games $games \
-openings file=./noob_3moves.epd order=random srand=0 \
-repeat \
-gauntlet \
-resign count=10 score=7000 \
-draw number=100 count=8 score=1 \
-pgn tournaments/pgn/out_test_"$ID"_depth_$depth.pgn 3 \
$command


$ordo_path -p tournaments/pgn/out_test_"$ID"_depth_$depth.pgn -c tournaments/ratings/ratings_$ID.csv