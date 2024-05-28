
# declare ID variable
ID=3
stockfish_path=/home/ap/Stockfish
nnue_path=/home/ap/nnue-pytorch
ordo_path=../PycharmProjects/Ordo/ordo
cli_path=../PycharmProjects/c-chess-cli/c-chess-cli
depth=9
games=500


if [ ! -d "tournaments" ]; then
  mkdir tournaments
fi

if [ ! -d "tournaments/pgn" ]; then
  mkdir tournaments/pgn
fi

if [ ! -d "tournaments/ratings" ]; then
  mkdir tournaments/ratings
fi



$cli_path -rounds 1 -concurrency 30 -each option.Hash=32 option.Threads=1 timeout=20 tc=100.0+0.04 -games $games \
-openings file=./noob_3moves.epd order=random srand=64188023 \
-repeat \
-resign count=10 score=7000 \
-draw number=100 count=8 score=1 \
-pgn tournaments/pgn/out_test_$ID.pgn 3 \
-engine cmd=$stockfish_path/src/stockfish name=regular_model option.EvalFile=$nnue_path/models/nn-epoch499_regular_model.nnue depth=$depth \
-engine cmd=$stockfish_path/src/stockfish name=no_bishop_pair_advantage option.EvalFile=$nnue_path/models/nn-epoch499_no_bishop_pair_advantage.nnue depth=$depth \
-engine cmd=$stockfish_path/src/stockfish name=no_queen_advantage option.EvalFile=$nnue_path/models/nn-epoch499_no_queen_advantage.nnue depth=$depth \
-engine cmd=$stockfish_path/src/stockfish name=no_knight_advantage option.EvalFile=$nnue_path/models/nn-epoch499_no_knight_pair_advantage.nnue depth=$depth


$ordo_path -p tournaments/pgn/out_test_$ID.pgn -o tournaments/ratings/ratings_$ID.txt