
corpus="ami"
declare -a models=("s2s" "rnn" "cl" "sr")
declare -a models=("s2s")
nsize=${#models[@]}
for (( i=1; i<${nsize}+1; i++ ));
do
    python script_bin/train_model.py --trainer --sentence-limit 20 --train-inputs data/${corpus}/inputs/train/ --train-labels data/${corpus}/labels/train/ --valid-inputs data/${corpus}/inputs/valid/ --valid-labels data/${corpus}/labels/valid/  --valid-refs data/${corpus}/human-abstracts/valid/ --weighted --gpu 0 --model models/${corpus}_${models[$i-1]}_rnn --results scores/${corpus}_${models[$i-1]}_rnn --seed 12345678 --emb --pretrained-embeddings ~/workspace/embeddings/glove.6B.200d.txt --enc rnn --ext ${models[$i-1]} --bidirectional
done


#declare -a models=("cl" "sr")
#nsize=${#models[@]}
#for (( i=1; i<${nsize}+1; i++ ));
#do
#    python script_bin/train_model.py --trainer --train-inputs data/${corpus}/inputs/train/ --train-labels data/${corpus}/labels/train/ --valid-inputs data/${corpus}/inputs/valid/ --valid-labels data/${corpus}/labels/valid/  --valid-refs data/${corpus}/human-abstracts/valid/ --weighted --gpu 1 --model models/${corpus}_${models[$i-1]}_rnn --results scores/${corpus}_${models[$i-1]}_rnn --seed 12345678 --emb --pretrained-embeddings ~/workspace/embeddings/glove.6B.200d.txt --enc rnn --ext ${models[$i-1]}
#done
