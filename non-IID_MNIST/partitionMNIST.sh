python3 partitionMNIST.py --n_parties=10 \
  --partition='noniid-#label4' \
  --beta=0.5\
  --datadir="MNIST/train.csv" \
  --outputdir="MNIST/10clients/" \
  --init_seed=0
$SHELL
