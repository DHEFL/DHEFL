python3 partitionMNIST.py --n_parties=80 \
  --partition='noniid-#label4' \
  --beta=0.5\
  --datadir="MNIST/train.csv" \
  --outputdir="MNIST/80clients/" \
  --init_seed=0
read
