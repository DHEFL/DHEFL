#!/bin/bash
python3 deleteFL_Enc.py
echo "Starting dealer"
echo "Starting server"
python3 dealer.py &
sleep 10
python3 server.py &
sleep 50

range=$(python3 -c "from config_FL import num_client; print(num_client())")
new_range=$((range-1))

for i in $(seq 0 $new_range); do
    echo "Starting client $i"
    python3 FL_client.py --partition=${i} &
done

# Wait for all background processes to complete
read