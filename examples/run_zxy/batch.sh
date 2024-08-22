for i in {1..7}; do
    echo "Running script $i"
    nohup bash examples/run_zxy/$i.sh >logs/bash/2024-8-22-gpu$i-0.log 2>&1 &
done