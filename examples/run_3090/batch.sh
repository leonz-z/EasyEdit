for i in {0..1}; do
    echo "Running script $i"
    nohup bash examples/run_3090/$i.sh >logs/bash/2024-8-22-gpu$i-1.log 2>&1 &
done