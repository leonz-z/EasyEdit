for i in {0..3}; do
    echo "$i.sh"
    nohup bash examples/run_type5_70/$i.sh >logs/bash/2024-8-23-gpu$i-1.log 2>&1 & echo $!
done