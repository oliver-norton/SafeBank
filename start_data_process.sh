#!/bin/bash

# Start tmux session
tmux new-session -d -s my_session

# Window 1: Start Python environment and PostgreSQL
tmux send-keys -t my_session:0 "source ~/SafeBank_env/bin/activate" C-m
tmux send-keys -t my_session:0 "free -h" C-m
tmux send-keys -t my_session:0 "sudo service postgresql start" C-m
tmux send-keys -t my_session:0 "free -h" C-m

# Window 2: Start Zookeeper
tmux new-window -t my_session -n zookeeper
tmux send-keys -t my_session:zookeeper "zookeeper-server-start.sh /opt/kafka_2.13-3.9.0/config/zookeeper.properties" C-m

# Window 3: Start Kafka
tmux new-window -t my_session -n kafka
tmux send-keys -t my_session:kafka "kafka-server-start.sh /opt/kafka_2.13-3.9.0/config/server.properties" C-m

# Window 4: Start producer script
tmux new-window -t my_session -n producer
tmux send-keys -t my_session:producer "source ~/SafeBank_env/bin/activate && python3 ./2-data-stream/b_stream_data/producer_chunks.py" C-m

# Window 5: Start consumer script
tmux new-window -t my_session -n consumer
tmux send-keys -t my_session:consumer "source ~/SafeBank_env/bin/activate && python3 ./2-data-stream/c-preprocess-classify/consumer_preprocess.py" C-m

# Attach to the session
tmux attach-session -t my_session
