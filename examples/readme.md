# Usage
- Start the talker vllm service first
```
MODEL_PATH=YOUR_MODEL_PATH
python talker/talker_vllm_server.py --model ${MODEL_PATH}/talker --gpu-memory-utilization 0.1 --port 8816
```
- Run demo
```
export PYTHONPATH=./:$PYTHONPATH
python3 examples/vllm_demo.py
```
