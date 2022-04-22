# defaults start
{"defaults": {
    "ops": {
        "is_output_quantized": "True"
    },
    "params": {
        "is_quantized": "True"
    },
    "strict_symmetric": "False",
    "unsigned_symmetric": "True",
    "per_channel_quantization": "False"
},
# defaults end
# params start
    "params": {
        "bias": {
            "is_quantized": "False"
        }
    },
# params end
# op_type start
    "op_type": {
      "Squeeze": {
        "is_output_quantized": "False"
      },
    # op_type end
# supergroups start
    "supergroups": [
        {
            "op_list": ["Conv", "Relu"]
        },
        {
            "op_list": ["Conv", "Clip"]
        },
        {
            "op_list": ["Add", "Relu"]
        },
        {
            "op_list": ["Gemm", "Relu"]
        }
    ],
# supergroups end
# model_input start
    "model_input": {
        "is_input_quantized": "True"
    },
# model_input end
# model_output start
    "model_output": {}}
# model_output end
