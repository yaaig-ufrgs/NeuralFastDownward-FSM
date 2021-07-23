First install the required dependency:

`pip install tf2onnx`

Use `get_pb_params.py` to get the inputs and outputs of the model's graph
definition.

Then, for example, run:

`python -m tf2onnx.convert --graphdef 15_ocls_ns_ubal_h3_sigmoid_inter_gen_sat_drp0_Kall_pruneOff_9_fold_model.pb --output model.onnx --inputs input_1_1:0,input_2_1:0 --outputs 0:0`

Adapting it to your use case.
