# -*- coding: utf-8 -*-
import argparse
import onnxruntime
import importlib

def parse_path(path: str) -> str:
    if path.endswith("/"):
        path = path[:-1]
    module_str = path.replace("/", ".").replace(".py", "")
    return module_str

def parse_args():
    parser = argparse.ArgumentParser(description="Run an ONNX model with given input data.")
    parser.add_argument("--model_path", type=str, help="Path to the ONNX model file.")
    parser.add_argument("--input_path", type=str, help="Path to the input data")
    return parser.parse_args()

def run_onnx_model():
    """
    Runs an ONNX model with the given input data.

    Args:
        model_path (str): Path to the ONNX model file.
        input_data (dict): A dictionary where keys are input names and values are numpy arrays.

    Returns:
        dict: A dictionary where keys are output names and values are numpy arrays.
    """
    args = parse_args()
    model_path = args.model_path
    input_path = args.input_path
    module_str = parse_path(input_path)
    # try:
    #     module = importlib.import_module(module_str)
    #     input_data = module.input_data  # Assuming the module has an attribute `input_data`
    # except ModuleNotFoundError as e:
    #     raise ValueError(f"Could not load input data from {input_path}: {e}")
    # except AttributeError as e:
    #     raise ValueError(f"The module {module_str} does not have an attribute 'input_data': {e}")

    import numpy as np
    input_data = dict(
        user_embedding_input=np.random.randn(2, 64).astype(np.float32),
        item_embedding_input=np.random.randn(2, 188, 64).astype(np.float32),
        selected_last_n_input=np.array([188, 10], dtype=np.int64)
    )
    # Create an ONNX Runtime session
    session = onnxruntime.InferenceSession(model_path)

    # Run the model
    outputs = session.run(None, input_data)

    # Get output names
    output_names = [output.name for output in session.get_outputs()]

    # Create a dictionary of outputs
    output_dict = {name: output for name, output in zip(output_names, outputs)}
    print(f"ONNX model outputs: {output_dict}")

    return output_dict

if __name__ == "__main__":
    run_onnx_model()