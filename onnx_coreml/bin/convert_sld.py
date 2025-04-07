from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import click
from onnx import onnx_pb
from onnx_coreml import convert
from typing import Text, IO
import coremltools as ct


@click.command(
    help='convert ONNX model to CoreML model',
    context_settings={
        'help_option_names': ['-h', '--help']
    }
)
@click.argument('onnx_model', type=click.File('rb'))
@click.option('-o', '--output', required=True,
              type=str,
              help='Output path for the CoreML *.mlmodel file')
def onnx_to_coreml(onnx_model, output):  # type: (IO[str], str) -> None
    onnx_model_proto = onnx_pb.ModelProto()
    onnx_model_proto.ParseFromString(onnx_model.read())

    hidden_state_prev_shape_lower = [1, 1, 128]
    hidden_state_prev_shape_upper = [1, 8, 128]

    cell_state_prev_shape_lower = [1, 1, 128]
    cell_state_prev_shape_upper = [1, 8, 128]

    ldmks_2d_shape_lower = [1, 1, 95, 2]
    ldmks_2d_shape_upper = [8, 1, 95, 2]

    ldmks_2d_norm_prev_shape_lower = [1, 1, 95, 2]
    ldmks_2d_norm_prev_shape_upper = [8, 1, 95, 2]

    width_height_shape_lower = [1, 2]
    width_height_shape_upper = [8, 2]
    shape_range = {
        'hidden_state_prev': {'lower': hidden_state_prev_shape_lower, 'upper': hidden_state_prev_shape_upper},
        'cell_state_prev': {'lower': cell_state_prev_shape_lower, 'upper': cell_state_prev_shape_upper},
        'ldmks_2d': {'lower': ldmks_2d_shape_lower, 'upper': ldmks_2d_shape_upper},
        'ldmks_2d_norm_prev': {'lower': ldmks_2d_norm_prev_shape_lower, 'upper': ldmks_2d_norm_prev_shape_upper},
        'width_height': {'lower': width_height_shape_lower, 'upper': width_height_shape_upper}
    }

    hidden_state_prev_shape = tuple(hidden_state_prev_shape_lower)
    cell_state_prev_shape = tuple(cell_state_prev_shape_lower)
    ldmks_2d_shape = tuple(ldmks_2d_shape_lower)
    ldmks_2d_norm_prev_shape = tuple(ldmks_2d_norm_prev_shape_lower)
    width_height_shape = tuple(width_height_shape_lower)
    coreml_model = convert(onnx_model_proto, minimum_ios_deployment_target='13', 
                           onnx_coreml_input_shape_map={'hidden_state_prev': hidden_state_prev_shape, 'cell_state_prev': cell_state_prev_shape, 'ldmks_2d': ldmks_2d_shape, 'ldmks_2d_norm_prev': ldmks_2d_norm_prev_shape, 'width_height': width_height_shape},
                           onnx_coreml_output_shape_map={'is_signing': [1, 1], 'sign_sigma': [1, 1], 'hidden_state': [1, 1, 128], 'cell_state': [1, 1, 128], 'ldmks_2d_norm_next_prev': [1, 1, 95, 2], 'ldmks_v_max': [1, 1]},
                           onnx_coreml_input_shape_range=shape_range)
    coreml_model.save(output)

if __name__ == '__main__':
    onnx_to_coreml()
