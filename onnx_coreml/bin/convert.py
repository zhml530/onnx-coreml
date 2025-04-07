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

    input_img_shape_lower = [1, 3, 64, 64]
    input_img_shape_upper = [1, 3, 2560, 2560]

    batch_indices_shape_lower = [1]
    batch_indices_shape_upper = [512]

    windows_shape_lower = [1, 4]
    windows_shape_upper = [512, 4]

    ori_img_size_shape_lower = [1, 2]
    ori_img_size_shape_upper = [8, 2]

    range_map = {
        'input_img': {'lower': input_img_shape_lower, 'upper': input_img_shape_upper},
        'batch_indices': {'lower': batch_indices_shape_lower, 'upper': batch_indices_shape_upper},
        'windows': {'lower': windows_shape_lower, 'upper': windows_shape_upper},
        'ori_img_size': {'lower': ori_img_size_shape_lower, 'upper': ori_img_size_shape_upper}
    }
    
    input_img_shape = (1, 3, 1080, 1920)
    batch_indices_shape = (1, )
    windows_shape = (1, 4)
    ori_img_size_shape = (1, 2) 
    coreml_model = convert(onnx_model_proto, minimum_ios_deployment_target='13', 
                           onnx_coreml_input_shape_map={'input_img': input_img_shape, 'batch_indices': batch_indices_shape, 'windows': windows_shape, 'ori_img_size': ori_img_size_shape},
                           onnx_coreml_output_shape_map={'ld2bbox/ld_2d_bboxes': [1, 4], 'sigmas': [1, 95], 'ldmks_2d': [1, 95, 2]},
                           onnx_coreml_input_shape_range=range_map)

    coreml_model.save(output)

if __name__ == '__main__':
    onnx_to_coreml()
