import torch
from models.yolo import Model
from models.common import Conv, DWConv

from torch.autograd import Variable
import onnxruntime
import onnx
import h5py
import numpy as np

model_name="yolor-d6.pt"

def onnx_test(model,onnx_model,img):

    mod_name = model_name.replace('.pt', '.onnx')  # filename

    torch_out_zeros = model(img)
    ort_session = onnxruntime.InferenceSession(mod_name)
    
    def to_numpy(tensor):
        return tensor.detach().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out_zeros[0]), ort_outs[0], rtol=5e-03, atol=1e-04)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    x_zeros_np=to_numpy(img)
    torch_out_zeros_np=to_numpy(torch_out_zeros[0])
    print(torch_out_zeros[0].shape)
    hfile = model_name.replace('.pt', '.h5')  # filename
    hf = h5py.File(hfile, 'w')

    hf.create_dataset('input_zero', data=x_zeros_np)
    hf.create_dataset('output_zero', data=torch_out_zeros_np)
    hf.close()
    


ckpt = torch.load(model_name, map_location='cpu')  # load checkpoint

model = Model("models/"+ model_name.replace('.pt', '.yaml') , ch=3, nc=80).to('cpu')  # create

state_dict = ckpt['model'].float().state_dict()  # to FP32

model.load_state_dict(state_dict, strict=False)  # load

# Compatibility updates
for m in model.modules():
    if type(m) in [torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ReLU6]:
        m.inplace = True  # pytorch 1.7.0 compatibility
    elif type(m) is Conv:
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

model.eval()
print(model)

img = torch.zeros(1, 3, 1280,1280,device='cpu')
f = model_name.replace('.pt', '.onnx')  # filename

torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'])

# Checks
onnx_model = onnx.load(f)  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
onnx_test(model,onnx_model,img)